"""
Structured logging tested standalone -- captured via caplog rather than
inspecting real stdout, and independent of any FastAPI/graph machinery.
"""
import json
import logging

import pytest

from src.observability.logging import log_event, log_run, log_trace, logger


@pytest.fixture(autouse=True)
def ensure_propagation(monkeypatch):
    """
    caplog's default capture mechanism relies on a handler attached to the
    ROOT logger, reached via propagation. configure_logging() deliberately
    sets propagate=False on our logger (to avoid double-printing once a real
    handler is attached) -- harmless in production, but if any other test
    module has already imported src.main (which calls configure_logging() at
    import time) in this same pytest session, propagate=False sticks for the
    rest of the process and caplog silently captures nothing here. Force
    propagation back on just for these tests.
    """
    monkeypatch.setattr(logger, "propagate", True)


def test_log_event_emits_one_json_line_with_event_and_fields(caplog):
    with caplog.at_level(logging.INFO, logger="campaign_copilot"):
        log_event("something_happened", run_id="run_abc", extra=1)

    assert len(caplog.records) == 1
    parsed = json.loads(caplog.records[0].message)
    assert parsed["event"] == "something_happened"
    assert parsed["run_id"] == "run_abc"
    assert parsed["extra"] == 1
    assert "ts" in parsed


def test_log_run_emits_start_and_end_with_latency(caplog):
    with caplog.at_level(logging.INFO, logger="campaign_copilot"):
        with log_run("run_1", "some goal text") as fields:
            fields["degraded"] = False

    events = [json.loads(r.message) for r in caplog.records]
    assert [e["event"] for e in events] == ["run_start", "run_end"]
    assert events[0]["run_id"] == events[1]["run_id"] == "run_1"
    assert events[1]["degraded"] is False
    assert events[1]["latency_ms"] >= 0


def test_log_run_still_emits_end_line_if_the_block_raises(caplog):
    with caplog.at_level(logging.INFO, logger="campaign_copilot"):
        try:
            with log_run("run_2", "goal"):
                raise ValueError("boom")
        except ValueError:
            pass

    events = [json.loads(r.message) for r in caplog.records]
    assert [e["event"] for e in events] == ["run_start", "run_end"]


def test_log_trace_emits_one_line_per_entry_tagged_with_run_id(caplog):
    trace = [
        {"type": "llm_turn", "content": "", "tool_calls": ["query_segment"]},
        {"tool": "query_segment", "input": {}, "result_summary": "{}"},
    ]
    with caplog.at_level(logging.INFO, logger="campaign_copilot"):
        log_trace("run_3", trace)

    events = [json.loads(r.message) for r in caplog.records]
    assert len(events) == 2
    assert all(e["event"] == "trace_step" and e["run_id"] == "run_3" for e in events)
    assert events[0]["tool_calls"] == ["query_segment"]
    assert events[1]["tool"] == "query_segment"


def test_configure_logging_is_idempotent_and_does_not_duplicate_handlers():
    from src.observability.logging import configure_logging

    configure_logging()  # first call: adds a handler if none exists yet
    handlers_after_first_call = len(logger.handlers)
    assert handlers_after_first_call >= 1

    configure_logging()  # second call: must be a no-op, not a second handler
    assert len(logger.handlers) == handlers_after_first_call


def test_configure_logging_quiets_the_noisy_gemini_schema_conversion_warnings():
    """
    langchain_google_genai logs a WARNING for every pydantic-schema key
    (anyOf/default/title) it drops converting Optional fields to Gemini's
    function-calling schema -- cosmetic, not a correctness issue (the actual
    field type still comes through), but noisy: ~15 Optional fields across
    3 tools x once per bind_tools() call floods stdout. configure_logging()
    quiets that specific logger down to ERROR.
    """
    from src.observability.logging import configure_logging

    configure_logging()
    assert logging.getLogger("langchain_google_genai._function_utils").level == logging.ERROR
