"""
Minimal structured (JSON-lines) logging to stdout (DESIGN.md SS9). Every run
gets a run_id; each log line is a single JSON object, easy to pipe into any
log aggregator.

This deliberately complements, not duplicates, the `trace` already returned
in the API response: the response trace is the richer, complete artifact for
"why did this specific run do X" when the caller has it in hand; these log
lines exist for server-side visibility when they don't (a crashed client, a
dropped connection, or aggregate monitoring across many runs).

Not built: OpenTelemetry spans, LangSmith hosted tracing -- both explicitly
optional per DESIGN.md SS9, kept out to avoid scope creep on a 2-day
timeline. This stdlib-logging + JSON-lines approach needs zero extra
services and is enough to debug a bad run from raw stdout.
"""
import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

logger = logging.getLogger("campaign_copilot")


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (src/main.py). Idempotent -- safe to call
    more than once (e.g. once per test module) without duplicating handlers."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))  # the message IS the JSON line
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    # langchain-google-genai logs a WARNING for every pydantic-schema key
    # (anyOf/default/title -- how Optional fields are represented in Pydantic
    # v2's JSON schema) it drops while converting our tools' args_schema to
    # Gemini's more restrictive function-calling schema format. Cosmetic only
    # -- the actual field types still come through correctly (confirmed via
    # live testing) -- but noisy: ~15 Optional fields across 3 tools times
    # once per bind_tools() call floods stdout with dozens of near-duplicate
    # lines per run. Quieted at the source rather than filtering our own
    # logger, since these aren't emitted through campaign_copilot's logger.
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)


def log_event(event: str, **fields: Any) -> None:
    """One structured JSON line: {"event": ..., "ts": ..., **fields}."""
    logger.info(json.dumps({"event": event, "ts": time.time(), **fields}, default=str))


@contextmanager
def log_run(run_id: str, goal: str) -> Iterator[Dict[str, Any]]:
    """
    Wraps a /copilot/run request. Always logs a run_end line (with latency_ms
    and whatever the caller adds to the yielded dict), even if the request
    raises -- so a crash is still visible in the logs, not just a run_start
    with no matching end.
    """
    start = time.time()
    log_event("run_start", run_id=run_id, goal=goal[:200])
    result_fields: Dict[str, Any] = {}
    try:
        yield result_fields
    finally:
        log_event(
            "run_end",
            run_id=run_id,
            latency_ms=round((time.time() - start) * 1000, 1),
            **result_fields,
        )


def log_trace(run_id: str, trace: List[dict]) -> None:
    """One log line per trace entry (LLM turn / tool call / fallback),
    tagged with run_id -- per-step server-side visibility without
    duplicating the full trace's content into a single giant line."""
    for entry in trace:
        log_event("trace_step", run_id=run_id, **entry)
