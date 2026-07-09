"""
Timeout + retry-with-backoff tested standalone, no LangGraph/LangChain in the
loop at all (DESIGN.md SS7's own stated rationale for keeping this testable
independently of the graph). Backoff delays are neutered so the test suite
doesn't actually wait seconds -- we're testing retry *logic*, not real time.

IMPORTANT: neutering must patch the `time` *name* inside src.agent.resilience
(one level), not `src.agent.resilience.time.sleep` (two levels) -- the latter
mutates the real, shared `time` module object (modules are singletons in
sys.modules), which would also neuter this test file's own real_time.sleep()
calls used to simulate slow work, making every "still too slow" scenario
silently instant and the test assert against the wrong call counts.
"""
import time as real_time
import types

import pytest

from src.agent.resilience import (
    ToolTimeoutError,
    dispatch_tool_with_retry,
    invoke_llm_with_retry,
    run_with_timeout,
)
from src.config import MAX_RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS


@pytest.fixture
def no_real_sleep(monkeypatch):
    monkeypatch.setattr("src.agent.resilience.time", types.SimpleNamespace(sleep=lambda seconds: None))


def test_run_with_timeout_returns_result_when_fast_enough():
    assert run_with_timeout(lambda: 42, timeout_seconds=1) == 42


def test_run_with_timeout_raises_tool_timeout_error_when_too_slow():
    with pytest.raises(ToolTimeoutError):
        run_with_timeout(lambda: real_time.sleep(0.3), timeout_seconds=0.02)


def test_dispatch_tool_with_retry_succeeds_first_try_without_retrying(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert dispatch_tool_with_retry(fn, timeout_seconds=1) == "ok"
    assert len(calls) == 1


def test_dispatch_tool_with_retry_retries_only_on_timeout_then_succeeds(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            real_time.sleep(0.3)  # too slow relative to the 0.02s timeout below
        return "ok"

    result = dispatch_tool_with_retry(fn, timeout_seconds=0.02)
    assert result == "ok"
    assert len(calls) == 2


def test_dispatch_tool_with_retry_gives_up_after_max_attempts(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        real_time.sleep(0.3)  # always too slow

    with pytest.raises(ToolTimeoutError):
        dispatch_tool_with_retry(fn, timeout_seconds=0.02)
    assert len(calls) == MAX_RETRY_ATTEMPTS


def test_dispatch_tool_with_retry_does_not_retry_non_timeout_exceptions(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad input -- won't get better on retry")

    with pytest.raises(ValueError):
        dispatch_tool_with_retry(fn, timeout_seconds=1)
    assert len(calls) == 1  # no retry attempted


def test_dispatch_tool_with_retry_backs_off_exponentially(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "src.agent.resilience.time", types.SimpleNamespace(sleep=lambda s: sleeps.append(s))
    )

    def fn():
        real_time.sleep(0.3)  # always too slow

    with pytest.raises(ToolTimeoutError):
        dispatch_tool_with_retry(fn, timeout_seconds=0.02)

    expected = [RETRY_BASE_DELAY_SECONDS * (2**i) for i in range(MAX_RETRY_ATTEMPTS - 1)]
    assert sleeps == expected


def test_invoke_llm_with_retry_succeeds_first_try(no_real_sleep):
    assert invoke_llm_with_retry(lambda: "ok", timeout_seconds=1) == "ok"


def test_invoke_llm_with_retry_retries_on_any_exception_then_succeeds(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient provider error")
        return "ok"

    result = invoke_llm_with_retry(fn, timeout_seconds=1)
    assert result == "ok"
    assert len(calls) == 3


def test_invoke_llm_with_retry_also_retries_on_timeout(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            real_time.sleep(0.3)
        return "ok"

    assert invoke_llm_with_retry(fn, timeout_seconds=0.02) == "ok"
    assert len(calls) == 2


def test_invoke_llm_with_retry_raises_last_exception_after_exhausting_attempts(no_real_sleep):
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        invoke_llm_with_retry(fn, timeout_seconds=1)
    assert len(calls) == MAX_RETRY_ATTEMPTS
