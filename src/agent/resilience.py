"""
Timeout + retry-with-backoff wrapping for LLM and tool calls (DESIGN.md SS7).
Framework-agnostic -- no LangGraph/LangChain dependency here -- so it's
testable standalone (tests/test_resilience.py), same rationale DESIGN.md
gives for keeping dispatch_tool_with_retry decoupled from the graph itself.
graph.py wires these in; it doesn't reimplement retry logic itself.

Two different retry policies, on purpose:
- Tool calls (dispatch_tool_with_retry): only a timeout is retried. A
  validation error or a real DB error will fail identically on a retry --
  retrying it just adds latency for no benefit. Only a timeout (a plausible
  transient hiccup on what should be a fast local SQLite/FAISS call) is
  worth a second attempt.
- LLM calls (invoke_llm_with_retry): any exception is retried. Real provider
  failures (429/5xx/network timeout) look different depending on SDK, and
  this module deliberately has no hard dependency on any one provider's
  exception types -- so instead of trying to classify "retryable", it retries
  broadly and relies on the downstream circuit-breaker fallback (graph.py) to
  bound the damage if retries are still exhausted.

Not built here: a provider-level circuit breaker that stays "open" across
requests after repeated failures -- each call gets its own fresh retry
budget. Acceptable at this scale; called out as a scaling tradeoff.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from src.config import MAX_RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=8)


class ToolTimeoutError(Exception):
    """Raised when a wrapped call exceeds its allotted timeout."""


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float) -> T:
    future = _executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        raise ToolTimeoutError(f"call exceeded {timeout_seconds}s timeout")


def dispatch_tool_with_retry(fn: Callable[[], T], timeout_seconds: float) -> T:
    """
    Runs fn with a hard timeout; retries ONLY on timeout, up to
    MAX_RETRY_ATTEMPTS, with exponential backoff (RETRY_BASE_DELAY_SECONDS *
    2**attempt). Any non-timeout exception propagates on the first attempt --
    see module docstring for why tool calls don't retry blindly.
    """
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return run_with_timeout(fn, timeout_seconds)
        except ToolTimeoutError:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))


def invoke_llm_with_retry(fn: Callable[[], T], timeout_seconds: float) -> T:
    """
    Runs fn with a hard timeout; retries on ANY exception (including
    timeout), up to MAX_RETRY_ATTEMPTS, with exponential backoff. Raises the
    last exception if still failing after all attempts -- the caller
    (graph.py's call_model) is responsible for catching that and routing to
    the circuit-breaker fallback rather than crashing the run.
    """
    last_exc: Exception = RuntimeError("unreachable: MAX_RETRY_ATTEMPTS must be >= 1")
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return run_with_timeout(fn, timeout_seconds)
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
    raise last_exc
