"""
Single factory (get_llm_client) so the future LangGraph orchestrator never
cares whether it's talking to the real Gemini API or a scripted mock -- both
sides expose the same interface a LangGraph agent node needs: `.bind_tools()`
and `.invoke()`.

Real side: ChatGoogleGenerativeAI already *is* a LangChain BaseChatModel, so
it natively implements that interface -- no wrapper class needed, just a thin
factory that reads config (single GOOGLE_API_KEY, same key search_guidelines'
embeddings use).

Mock side: MockLLMClient replays a pre-scripted sequence of AIMessages (one
per .invoke() call), so eval-harness golden-set fixtures (DESIGN.md SS10) can
pin down exactly what "the agent" does at each step -- deterministic, free,
and immune to model drift. Test code constructs MockLLMClient directly with
its own scripted conversation; get_llm_client() only wires it up when
LLM_PROVIDER=mock and a script was supplied, so `make run`/tests fail loudly
and immediately if someone asks for mock mode without scripting anything,
rather than silently doing nothing useful.
"""
from typing import List, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from src.config import GEMINI_MODEL, GOOGLE_API_KEY, LLM_PROVIDER

ScriptedResponse = Union[AIMessage, Exception]


class MockLLMClient:
    """
    Deterministic stand-in for a chat model. Construct with the exact
    sequence of responses "the agent" should produce, in order (a tool-call
    message per loop iteration, then a final content-only message to end the
    loop). Each .invoke() call consumes the next scripted entry, regardless
    of what messages are actually passed in -- the test author is expected to
    have scripted a conversation consistent with what the tools will
    actually return.

    A scripted entry may be an Exception instance instead of an AIMessage --
    .invoke() raises it instead of returning it, so tests can simulate a
    transient LLM-provider failure (and the resilience-wrapping retry/
    circuit-breaker path in src/agent/resilience.py + graph.py) without
    needing a real flaky provider.
    """

    def __init__(self, responses: Sequence[ScriptedResponse]):
        if not responses:
            raise ValueError("MockLLMClient needs at least one scripted response")
        self._responses: List[ScriptedResponse] = list(responses)
        self._call_count = 0

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs) -> "MockLLMClient":
        # No-op: responses are already fully-formed AIMessages (tool_calls
        # pre-populated where relevant), so there's no schema to bind.
        return self

    def invoke(self, messages: Sequence[BaseMessage], **kwargs) -> AIMessage:
        if self._call_count >= len(self._responses):
            raise IndexError(
                f"MockLLMClient exhausted: only {len(self._responses)} response(s) scripted, "
                f"but invoke() was called a {self._call_count + 1} time(s). "
                "Script one more response, or check the step budget/loop condition."
            )
        response = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


def get_llm_client(
    provider: Optional[str] = None,
    mock_responses: Optional[Sequence[AIMessage]] = None,
    api_key: Optional[str] = None,
) -> Union[BaseChatModel, MockLLMClient]:
    """
    provider/api_key default to config (LLM_PROVIDER / GOOGLE_API_KEY) but can
    be overridden explicitly -- callers/tests shouldn't need to monkeypatch
    env vars (which config.py already read at *its* import time, not the
    caller's) just to pick a provider or inject a key.
    """
    provider = provider or LLM_PROVIDER

    if provider == "mock":
        if not mock_responses:
            raise ValueError(
                "LLM_PROVIDER=mock requires mock_responses (a scripted list of AIMessages); "
                "none were supplied."
            )
        return MockLLMClient(mock_responses)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key or GOOGLE_API_KEY, temperature=0
    )
