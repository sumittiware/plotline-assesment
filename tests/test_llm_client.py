"""
llm_client tested in isolation from any orchestrator/graph code -- the graph
isn't built yet, so this only proves the shared interface (bind_tools/invoke)
behaves identically enough that a future graph node can't tell which backend
it's talking to.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm_client import MockLLMClient, get_llm_client


def test_mock_replays_scripted_responses_in_order():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "query_segment", "args": {}, "id": "1"}]),
        AIMessage(content="Done."),
    ]
    client = MockLLMClient(responses)
    first = client.invoke([HumanMessage(content="win back dormant users")])
    second = client.invoke([HumanMessage(content="win back dormant users"), first])

    assert first.tool_calls[0]["name"] == "query_segment"
    assert second.content == "Done."


def test_mock_raises_when_exhausted():
    client = MockLLMClient([AIMessage(content="only one")])
    client.invoke([HumanMessage(content="go")])
    with pytest.raises(IndexError):
        client.invoke([HumanMessage(content="go")])


def test_mock_requires_at_least_one_response():
    with pytest.raises(ValueError):
        MockLLMClient([])


def test_mock_bind_tools_is_a_noop_returning_something_invokable():
    client = MockLLMClient([AIMessage(content="ok")])
    bound = client.bind_tools([])
    result = bound.invoke([HumanMessage(content="go")])
    assert result.content == "ok"


def test_get_llm_client_mock_provider_without_responses_raises_clear_error():
    with pytest.raises(ValueError):
        get_llm_client(provider="mock")


def test_get_llm_client_mock_provider_returns_scripted_mock():
    responses = [AIMessage(content="scripted")]
    client = get_llm_client(provider="mock", mock_responses=responses)
    assert isinstance(client, MockLLMClient)
    assert client.invoke([]).content == "scripted"


def test_get_llm_client_gemini_provider_constructs_chat_google_generative_ai():
    """
    api_key passed explicitly rather than via env var + monkeypatch: config.py
    already read GOOGLE_API_KEY at its own import time, so setting the env var
    from a test wouldn't retroactively change the frozen module constant.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    client = get_llm_client(provider="gemini", api_key="test-key-not-real")
    assert isinstance(client, ChatGoogleGenerativeAI)


def test_get_llm_client_defaults_to_config_provider_when_none_given(monkeypatch):
    """
    provider=None -> falls back to whatever config.LLM_PROVIDER resolved to at
    import time. Patch src.agent.llm_client's already-imported LLM_PROVIDER
    constant directly rather than asserting on the ambient env/.env state --
    otherwise this test's outcome depends on whether a .env exists and what's
    in it (e.g. `make setup` writes LLM_PROVIDER=gemini into .env), which is
    exactly the kind of environment-fragile test that broke once already.
    """
    monkeypatch.setattr("src.agent.llm_client.LLM_PROVIDER", "mock")
    with pytest.raises(ValueError):
        get_llm_client()
