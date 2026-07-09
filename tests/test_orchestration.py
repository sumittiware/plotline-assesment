"""
The LangGraph plan->act->observe loop, tested against MockLLMClient so
behavior is deterministic and free -- no network/API dependency (DESIGN.md
SS10 "deterministic mode"). Real tool objects are used throughout (built via
src/tools/registry.py against an isolated in-memory DB and a hashing-embedder
FAISS index), so this exercises the actual dispatch path, not a double mock.
"""
import json
import sqlite3
import types

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import src.agent.graph as graph_module
from src.agent.graph import run
from src.agent.llm_client import MockLLMClient
from src.config import MAX_AGENT_STEPS, MAX_RETRY_ATTEMPTS
from src.data_access.db import apply_schema
from src.rag.ingest import build_index
from src.tools.registry import build_tools
from tests.fixtures.hashing_embeddings import HashingEmbeddings


@pytest.fixture
def no_real_sleep(monkeypatch):
    """
    Resilience retries add real backoff sleeps (1s/2s/4s per src/config.py) --
    neuter them for tests that deliberately trigger retries, so the suite
    stays fast. Patches the `time` *name* inside src.agent.resilience (one
    level), not `.time.sleep` (two levels) -- the latter would mutate the
    real, shared `time` module object since modules are singletons.
    """
    monkeypatch.setattr("src.agent.resilience.time", types.SimpleNamespace(sleep=lambda seconds: None))


@pytest.fixture
def conn():
    # check_same_thread=False: tool calls now run inside resilience.py's
    # ThreadPoolExecutor (timeout enforcement), so this connection (created
    # on the test's thread) gets used from a worker thread -- sequential
    # access only, never concurrent, so this is safe.
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    apply_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def guidelines_store(tmp_path_factory):
    index_path = str(tmp_path_factory.mktemp("guidelines_index"))
    return build_index(index_path=index_path, embeddings=HashingEmbeddings())


@pytest.fixture
def tools(conn, guidelines_store):
    return build_tools(conn, guidelines_store)


def _tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_full_happy_path_creates_campaign_and_ends(tools):
    responses = [
        _tool_call_message("query_segment", {"inactive_days_min": 14}, "1"),
        _tool_call_message("search_guidelines", {"query": "winback churned users", "k": 3}, "2"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "win back dormant users",
                "segment_def": {"inactive_days_min": 14},
                "segment_size": 0,
                "channel": "push",
                "message_copy": "come back!",
                "guideline_citations": [],
            },
            "3",
        ),
        AIMessage(content="Created campaign for dormant users via push."),
    ]
    final_state = run("Win back dormant users", MockLLMClient(responses), tools)

    assert final_state["degraded"] is False
    assert final_state["steps_taken"] == 3
    assert final_state["messages"][-1].content == "Created campaign for dormant users via push."
    tool_names_called = [t["tool"] for t in final_state["trace"] if "tool" in t]
    assert "query_segment" in tool_names_called
    assert "search_guidelines" in tool_names_called
    assert "create_campaign" in tool_names_called


def test_step_budget_exceeded_routes_to_fallback_instead_of_looping_forever(tools):
    # Script more tool-call turns than MAX_AGENT_STEPS allows, and never end --
    # should_continue must cut the loop rather than let it run away.
    responses = [
        _tool_call_message("search_guidelines", {"query": "brand voice"}, str(i))
        for i in range(MAX_AGENT_STEPS + 3)
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    assert final_state["degraded"] is True
    assert final_state["steps_taken"] == MAX_AGENT_STEPS
    fallback_entries = [t for t in final_state["trace"] if t.get("type") == "fallback"]
    assert len(fallback_entries) == 1
    assert fallback_entries[0]["reason"] == "max_steps_exceeded"
    fallback_entries = [t for t in final_state["trace"] if t.get("type") == "fallback"]
    assert len(fallback_entries) == 1


def test_tool_exception_becomes_structured_error_not_a_crash(tools):
    # query_segment's pydantic schema rejects a wrong-typed value (a string
    # where an int is required) -- should surface as a structured error
    # message the LLM/loop can see, not raise and crash the run.
    responses = [
        _tool_call_message("query_segment", {"recency_days_max": "not-a-number"}, "1"),
        AIMessage(content="Handled the error."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "error" in tool_messages[0].content.lower()
    assert final_state["degraded"] is False


def test_create_campaign_for_push_channel_auto_cites_compliance_doc_even_if_llm_did_not(tools):
    responses = [
        _tool_call_message("query_segment", {}, "0"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "re-engage lapsed users",
                "segment_def": {},
                "segment_size": 0,
                "channel": "push",
                "message_copy": "hey, come back",
                "guideline_citations": [],  # LLM cited nothing
            },
            "1",
        ),
        AIMessage(content="Done."),
    ]
    final_state = run("Re-engage lapsed users", MockLLMClient(responses), tools)

    create_campaign_calls = [t for t in final_state["trace"] if t.get("tool") == "create_campaign"]
    assert len(create_campaign_calls) == 1
    cited_topic_slugs = {c["topic_slug"] for c in create_campaign_calls[0]["input"]["guideline_citations"]}
    assert "consent-compliance-and-opt-outs" in cited_topic_slugs

    forced_retrievals = [
        t for t in final_state["trace"] if t.get("input", {}).get("forced_compliance_retrieval")
    ]
    assert len(forced_retrievals) == 1


def test_create_campaign_for_in_app_channel_does_not_force_compliance_citation(tools):
    responses = [
        _tool_call_message("query_segment", {}, "0"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "nudge users to try a feature",
                "segment_def": {},
                "segment_size": 0,
                "channel": "in_app",
                "message_copy": "try our new feature",
                "guideline_citations": [],
            },
            "1",
        ),
        AIMessage(content="Done."),
    ]
    final_state = run("Nudge users", MockLLMClient(responses), tools)

    forced_retrievals = [
        t for t in final_state["trace"] if t.get("input", {}).get("forced_compliance_retrieval")
    ]
    assert len(forced_retrievals) == 0


def _last_create_campaign_result(state) -> dict:
    """Filters by tool name rather than assuming position/count -- the SS5.4
    compliance override can insert an extra search_guidelines ToolMessage
    ahead of create_campaign's for external channels."""
    matches = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "create_campaign"]
    return json.loads(matches[-1].content)


def test_create_campaign_is_idempotent_across_two_full_runs(tools):
    """Same goal/segment/channel run twice through the whole graph -> one campaign, not two."""
    # query_segment first: the empty test DB always returns size=0, so
    # segment_size=0 below satisfies SS5.3 grounding (create_campaign's
    # segment_size must match the last real query_segment result).
    segment_call = _tool_call_message("query_segment", {"inactive_days_min": 14}, "0")
    payload = {
        "goal_text": "win back dormant users",
        "segment_def": {"inactive_days_min": 14},
        "segment_size": 0,
        "channel": "email",
        "message_copy": "come back!",
        "guideline_citations": [],
    }
    responses_a = [segment_call, _tool_call_message("create_campaign", payload, "1"), AIMessage(content="Done.")]
    responses_b = [segment_call, _tool_call_message("create_campaign", payload, "1"), AIMessage(content="Done.")]

    state_a = run("Win back dormant users", MockLLMClient(responses_a), tools)
    state_b = run("Win back dormant users", MockLLMClient(responses_b), tools)

    result_a = _last_create_campaign_result(state_a)
    result_b = _last_create_campaign_result(state_b)
    assert result_a["campaign_id"] == result_b["campaign_id"]  # same campaign, not a new row
    assert result_a["idempotent_replay"] is False
    assert result_b["idempotent_replay"] is True


def test_client_supplied_idempotency_key_overrides_whatever_the_llm_used(tools):
    """A caller-supplied key (e.g. from the API request) always wins, even if
    the LLM's two create_campaign calls describe different-looking campaigns --
    the client key is the one source of truth for "is this a retry"."""
    key = "client-supplied-key-123"
    segment_call = _tool_call_message("query_segment", {"plan": "pro"}, "0")
    responses_a = [
        segment_call,
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal A",
                "segment_def": {"plan": "pro"},
                "segment_size": 0,
                "channel": "email",
                "message_copy": "version A",
                "guideline_citations": [],
            },
            "1",
        ),
        AIMessage(content="Done."),
    ]
    responses_b = [
        segment_call,
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal B -- different text",
                "segment_def": {"plan": "free"},
                "segment_size": 0,
                "channel": "email",
                "message_copy": "version B",
                "guideline_citations": [],
            },
            "1",
        ),
        AIMessage(content="Done."),
    ]

    state_a = run("goal A", MockLLMClient(responses_a), tools, idempotency_key=key)
    state_b = run("goal B", MockLLMClient(responses_b), tools, idempotency_key=key)

    result_a = _last_create_campaign_result(state_a)
    result_b = _last_create_campaign_result(state_b)
    assert result_a["campaign_id"] == result_b["campaign_id"]
    assert result_b["idempotent_replay"] is True


def test_llm_recovers_within_retry_budget_continues_normally(tools, no_real_sleep):
    """LLM fails twice (transient provider errors), succeeds on the 3rd
    attempt (within MAX_RETRY_ATTEMPTS) -- the run should proceed exactly as
    if it had succeeded first try, no degradation."""
    assert MAX_RETRY_ATTEMPTS == 3, "test assumes the default retry budget"
    responses = [
        RuntimeError("503 Service Unavailable"),
        RuntimeError("503 Service Unavailable"),
        AIMessage(content="All good, no campaign needed."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    assert final_state["degraded"] is False
    assert final_state["messages"][-1].content == "All good, no campaign needed."
    llm_failures = [t for t in final_state["trace"] if t.get("type") == "llm_failure"]
    assert llm_failures == []  # retries succeeded silently -- no failure surfaced in the trace


def test_llm_exhausts_retries_routes_to_fallback_with_no_prior_segment(tools, no_real_sleep):
    """LLM fails every attempt -- should_continue must route to fallback
    (circuit breaker), not crash the run, per DESIGN.md SS7."""
    responses = [RuntimeError("429 rate limited")] * MAX_RETRY_ATTEMPTS
    final_state = run("Some goal", MockLLMClient(responses), tools)

    assert final_state["degraded"] is True
    fallback_entries = [t for t in final_state["trace"] if t.get("type") == "fallback"]
    assert len(fallback_entries) == 1
    assert fallback_entries[0]["reason"] == "llm_provider_failure"
    assert "could not be completed" in final_state["messages"][-1].content.lower()
    assert "no segment was resolved" in final_state["messages"][-1].content.lower()


def test_llm_exhausts_retries_after_a_segment_was_resolved_uses_it_in_fallback(tools, no_real_sleep):
    """The circuit-breaker fallback should use the last successfully-resolved
    query_segment result (DESIGN.md SS7: "use the raw filters the last
    successful tool call resolved"), not just a bare apology."""
    responses = [
        _tool_call_message("query_segment", {"plan": "pro"}, "1"),
        *([RuntimeError("429 rate limited")] * MAX_RETRY_ATTEMPTS),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    assert final_state["degraded"] is True
    fallback_entries = [t for t in final_state["trace"] if t.get("type") == "fallback"]
    assert fallback_entries[0]["reason"] == "llm_provider_failure"
    final_content = final_state["messages"][-1].content.lower()
    assert "no segment was resolved" not in final_content
    assert "users, filters:" in final_content


def test_grounding_rejects_create_campaign_without_any_prior_query_segment(tools):
    """SS5.3: create_campaign must never persist a segment_size that wasn't
    actually produced by a real query_segment call in this conversation."""
    responses = [
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 5,
                "channel": "in_app",
                "message_copy": "hi",
                "guideline_citations": [],
            },
            "1",
        ),
        AIMessage(content="Handled the rejection."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    result = _last_create_campaign_result(final_state)
    assert "error" in result
    assert "no query_segment call" in result["error"].lower()
    assert final_state["degraded"] is False  # a rejected tool call, not a crash


def test_grounding_rejects_create_campaign_with_mismatched_segment_size(tools):
    responses = [
        _tool_call_message("query_segment", {}, "0"),  # empty test DB -> size=0
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 999,  # doesn't match the real result (0)
                "channel": "in_app",
                "message_copy": "hi",
                "guideline_citations": [],
            },
            "1",
        ),
        AIMessage(content="Handled the rejection."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    result = _last_create_campaign_result(final_state)
    assert "error" in result
    assert "999" in result["error"] and "0" in result["error"]


def test_grounding_rejects_create_campaign_with_fabricated_citation(tools):
    responses = [
        _tool_call_message("query_segment", {}, "0"),
        _tool_call_message("search_guidelines", {"query": "brand voice"}, "1"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 0,
                "channel": "in_app",
                "message_copy": "hi",
                "guideline_citations": [{"chunk_id": "made-up-chunk-that-was-never-retrieved"}],
            },
            "2",
        ),
        AIMessage(content="Handled the rejection."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    result = _last_create_campaign_result(final_state)
    assert "error" in result
    assert "made-up-chunk-that-was-never-retrieved" in result["error"]


def test_grounding_accepts_a_citation_from_the_forced_compliance_retrieval(tools):
    """The SS5.4 compliance override's forced search_guidelines call must
    itself count as a real retrieval for SS5.3 grounding purposes -- the LLM
    shouldn't be penalized for citing a chunk_id it never explicitly asked
    search_guidelines for. Fetch the real compliance chunk_id directly (via
    the same search_guidelines tool the graph itself uses) rather than
    hardcoding or guessing it."""
    search_guidelines_tool = [t for t in tools if t.name == "search_guidelines"][0]
    compliance_chunks = search_guidelines_tool.invoke(
        {"query": "consent, compliance and opt-outs", "k": 1, "topic_slug": "consent-compliance-and-opt-outs"}
    )
    compliance_chunk_id = compliance_chunks[0]["chunk_id"]

    responses = [
        _tool_call_message("query_segment", {}, "0"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 0,
                "channel": "push",
                "message_copy": "hi",
                # LLM proactively cites the compliance chunk itself, without
                # ever having called search_guidelines for it directly.
                "guideline_citations": [{"chunk_id": compliance_chunk_id}],
            },
            "1",
        ),
        AIMessage(content="Done."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)
    result = _last_create_campaign_result(final_state)
    assert "error" not in result
    assert result["status"] == "created"


def test_llm_self_corrects_after_grounding_rejection_and_campaign_is_eventually_created(tools):
    """Mirrors DESIGN.md SS5.3's "rejects/retries the completion" language --
    a rejected create_campaign call is just another structured tool error the
    LLM can see and adapt to, same as any other, not a terminal failure."""
    responses = [
        _tool_call_message("query_segment", {}, "0"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 42,  # wrong -- will be rejected
                "channel": "in_app",
                "message_copy": "hi",
                "guideline_citations": [],
            },
            "1",
        ),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "goal",
                "segment_def": {},
                "segment_size": 0,  # corrected to match the real result
                "channel": "in_app",
                "message_copy": "hi",
                "guideline_citations": [],
            },
            "2",
        ),
        AIMessage(content="Created after correcting the segment size."),
    ]
    final_state = run("Some goal", MockLLMClient(responses), tools)

    create_campaign_messages = [
        m for m in final_state["messages"] if isinstance(m, ToolMessage) and m.name == "create_campaign"
    ]
    assert len(create_campaign_messages) == 2
    assert "error" in json.loads(create_campaign_messages[0].content)
    second_result = json.loads(create_campaign_messages[1].content)
    assert second_result["status"] == "created"
    assert final_state["degraded"] is False


def test_run_generates_a_run_id_when_none_supplied(tools):
    responses = [AIMessage(content="Done.")]
    final_state = run("Some goal", MockLLMClient(responses), tools)
    assert final_state["run_id"].startswith("run_")


def test_run_respects_an_explicitly_supplied_run_id(tools):
    responses = [AIMessage(content="Done.")]
    final_state = run("Some goal", MockLLMClient(responses), tools, run_id="run_explicit_123")
    assert final_state["run_id"] == "run_explicit_123"


def test_run_passes_run_id_through_to_the_graph_invocation_config(tools, monkeypatch):
    """
    LangSmith tracing (DESIGN.md SS9, optional) correlates via run_name/
    metadata on the LangGraph invoke config -- this doesn't require a real
    LANGCHAIN_API_KEY to verify: if the config is wrong, tracing would be
    silently useless even with a real key, so the wiring itself is what
    matters here, checked by spying on build_graph's returned compiled graph.
    """
    captured = {}
    real_build_graph = graph_module.build_graph

    def spying_build_graph(llm, tools_):
        compiled = real_build_graph(llm, tools_)
        real_invoke = compiled.invoke

        def spying_invoke(state, config=None, **kwargs):
            captured["config"] = config
            return real_invoke(state, config, **kwargs)

        compiled.invoke = spying_invoke
        return compiled

    monkeypatch.setattr(graph_module, "build_graph", spying_build_graph)

    responses = [AIMessage(content="Done.")]
    run("Some goal", MockLLMClient(responses), tools, run_id="run_trace_test")

    assert captured["config"]["run_name"] == "run_trace_test"
    assert captured["config"]["metadata"]["app_run_id"] == "run_trace_test"
    assert "campaign-copilot" in captured["config"]["tags"]
