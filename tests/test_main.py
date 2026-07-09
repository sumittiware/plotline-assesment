"""
Full HTTP request/response cycle through the async job pattern
(POST /copilot/run + GET /copilot/run/{run_id}), via FastAPI's TestClient --
this exercises the actual endpoint code, not a reimplementation of it.

TestClient runs BackgroundTasks synchronously as part of the same
request/response cycle (verified directly against a minimal FastAPI app
before relying on it here), so by the time client.post(...) returns, the
agent's background execution has already completed -- no sleeping/polling
needed to make these tests deterministic.

DB, guidelines-store, and LLM dependencies are monkeypatched directly onto
src.main's module-level names (get_guidelines_store, get_agent_llm,
connection_scope) rather than via FastAPI's dependency_overrides -- the
actual agent work now runs inside a BackgroundTasks callable, which executes
outside any HTTP request context, so there's no Depends()-bound request to
override against. This matches the project's established convention
(monkeypatch the consuming module's already-imported name -- see CLAUDE.md).
"""
import sqlite3
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agent.llm_client import MockLLMClient
from src.data_access.db import apply_schema
from src.main import app
from src.rag.ingest import build_index
from tests.fixtures.hashing_embeddings import HashingEmbeddings


@pytest.fixture
def conn():
    # check_same_thread=False: TestClient dispatches the endpoint (and the
    # background task it schedules) to a worker thread, but this fixture
    # creates the connection in the test's own (main) thread and shares that
    # one object across the whole test via the monkeypatched connection_scope
    # below. Production code never hits this -- there, connection_scope()
    # opens a fresh connection inside whichever thread actually calls it.
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    apply_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def guidelines_store(tmp_path_factory):
    index_path = str(tmp_path_factory.mktemp("guidelines_index"))
    return build_index(index_path=index_path, embeddings=HashingEmbeddings())


def _tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def client(conn, guidelines_store, monkeypatch):
    @contextmanager
    def _fake_connection_scope():
        # Reused (not closed) across every call within a test -- start_run's
        # initial INSERT, _execute_run's work, and get_run's poll all need to
        # see the SAME isolated in-memory DB, not three separate ones.
        yield conn

    monkeypatch.setattr("src.main.connection_scope", _fake_connection_scope)
    monkeypatch.setattr("src.main.get_guidelines_store", lambda: guidelines_store)
    yield TestClient(app)


def _override_llm(monkeypatch, responses):
    monkeypatch.setattr("src.main.get_agent_llm", lambda: MockLLMClient(responses))


def _run_and_poll(client, goal, idempotency_key=None):
    """POST + GET in one step -- the pattern nearly every test needs. Returns
    (post_response, get_response); the caller inspects whichever it needs."""
    payload = {"goal": goal}
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    post_response = client.post("/copilot/run", json=payload)
    run_id = post_response.json()["run_id"]
    get_response = client.get(f"/copilot/run/{run_id}")
    return post_response, get_response


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_copilot_run_returns_202_pending_immediately(client, monkeypatch):
    """The POST response itself must reflect state as of return time (before
    the background task runs) -- pending, no result yet -- regardless of how
    fast the test happens to execute the background task afterward."""
    _override_llm(monkeypatch, [AIMessage(content="Done.")])

    response = client.post("/copilot/run", json={"goal": "Some goal"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["result"] is None
    assert body["run_id"].startswith("run_")
    assert body["goal"] == "Some goal"


def test_copilot_run_happy_path_returns_campaign(client, monkeypatch):
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
                "image_prompt": "A friendly illustration of the app icon waving hello.",
                "guideline_citations": [],
            },
            "3",
        ),
        AIMessage(content="Created a push campaign for dormant users."),
    ]
    _override_llm(monkeypatch, responses)

    _, get_response = _run_and_poll(client, "Win back dormant users")

    assert get_response.status_code == 200
    envelope = get_response.json()
    assert envelope["status"] == "completed"
    assert envelope["error"] is None
    body = envelope["result"]
    assert body["campaign_id"] is not None
    assert body["status"] == "created"
    assert body["idempotent_replay"] is False
    assert body["channel"] == "push"
    assert body["message_copy"] == "come back!"
    # regression: image_prompt used to be persisted to the DB but never
    # surfaced in the API response at all (see CLAUDE.md) -- must round-trip.
    assert body["image_prompt"] == "A friendly illustration of the app icon waving hello."
    # segment_member_count: the actual snapshotted user_id count (campaign_segment_members),
    # not just segment_size -- 0 here since the isolated test DB has no seeded users, but
    # confirms the field is wired through from create_campaign's real result.
    assert body["segment_member_count"] == 0
    assert body["degraded"] is False
    assert body["final_message"] == "Created a push campaign for dormant users."
    assert body["steps_taken"] == 3
    assert len(body["trace"]) > 0
    # push is an external channel -- compliance citation should be auto-forced
    cited_topics = {c["topic_slug"] for c in body["guideline_citations"]}
    assert "consent-compliance-and-opt-outs" in cited_topics


def test_copilot_run_rejects_missing_goal(client, monkeypatch):
    _override_llm(monkeypatch, [AIMessage(content="unused")])
    response = client.post("/copilot/run", json={})
    assert response.status_code == 422


def test_get_run_for_unknown_run_id_returns_404(client):
    response = client.get("/copilot/run/run_does-not-exist")
    assert response.status_code == 404


def test_copilot_run_with_client_idempotency_key_is_stable_across_two_calls(client, monkeypatch):
    # query_segment first: the empty test DB always returns size=0, so
    # segment_size=0 below satisfies SS5.3 grounding.
    segment_call = _tool_call_message("query_segment", {"inactive_days_min": 14}, "0")
    payload = {
        "goal_text": "win back dormant users",
        "segment_def": {"inactive_days_min": 14},
        "segment_size": 0,
        "channel": "email",
        "message_copy": "come back!",
        "guideline_citations": [],
    }
    key = "http-idempotency-key-1"

    _override_llm(monkeypatch, [segment_call, _tool_call_message("create_campaign", payload, "1"), AIMessage(content="Done.")])
    _, first = _run_and_poll(client, "Win back dormant users", idempotency_key=key)

    _override_llm(monkeypatch, [segment_call, _tool_call_message("create_campaign", payload, "1"), AIMessage(content="Done.")])
    _, second = _run_and_poll(client, "Win back dormant users", idempotency_key=key)

    first_result = first.json()["result"]
    second_result = second.json()["result"]
    assert first_result["campaign_id"] == second_result["campaign_id"]
    assert first_result["idempotent_replay"] is False
    assert second_result["idempotent_replay"] is True


def test_copilot_run_step_budget_exceeded_returns_degraded_not_a_500(client, monkeypatch):
    from src.config import MAX_AGENT_STEPS

    responses = [
        _tool_call_message("search_guidelines", {"query": "brand voice"}, str(i))
        for i in range(MAX_AGENT_STEPS + 3)
    ]
    _override_llm(monkeypatch, responses)

    _, get_response = _run_and_poll(client, "Some goal")

    assert get_response.status_code == 200
    envelope = get_response.json()
    assert envelope["status"] == "completed"  # the AGENT degraded gracefully; the JOB still completed, didn't crash
    assert envelope["result"]["degraded"] is True
    assert envelope["result"]["campaign_id"] is None


def test_mock_provider_is_refused_for_live_traffic_when_llm_provider_is_mock(conn, guidelines_store, monkeypatch):
    """
    get_agent_llm() must refuse to serve real HTTP traffic in mock mode --
    mock mode needs a caller-scripted response list, meaningless over HTTP.
    This is checked synchronously in start_run BEFORE anything is enqueued,
    so it should still 503 immediately, same as before this endpoint became
    async/job-based. Patches src.main's already-imported LLM_PROVIDER
    constant directly rather than relying on the ambient env/.env state.
    """
    monkeypatch.setattr("src.main.LLM_PROVIDER", "mock")

    @contextmanager
    def _fake_connection_scope():
        yield conn

    monkeypatch.setattr("src.main.connection_scope", _fake_connection_scope)
    monkeypatch.setattr("src.main.get_guidelines_store", lambda: guidelines_store)
    # Deliberately do NOT override get_agent_llm -- exercise the real one.

    client = TestClient(app)
    response = client.post("/copilot/run", json={"goal": "Some goal"})
    assert response.status_code == 503


def test_copilot_run_survives_a_rejected_create_campaign_with_malformed_args(client, monkeypatch):
    """
    Regression test: a create_campaign call with a malformed arg (offer as a
    string instead of a dict) is correctly rejected inside the graph (a
    structured {"error": ...} tool result, not a crash) -- but if that
    rejected call is the LAST create_campaign attempt in the run, naively
    forwarding its raw input args into the result payload (which types offer
    as Optional[dict]) used to crash response construction itself with the
    exact same pydantic error, uncaught, as a real 500. Reproduces the exact
    payload a real Gemini call once produced.
    """
    responses = [
        _tool_call_message("query_segment", {"plan": "pro"}, "1"),
        _tool_call_message(
            "create_campaign",
            {
                "goal_text": "test",
                "segment_def": {"plan": "pro"},
                "segment_size": 0,
                "channel": "push",
                "message_copy": "hi",
                "offer": "20% off next purchase",  # invalid: should be a dict
                "guideline_citations": [],
            },
            "2",
        ),
        AIMessage(content="Handled it."),
    ]
    _override_llm(monkeypatch, responses)

    _, get_response = _run_and_poll(client, "test goal")

    assert get_response.status_code == 200
    envelope = get_response.json()
    assert envelope["status"] == "completed"  # the tool call failed; the JOB still completed cleanly
    body = envelope["result"]
    assert body["campaign_id"] is None
    assert body["channel"] is None
    assert body["offer"] is None
    assert body["degraded"] is False
    assert body["final_message"] == "Handled it."
    # The failure is still fully visible in the trace, just not surfaced as
    # if it were a successful campaign's data.
    create_campaign_entries = [t for t in body["trace"] if t.get("tool") == "create_campaign"]
    assert "error" in create_campaign_entries[0]["result_summary"].lower()
