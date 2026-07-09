"""
The eval harness (DESIGN.md SS10): a small golden-set regression suite, not a
full eval framework, per the assignment's own framing ("the instinct to
measure quality in a non-deterministic system, not a full eval framework").

Two modes over the SAME fixtures (tests/fixtures/golden_set.py):
- Deterministic mode (runs in `make test`, free, instant): each fixture's
  goal replayed through a scripted MockLLMClient trajectory. This is a
  regression test of the orchestration+tools+grounding+compliance pipeline
  -- it does NOT evaluate LLM reasoning quality, since the trajectory is
  scripted, not produced. Property assertions are strict since we control
  every input.
- Live mode (opt-in via RUN_LIVE_EVAL=1, skipped by default -- costs real
  API quota and isn't deterministic): the SAME goal texts run through the
  REAL Gemini client end-to-end. This is what actually evaluates reasoning
  quality. Assertions are relaxed to structural properties (segment
  resolved, campaign created, compliance citation present when expected) --
  not exact filter/channel matches, since a real agent may reasonably
  resolve a goal differently than our own scripted guess.

Also covers DESIGN.md SS10 item 6: a retrieval spot-check over hand-labeled
queries, sanity-checking chunking/MMR given how much the guideline corpus
intentionally overlaps.
"""
import json
import os
import sqlite3

import pytest
from langchain_core.messages import ToolMessage

from src.agent.graph import run
from src.agent.llm_client import get_llm_client
from src.config import COMPLIANCE_DOC_SLUG, EXTERNAL_CHANNELS
from src.data_access.db import apply_schema
from src.rag.ingest import build_index
from src.tools.registry import build_tools
from tests.fixtures.golden_set import GOLDEN_FIXTURES, RETRIEVAL_SPOT_CHECKS
from tests.fixtures.hashing_embeddings import HashingEmbeddings


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    apply_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def guidelines_store(tmp_path_factory):
    index_path = str(tmp_path_factory.mktemp("goldenset_guidelines_index"))
    return build_index(index_path=index_path, embeddings=HashingEmbeddings())


@pytest.fixture
def tools(conn, guidelines_store):
    return build_tools(conn, guidelines_store)


def _last_tool_result(messages, tool_name):
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == tool_name:
            return json.loads(message.content)
    return None


def _last_tool_call_input(trace, tool_name):
    for entry in reversed(trace):
        if entry.get("tool") == tool_name:
            return entry.get("input")
    return None


def _check_properties(final_state, expected, *, strict: bool):
    """
    strict=True (deterministic mode): every property checked exactly, since
    we scripted the trajectory ourselves.
    strict=False (live mode): only structural invariants that must hold
    regardless of how the real agent phrased its own reasoning.
    """
    segment_result = _last_tool_result(final_state["messages"], "query_segment")
    campaign_result = _last_tool_result(final_state["messages"], "create_campaign")
    campaign_input = _last_tool_call_input(final_state["trace"], "create_campaign") or {}

    assert segment_result is not None, "expected at least one query_segment call"
    assert segment_result.get("size", -1) >= 0

    assert campaign_result is not None, "expected at least one create_campaign call"
    assert "error" not in campaign_result, f"create_campaign was rejected: {campaign_result}"
    assert campaign_result.get("status") == "created"

    channel = campaign_input.get("channel")
    assert channel, "expected a non-empty channel"

    if strict:
        filters_applied = segment_result.get("filters_applied", {})
        for key in expected.filters_include:
            assert key in filters_applied, f"expected filter '{key}' in {filters_applied}"
        assert channel == expected.channel
        if expected.offer_required:
            assert campaign_input.get("offer"), "expected a non-empty offer"
        if expected.image_required:
            assert campaign_input.get("image_prompt"), "expected a non-empty image_prompt"

    if expected.compliance_expected or channel in EXTERNAL_CHANNELS:
        cited_topics = {c.get("topic_slug") for c in campaign_input.get("guideline_citations", [])}
        assert COMPLIANCE_DOC_SLUG in cited_topics, (
            f"external channel '{channel}' must cite the compliance doc, got {cited_topics}"
        )


@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES, ids=[f.name for f in GOLDEN_FIXTURES])
def test_golden_fixture_deterministic(fixture, tools):
    from src.agent.llm_client import MockLLMClient

    final_state = run(fixture.goal, MockLLMClient(fixture.mock_responses), tools)

    assert final_state["degraded"] is False, f"unexpected degraded run for '{fixture.name}'"
    _check_properties(final_state, fixture.expected, strict=True)


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_EVAL"),
    reason="Live mode hits the real Gemini API (costs quota, non-deterministic) -- opt in with RUN_LIVE_EVAL=1",
)
@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES, ids=[f.name for f in GOLDEN_FIXTURES])
def test_golden_fixture_live(fixture, tools):
    llm = get_llm_client(provider="gemini")
    final_state = run(fixture.goal, llm, tools)

    if final_state["degraded"]:
        pytest.skip(f"real run degraded (provider issue, not a reasoning failure): {final_state['messages'][-1].content}")
    _check_properties(final_state, fixture.expected, strict=False)


@pytest.mark.parametrize("query,expected_topics", RETRIEVAL_SPOT_CHECKS, ids=[q for q, _ in RETRIEVAL_SPOT_CHECKS])
def test_retrieval_spot_check(query, expected_topics, guidelines_store):
    from src.rag.index import search

    results = search(query, k=5, topic_slug=None, store=guidelines_store)
    found_topics = {doc.metadata["topic_slug"] for doc, _score in results}
    assert found_topics & expected_topics, f"expected one of {expected_topics} in top-k for '{query}', got {found_topics}"
