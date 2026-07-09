"""
search_guidelines tested against a real FAISS index built from the actual
/guidelines corpus (17 short docs), with chunking/metadata exactly as
production ingest.py produces them.

Embeddings are swapped for tests/fixtures/hashing_embeddings.py -- a
deterministic, local, keyword-overlap embedder -- rather than the real
sentence-transformers model. That keeps this suite fast and independent of
being able to reach a model registry over the network, while still
exercising the real chunking, MMR, and metadata-filtering logic. Test
queries are phrased to share literal vocabulary with the doc they should
match, which is enough for a keyword-overlap embedder to get right.

Module-scoped fixture so the index-build cost is paid once, not once per test.
"""
import pytest

from src.rag.ingest import build_index
from src.tools.search_guidelines import GuidelinesQuery, search_guidelines
from tests.fixtures.hashing_embeddings import HashingEmbeddings


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    index_path = str(tmp_path_factory.mktemp("guidelines_index"))
    return build_index(index_path=index_path, embeddings=HashingEmbeddings())


def test_returns_at_most_k_results(store):
    results = search_guidelines(GuidelinesQuery(query="how often should we message users", k=3), store)
    assert 0 < len(results) <= 3


def test_winback_query_surfaces_winback_or_reengagement_doc(store):
    results = search_guidelines(GuidelinesQuery(query="winning back dormant, churned users", k=5), store)
    topic_slugs = {r.topic_slug for r in results}
    assert topic_slugs & {"winback-churned-users", "re-engagement-playbook"}


def test_push_copy_query_surfaces_push_doc(store):
    results = search_guidelines(GuidelinesQuery(query="writing effective push notification copy", k=4), store)
    source_docs = {r.source_doc for r in results}
    assert "03-push-notification-copy.md" in source_docs


def test_topic_slug_filter_narrows_to_that_doc_only(store):
    results = search_guidelines(
        GuidelinesQuery(query="best practices", k=5, topic_slug="email-best-practices"), store
    )
    assert results
    assert all(r.topic_slug == "email-best-practices" for r in results)


def test_chunks_carry_citation_metadata(store):
    results = search_guidelines(GuidelinesQuery(query="unsubscribe and consent", k=3), store)
    for r in results:
        assert r.chunk_id
        assert r.source_doc.endswith(".md")
        assert r.section_header
        assert 0.0 <= r.score <= 1.0


def test_mmr_avoids_returning_only_duplicate_doc_for_overlapping_query(store):
    """Frequency/re-engagement/winback all discuss 'how often to message' --
    naive top-k tends to collapse onto one doc; MMR should surface more than one."""
    results = search_guidelines(GuidelinesQuery(query="how often should we message users", k=5), store)
    source_docs = {r.source_doc for r in results}
    assert len(source_docs) > 1


def test_compliance_doc_is_retrievable_directly(store):
    results = search_guidelines(GuidelinesQuery(query="opt-outs, consent and suppression lists", k=3), store)
    source_docs = {r.source_doc for r in results}
    assert "15-consent-compliance-and-opt-outs.md" in source_docs
