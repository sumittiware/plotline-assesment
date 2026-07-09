"""
Runtime load/query wrapper over the FAISS index built by ingest.py. Loads once
per process (index load + embedding model init aren't free) and exposes a
single `search` used by the search_guidelines tool.
"""
from functools import lru_cache
from typing import Optional

from langchain_community.vectorstores import FAISS

from src.config import VECTOR_INDEX_PATH
from src.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def load_index(index_path: str = VECTOR_INDEX_PATH) -> FAISS:
    return FAISS.load_local(
        index_path, get_embeddings(), allow_dangerous_deserialization=True
    )


def search(query: str, k: int, topic_slug: Optional[str], store: FAISS):
    """
    MMR over pure top-k similarity: the guideline corpus intentionally overlaps
    (re-engagement/winback/frequency-capping all touch "how often to message"),
    so naive top-k tends to return near-duplicate chunks. MMR trades a little
    relevance for diversity, which matters more here (DESIGN.md SS5.4).

    topic_slug is an optional metadata filter, never filter-only -- retrieval
    stays hybrid since the caller's inferred topic could be wrong.

    Returns (doc, score) pairs, score normalized to 0-1 (higher = more
    relevant) via the store's own relevance_score_fn, so citations/UI don't
    have to reason about raw FAISS L2 distance.
    """
    filter_ = {"topic_slug": topic_slug} if topic_slug else None
    # Embed with the store's own embedding function, never a fresh global one --
    # a query embedded with a different model than the index was built with
    # would silently produce meaningless similarity scores.
    embedding = store.embeddings.embed_query(query)
    # fetch_k wider than k so MMR has a real candidate pool to diversify over,
    # especially once a metadata filter narrows things down.
    docs_and_scores = store.max_marginal_relevance_search_with_score_by_vector(
        embedding, k=k, fetch_k=max(4 * k, 20), filter=filter_
    )
    relevance_score_fn = store._select_relevance_score_fn()
    # FAISS's flat index returns *squared* L2 distance, but langchain's default
    # relevance formula assumes non-squared -- weak matches can come out
    # slightly negative. Clamp rather than leak that quirk to callers.
    return [(doc, max(0.0, min(1.0, relevance_score_fn(score)))) for doc, score in docs_and_scores]
