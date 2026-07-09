"""
search_guidelines -- plain Python function, zero LangChain/agent knowledge.
Wraps the FAISS retriever built by src/rag/ingest.py; wrapped as a LangChain
@tool only in registry.py.

Returns chunk_id + source_doc + section_header alongside the text so the
agent's final output can cite exactly which guideline snippet informed a
piece of copy (DESIGN.md SS5.3 grounding: no guideline claim without a
chunk_id that traces back to a real retrieved chunk).
"""
from typing import Optional

from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from src.config import RETRIEVAL_K
from src.rag.index import search as index_search


class GuidelinesQuery(BaseModel):
    query: str
    k: int = RETRIEVAL_K
    topic_slug: Optional[str] = None


class GuidelineChunk(BaseModel):
    chunk_id: str
    text: str
    source_doc: str
    section_header: str
    topic_slug: str
    score: float


def search_guidelines(params: GuidelinesQuery, store: FAISS) -> list[GuidelineChunk]:
    results = index_search(params.query, params.k, params.topic_slug, store)
    return [
        GuidelineChunk(
            chunk_id=doc.metadata["chunk_id"],
            text=doc.page_content,
            source_doc=doc.metadata["source_doc"],
            section_header=doc.metadata["section_header"],
            topic_slug=doc.metadata["topic_slug"],
            score=score,
        )
        for doc, score in results
    ]
