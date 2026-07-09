"""
Chunk + embed /guidelines into a local FAISS index, persisted to disk.

Chunking strategy (see DESIGN.md SS5.4): header-aware first. Each markdown doc
is a handful of short, semantically complete sections (a "##" is one guideline
recommendation), so splitting on headers keeps a chunk whole rather than
slicing a bullet list in half the way a fixed-size window would. A secondary
character-based splitter only kicks in for sections still too long after that
-- with docs this short, it rarely fires.

Run via `python -m src.rag.ingest` (wired to `make ingest`). Embeddings are
computed once here and cached to disk -- the runtime path (index.py) never
re-embeds the corpus per request.
"""
import os
import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import GUIDELINES_DIR, VECTOR_INDEX_PATH
from src.rag.embeddings import get_embeddings

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
MAX_CHUNK_CHARS = 1600  # ~300-500 tokens; secondary split only above this
CHUNK_OVERLAP_CHARS = 240  # ~15% of MAX_CHUNK_CHARS


def _topic_slug(filename: str) -> str:
    """'07-winback-churned-users.md' -> 'winback-churned-users'."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r"^\d+-", "", stem)


def _section_slug(section_header: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", section_header.lower()).strip("-")
    return slug or "root"


def load_guideline_docs(guidelines_dir: str = GUIDELINES_DIR) -> list[tuple[str, str]]:
    """Returns [(filename, raw_markdown_text)], skipping README.md (not a guideline doc)."""
    docs = []
    for filename in sorted(os.listdir(guidelines_dir)):
        if not filename.endswith(".md") or filename.upper() == "README.MD":
            continue
        with open(os.path.join(guidelines_dir, filename)) as f:
            docs.append((filename, f.read()))
    return docs


def chunk_documents(docs: list[tuple[str, str]]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS
    )

    chunks: list[Document] = []
    for filename, text in docs:
        topic_slug = _topic_slug(filename)
        header_sections = header_splitter.split_text(text)

        for section in header_sections:
            doc_title = section.metadata.get("h1", topic_slug)
            section_header = section.metadata.get("h3") or section.metadata.get("h2") or doc_title
            base_metadata = {
                "source_doc": filename,
                "doc_title": doc_title,
                "section_header": section_header,
                "topic_slug": topic_slug,
            }

            if len(section.page_content) <= MAX_CHUNK_CHARS:
                parts = [section.page_content]
            else:
                parts = char_splitter.split_text(section.page_content)

            section_slug = _section_slug(section_header)
            for i, part in enumerate(parts):
                chunk_id = f"{topic_slug}::{section_slug}"
                if len(parts) > 1:
                    chunk_id += f"::{i}"
                chunks.append(
                    Document(
                        page_content=part,
                        metadata={**base_metadata, "chunk_id": chunk_id},
                    )
                )
    return chunks


def build_index(
    guidelines_dir: str = GUIDELINES_DIR,
    index_path: str = VECTOR_INDEX_PATH,
    embeddings=None,
) -> FAISS:
    """
    `embeddings` is injectable (defaults to get_embeddings()) so tests can swap
    in a lightweight local embedder instead of paying for a real model load --
    same rationale as query_segment/create_campaign taking `conn` as a param
    rather than reaching for a global connection.
    """
    docs = load_guideline_docs(guidelines_dir)
    chunks = chunk_documents(docs)
    embeddings = embeddings or get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    store.save_local(index_path)
    return store


if __name__ == "__main__":
    store = build_index()
    print(f"Indexed guidelines into {VECTOR_INDEX_PATH}")
