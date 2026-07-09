"""
Single place that decides which embedding backend to use, so ingest.py (build
time) and index.py (query time) can never drift out of sync -- a query
embedded with a different model than the index was built with silently
produces garbage similarity scores.

Defaults to Gemini (models/gemini-embedding-001) so the whole app runs off a
single GOOGLE_API_KEY, same key as the planning LLM. OpenAI and a fully local
sentence-transformers fallback (no API key at all) remain available via
EMBEDDING_PROVIDER for anyone who'd rather not hold a Google key, or has no
key at all.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_PROVIDER, GEMINI_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
