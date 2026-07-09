"""
A tiny local embedder for tests -- no network, no model download, no
non-determinism. Production code (src/rag/embeddings.py) defaults to a real
sentence-transformers model; tests inject this instead so retrieval-mechanism
tests (MMR, metadata filtering, chunk citation fields) don't depend on being
able to reach a model registry.

It's a bag-of-words hashing vectorizer: lowercase, split on non-alphanumerics,
hash each token into a fixed-size vector, L2-normalize. Since our test queries
deliberately share literal vocabulary with the guideline doc they're meant to
match (e.g. "winning back churned users" vs. "Win-back & Churned Users"),
plain keyword overlap is enough to exercise the retrieval pipeline correctly
without needing real semantic embeddings.
"""
import re
import zlib
from collections import Counter
from math import sqrt

from langchain_core.embeddings import Embeddings

DIM = 256


def _stable_hash(token: str) -> int:
    """crc32, not the builtin hash() -- PYTHONHASHSEED randomizes str hashing
    per-process, which would make vectors non-reproducible across test runs."""
    return zlib.crc32(token.encode())


def _vectorize(text: str) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    counts = Counter(_stable_hash(tok) % DIM for tok in tokens)
    vec = [0.0] * DIM
    for idx, count in counts.items():
        vec[idx] = float(count)
    norm = sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vectorize(text)
