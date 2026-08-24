"""Hybrid retrieval - BM25 (exact tokens) + Dense (semantics), fused with RRF.

BM25 catches literal terms the embedder blurs away; dense catches paraphrases
BM25 misses. Reciprocal Rank Fusion merges the two rankings without needing
their scores to be on a comparable scale.
"""

from pathlib import Path
from typing import Any

from .embedding_model import EmbeddingModel
from .vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25 (Thai needs a word segmenter, not whitespace splitting)."""
    raise NotImplementedError


class BM25Index:
    """Sparse, exact-token index persisted as bm25_index.pkl."""

    def __init__(self) -> None:
        self.bm25 = None
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        raise NotImplementedError


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse several ranked lists: score = sum of 1 / (rrf_k + rank).

    Chunks are matched across lists by chunk_id.
    """
    raise NotImplementedError


class HybridRetriever:
    """BM25 + Dense + RRF fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingModel,
        bm25: BM25Index,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25

    def retrieve(self, query: str, k: int, candidate_k: int = 20) -> list[dict[str, Any]]:
        """Pull candidate_k from each retriever, fuse the rankings, return the top k."""
        raise NotImplementedError
