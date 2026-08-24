"""FAISS vector database - dense semantic search.

The FAISS index holds vectors only; chunk_store.json holds the text and
metadata in exactly the same order, so a FAISS id maps straight to a chunk.
"""

from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """Thin wrapper around a FAISS index plus its aligned chunk store."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = None
        self.chunks: list[dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        """Create the index from scratch. len(chunks) must equal len(embeddings)."""
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int) -> list[dict[str, Any]]:
        """Return the k nearest chunks, each with a score field."""
        raise NotImplementedError

    def save(self, index_path: Path, chunk_store_path: Path) -> None:
        """Persist the index and the chunk store."""
        raise NotImplementedError

    @classmethod
    def load(cls, index_path: Path, chunk_store_path: Path) -> "VectorStore":
        """Load a previously built index from disk."""
        raise NotImplementedError
