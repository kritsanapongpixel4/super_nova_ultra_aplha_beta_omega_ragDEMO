"""FAISS vector database - dense semantic search.

The FAISS index holds vectors only; chunk_store.json holds the text and
metadata in exactly the same order, so a FAISS id maps straight to a chunk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    """FAISS needs a C-contiguous float32 2-D array; coerce whatever we got."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return np.ascontiguousarray(matrix)


class VectorStore:
    """Thin wrapper around a FAISS index plus its aligned chunk store."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = None
        self.chunks: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def build(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        """Create the index from scratch. len(chunks) must equal len(embeddings)."""
        import faiss

        matrix = _as_matrix(embeddings)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings ({matrix.shape[0]}) and chunks ({len(chunks)}) "
                "must line up — a FAISS id is an index into chunk_store"
            )
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {matrix.shape[1]} != VectorStore dim {self.dim}"
            )

        # Inner product, not L2: the vectors are unit-length (see
        # config.NORMALIZE_EMBEDDINGS), so a dot product *is* cosine
        # similarity — and higher means closer, unlike an L2 distance.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)
        self.chunks = list(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[dict[str, Any]]:
        """Return the k nearest chunks, each with a score field."""
        if self.index is None:
            raise RuntimeError("index is empty — call build() or load() first")

        k = max(1, min(k, len(self)))
        scores, ids = self.index.search(_as_matrix(query_vector), k)

        results: list[dict[str, Any]] = []
        for rank, (chunk_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if chunk_id < 0:  # FAISS pads with -1 when it finds fewer than k
                continue
            hit = dict(self.chunks[chunk_id])
            hit["score"] = float(score)
            hit["rank"] = rank
            results.append(hit)
        return results

    def save(self, index_path: Path, chunk_store_path: Path) -> None:
        """Persist the index and the chunk store."""
        import faiss

        if self.index is None:
            raise RuntimeError("nothing to save — call build() first")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        # faiss.write_index() goes through C++ file IO, which cannot open a
        # path containing non-ASCII characters on Windows — and this project
        # sits under a Thai directory name.  Serialise in memory and let
        # Python write the bytes instead.
        index_path.write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(chunk_store_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: Path, chunk_store_path: Path) -> "VectorStore":
        """Load a previously built index from disk."""
        import faiss

        for path in (index_path, chunk_store_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run pipeline/create_vector_db.py first"
                )

        # Mirror of save(): read the bytes in Python, rebuild in memory.
        payload = np.frombuffer(index_path.read_bytes(), dtype=np.uint8).copy()
        index = faiss.deserialize_index(payload)
        with open(chunk_store_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if index.ntotal != len(chunks):
            raise ValueError(
                f"index holds {index.ntotal} vectors but chunk_store has "
                f"{len(chunks)} chunks — they are out of sync, rebuild the index"
            )

        store = cls(index.d)
        store.index = index
        store.chunks = chunks
        return store
