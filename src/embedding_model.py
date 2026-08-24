"""Embedding model wrapper.

Keeps the rest of the code independent of which sentence-transformer is used;
only this module knows about the model name and batching.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Encodes text into dense vectors."""

    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = None  # lazily loaded on first use

    # ── Lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the underlying model into memory.

        Downloads it on first use (BAAI/bge-m3 is ~2.2 GB), so this is
        deliberately lazy — importing the module stays cheap.
        """
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("⏳ กำลังโหลดโมเดล %s ...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info(
            "✅ โหลดเสร็จ (มิติ=%d, อุปกรณ์=%s)", self.dim, self._model.device
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this model produces."""
        self.load()
        return self._model.get_embedding_dimension()

    # ── Encoding ────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of documents. Returns shape (len(texts), dim)."""
        self.load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (dim,).

        bge-m3 needs no instruction prefix, so a query is embedded exactly
        the same way as a document — keep it that way or the query and the
        index will no longer live in the same space.
        """
        return self.encode([query])[0]
