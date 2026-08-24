"""Cross-Encoder reranking.

The retriever scores query and chunk separately; a cross-encoder reads both
together and is far more accurate - but too slow to run over the whole corpus,
so it only reorders the candidates the retriever already found.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rescore (query, chunk) pairs and reorder them."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """Load the cross-encoder into memory.

        Downloads on first use (BAAI/bge-reranker-v2-m3 is ~2.2 GB) and is
        slow on CPU, so this stays lazy — nothing pays for it unless a
        rerank actually happens.
        """
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        logger.info("⏳ กำลังโหลด reranker %s ...", self.model_name)
        self._model = CrossEncoder(self.model_name)
        logger.info("✅ โหลด reranker เสร็จ")

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top_k chunks reordered, each with a rerank_score field."""
        if not chunks:
            return []

        self.load()
        scores = self._model.predict([(query, chunk["text"]) for chunk in chunks])

        rescored = [dict(chunk) for chunk in chunks]
        for chunk, score in zip(rescored, scores):
            chunk["rerank_score"] = float(score)

        rescored.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
        for rank, chunk in enumerate(rescored[: max(1, top_k)], start=1):
            # Keep the retriever's own score under a separate name — losing it
            # would make it impossible to tell what the reranker changed.
            chunk["retrieval_score"] = chunk.get("score")
            chunk["score"] = chunk["rerank_score"]
            chunk["rank"] = rank
        return rescored[: max(1, top_k)]
