"""Cross-Encoder reranking.

The retriever scores query and chunk separately; a cross-encoder reads both
together and is far more accurate - but too slow to run over the whole corpus,
so it only reorders the candidates the retriever already found.
"""

from typing import Any


class CrossEncoderReranker:
    """Rescore (query, chunk) pairs and reorder them."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """Load the cross-encoder into memory."""
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top_k chunks reordered, each with a rerank_score field."""
        raise NotImplementedError
