"""End-to-end RAG pipeline.

    query -> (transform) -> retrieve -> (rerank) -> generate -> answer + sources

Every stage is optional and swappable, so the evaluation scripts can compare
configurations by assembling different pipelines.
"""

from typing import Any

from .generator import Generator
from .memory import ConversationMemory
from .rerankers import CrossEncoderReranker


class RAGPipeline:
    """Ties retrieval, reranking, and generation together."""

    def __init__(
        self,
        retriever: Any,
        generator: Generator,
        reranker: CrossEncoderReranker | None = None,
        memory: ConversationMemory | None = None,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.memory = memory
        self.top_k = top_k
        self.candidate_k = candidate_k

    @classmethod
    def from_config(cls, use_memory: bool = False) -> "RAGPipeline":
        """Build the default pipeline from config.py and the on-disk indexes.

        Raises if the index is stale (see src.index_meta.is_stale).
        """
        raise NotImplementedError

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve candidates, then rerank them down to top_k."""
        raise NotImplementedError

    def answer(self, query: str) -> dict[str, Any]:
        """Answer a question.

        Returns:
            {"question": ..., "answer": ..., "sources": [chunk, ...]}
        """
        raise NotImplementedError
