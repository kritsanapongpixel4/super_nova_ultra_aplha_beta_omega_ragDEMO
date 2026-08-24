"""Dense-only retrieval - embed the query, search FAISS, return top-k chunks."""

from typing import Any

from .embedding_model import EmbeddingModel
from .vector_store import VectorStore


class DenseRetriever:
    """Baseline retriever: pure vector similarity."""

    def __init__(self, store: VectorStore, embedder: EmbeddingModel) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return the k most similar chunks, ranked best-first."""
        return self.store.search(self.embedder.encode_query(query), k)
