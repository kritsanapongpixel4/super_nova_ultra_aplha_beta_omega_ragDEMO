"""Embedding model wrapper.

Keeps the rest of the code independent of which sentence-transformer is used;
only this module knows about the model name and batching.
"""

import numpy as np


class EmbeddingModel:
    """Encodes text into dense vectors."""

    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = None  # lazily loaded on first use

    def load(self) -> None:
        """Load the underlying model into memory."""
        raise NotImplementedError

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of documents. Returns shape (len(texts), dim)."""
        raise NotImplementedError

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (dim,)."""
        raise NotImplementedError
