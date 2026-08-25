"""Embedding model wrapper.

Keeps the rest of the code independent of which sentence-transformer is used;
only this module (and the registry it reads) knows about model names, the
prompt each family expects, and which device the weights land on.

Two things here are easy to get wrong and expensive to get wrong:

**Prompts.** E5 and Qwen3-style embedders were trained asymmetrically — the
query side carries a prefix the document side does not.  Encoding both the
same way puts questions and passages in slightly different regions of the
space and quietly costs recall, with no error anywhere to notice.  The
prefixes come from ``src/model_registry.py``, which copied them from each
model's own config.

**Sequence length.** sentence-transformers takes the tokenizer's maximum,
which for Qwen3 is 32k tokens.  Attention is quadratic, so a model that
would encode a 300-token chunk in milliseconds instead reserves buffers for
a context 100x longer than anything in this corpus.  The registry caps it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from . import model_registry

logger = logging.getLogger(__name__)


def pick_device(requested: str | None = None) -> str:
    """Return the device to load on: the request, else CUDA, else CPU."""
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch missing or broken — CPU is still correct
        pass
    return "cpu"


class EmbeddingModel:
    """Encodes text into dense vectors."""

    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        device: str | None = None,
        dtype: str | None = None,
        spec: model_registry.EmbeddingSpec | None = None,
    ) -> None:
        # Accept a registry key or a raw HuggingFace id, so both
        # `EmbeddingModel("qwen3-0.6b")` and the historical
        # `EmbeddingModel("BAAI/bge-m3")` work.
        self.spec = spec or model_registry.resolve(model_name)
        self.model_name = self.spec.hf_id
        self.normalize = normalize
        self.device = pick_device(device)
        self.dtype = dtype
        self._model = None  # lazily loaded on first use
        self.load_seconds: float | None = None

    def __repr__(self) -> str:  # shows up in logs and benchmark output
        return f"<EmbeddingModel {self.spec.key} on {self.device}>"

    # ── Lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the underlying model into memory.

        Downloads it on first use (bge-m3 is ~2.2 GB, Qwen3-4B is ~8 GB), so
        this is deliberately lazy — importing the module stays cheap.
        """
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(
            "⏳ กำลังโหลดโมเดล %s (%s) บน %s ...",
            self.spec.key,
            self.model_name,
            self.device,
        )
        started = time.perf_counter()

        kwargs: dict[str, Any] = {"device": self.device}
        if self.dtype:
            # float16 halves the resident size, which is the difference
            # between a 4B model fitting in 8 GB of VRAM and not.  Only ever
            # requested explicitly — on CPU, float16 matmul is slower than
            # float32, not faster.
            kwargs["model_kwargs"] = {"dtype": self.dtype}

        self._model = SentenceTransformer(self.model_name, **kwargs)

        if self.spec.max_seq_length:
            # Never raise it above what the checkpoint supports — e5-base has
            # 514 learned position embeddings and reading past them is not a
            # slow path, it is garbage output.
            self._model.max_seq_length = min(
                self.spec.max_seq_length, self._model.max_seq_length
            )

        self.load_seconds = time.perf_counter() - started
        logger.info(
            "✅ โหลดเสร็จใน %.1fs (มิติ=%d, seq=%d, อุปกรณ์=%s)",
            self.load_seconds,
            self.dim,
            self._model.max_seq_length,
            self._model.device,
        )

    def unload(self) -> None:
        """Drop the weights and give the VRAM back.

        The benchmark loads eight models one after another in a single
        process; without this the second 4 GB model meets a GPU that is
        still holding the first.
        """
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this model produces."""
        self.load()
        # sentence-transformers renamed this between major versions; the
        # registry's declared dimension is the last resort so that a model
        # not in the registry still reports something real.
        for name in ("get_sentence_embedding_dimension", "get_embedding_dimension"):
            getter = getattr(self._model, name, None)
            if getter is not None:
                value = getter()
                if value:
                    return int(value)
        return self.spec.dim

    # ── Encoding ────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts. Returns shape (len(texts), dim).

        Documents by default; pass ``is_query=True`` for questions so the
        query-side prompt is applied instead of the document one.
        """
        self.load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefix = self.spec.query_prompt if is_query else self.spec.doc_prompt
        prepared = [prefix + text for text in texts] if prefix else texts

        vectors = self._model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (dim,).

        The query prompt is applied here and only here.  bge-m3 declares an
        empty one, so for that model this is identical to encoding a
        document — which is exactly how it was trained.
        """
        return self.encode([query], is_query=True)[0]
