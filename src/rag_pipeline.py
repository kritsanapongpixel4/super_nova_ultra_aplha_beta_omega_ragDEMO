"""End-to-end RAG pipeline.

    query -> (transform) -> retrieve -> (rerank) -> generate -> answer + sources

Every stage is optional and swappable, so the evaluation scripts can compare
configurations by assembling different pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

from .generator import Generator
from .memory import ConversationMemory
from .query_transform import rewrite_query
from .rerankers import CrossEncoderReranker

logger = logging.getLogger(__name__)


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
        import config

        from .embedding_model import EmbeddingModel
        from .hybrid_retriever import BM25Index, HybridRetriever
        from .index_meta import is_stale, read_meta
        from .vector_store import VectorStore

        if not config.FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"ไม่พบ {config.FAISS_INDEX_FILE} — รัน pipeline/create_vector_db.py ก่อน"
            )
        if is_stale(config.SOURCE_FILES, config.INDEX_META_FILE):
            raise RuntimeError(
                "index ไม่ตรงกับข้อมูลใน data/ แล้ว — รัน pipeline/extract_text.py "
                "→ chunking.py → create_embeddings.py → create_vector_db.py ใหม่"
            )
        if read_meta(config.INDEX_META_FILE) is None:
            logger.warning("⚠️  ไม่มี index_meta.json — ข้ามการตรวจว่า index ตรงกับข้อมูล")

        store = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
        embedder = EmbeddingModel(
            config.EMBEDDING_MODEL, normalize=config.NORMALIZE_EMBEDDINGS
        )

        # BM25 is cheap to rebuild (~1.5s for 1.8k chunks) but must describe the
        # same chunks as the FAISS index, or fusion mixes two different corpora.
        bm25 = BM25Index()
        if config.BM25_INDEX_FILE.exists():
            bm25 = BM25Index.load(config.BM25_INDEX_FILE)
        if len(bm25) != len(store.chunks):
            bm25 = BM25Index()
            bm25.build(store.chunks)
            bm25.save(config.BM25_INDEX_FILE)

        return cls(
            retriever=HybridRetriever(store, embedder, bm25),
            generator=Generator(
                model=config.LLM_MODEL, max_tokens=config.LLM_MAX_TOKENS
            ),
            reranker=(
                CrossEncoderReranker(config.RERANKER_MODEL)
                if config.USE_RERANKER
                else None
            ),
            memory=ConversationMemory(config.MEMORY_MAX_TURNS) if use_memory else None,
            top_k=config.TOP_K,
            candidate_k=config.CANDIDATE_K,
        )

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve candidates, then rerank them down to top_k."""
        import config

        if self.reranker is None:
            return self.retriever.retrieve(
                query,
                k=self.top_k,
                candidate_k=self.candidate_k,
                rrf_k=config.RRF_K,
            )

        # With a reranker, hand it the full candidate pool — reranking only the
        # top_k the retriever already picked cannot rescue anything the
        # retriever ranked low, which is the whole point of reranking.
        candidates = self.retriever.retrieve(
            query,
            k=self.candidate_k,
            candidate_k=self.candidate_k,
            rrf_k=config.RRF_K,
        )
        return self.reranker.rerank(query, candidates, self.top_k)

    def answer(self, query: str) -> dict[str, Any]:
        """Answer a question.

        Returns:
            {"question": ..., "answer": ..., "sources": [chunk, ...]}
        """
        history = self.memory.history() if self.memory else None

        # Retrieval sees one string, never the conversation — so a follow-up
        # like "แล้ว CLO ข้อแรกของวิชานั้น" has to be made standalone first or
        # it searches for a course it never names.
        search_query = (
            rewrite_query(
                query,
                history,
                model=self.generator.model,
                client=self.generator.client,
            )
            if history
            else query
        )

        chunks = self.retrieve(search_query)
        text = self.generator.generate(query, chunks, history=history)

        if self.memory is not None:
            # Store the plain question, not the prompt built around it — the
            # retrieved context belongs to one turn and would otherwise pile
            # up in every later request.
            self.memory.add_user(query)
            self.memory.add_assistant(text)

        return {"question": query, "answer": text, "sources": chunks}
