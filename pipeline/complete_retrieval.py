"""Step 7/7 - The complete retrieval pipeline.

Combines everything: hybrid retrieval (BM25 + dense + RRF) plus cross-encoder
reranking, saved to outputs/retrieval_results.json for inspection.

Run: python pipeline/complete_retrieval.py "your question"
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402
from src.hybrid_retriever import BM25Index, HybridRetriever  # noqa: E402
from src.rerankers import CrossEncoderReranker  # noqa: E402
from src.retriever import DenseRetriever  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_QUESTION = "วิชาปฏิบัติการควบคุมเวอร์ชันมี CLO อะไรบ้าง"


def load_bm25(chunks: list[dict]) -> BM25Index:
    """Load the BM25 index, building and saving it the first time."""
    if config.BM25_INDEX_FILE.exists():
        index = BM25Index.load(config.BM25_INDEX_FILE)
        if len(index) == len(chunks):
            print(f"📦 โหลด BM25 index เดิม ({len(index)} chunks)")
            return index
        # A stale pickle would rank against documents the FAISS index no
        # longer has — rebuild rather than fuse two different corpora.
        print(f"♻️  BM25 index เดิมมี {len(index)} chunks แต่คลังมี {len(chunks)} — สร้างใหม่")

    started = time.perf_counter()
    index = BM25Index()
    index.build(chunks)
    index.save(config.BM25_INDEX_FILE)
    print(f"🔧 สร้าง BM25 index ({len(index)} chunks) ใน {time.perf_counter()-started:.1f}s")
    return index


def show(title: str, hits: list[dict], elapsed: float) -> None:
    print(f"\n── {title}  ({elapsed:.2f}s) " + "─" * max(0, 46 - len(title)))
    if not hits:
        print("   (ไม่พบผลลัพธ์)")
        return
    for hit in hits:
        head = hit["text"].splitlines()[0][:80]
        extra = ""
        if "rerank_score" in hit:
            extra = f"  (retrieval={hit.get('retrieval_score'):.4f})" if hit.get("retrieval_score") is not None else ""
        print(f"   {hit['rank']}. {hit['score']:.4f}{extra}  [{hit['source'][:24]}]")
        print(f"      {head}")


def main() -> None:
    config.ensure_dirs()

    # config.USE_RERANKER is off for interactive use because the cross-encoder
    # costs minutes per query on CPU; --rerank turns it back on for the
    # evaluation comparison, which is the one place that cost is worth paying.
    argv = [arg for arg in sys.argv[1:] if arg != "--rerank"]
    use_reranker = config.USE_RERANKER or "--rerank" in sys.argv[1:]
    question = " ".join(argv).strip() or DEFAULT_QUESTION

    if not config.FAISS_INDEX_FILE.exists():
        print(f"❌ ไม่พบ {config.FAISS_INDEX_FILE}")
        print("   กรุณารัน pipeline/create_vector_db.py ก่อน")
        sys.exit(1)

    store = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    embedder = EmbeddingModel(config.EMBEDDING_MODEL, normalize=config.NORMALIZE_EMBEDDINGS)
    bm25 = load_bm25(store.chunks)

    print(f"📚 index: {len(store)} เวกเตอร์ | ❓ {question}")

    results: dict[str, list[dict]] = {}

    # 1. Dense only — the baseline every other row is compared against.
    started = time.perf_counter()
    results["dense"] = DenseRetriever(store, embedder).retrieve(question, config.TOP_K)
    show("dense-only", results["dense"], time.perf_counter() - started)

    # 2. Hybrid: BM25 + dense, fused by RRF.
    hybrid = HybridRetriever(store, embedder, bm25)
    started = time.perf_counter()
    results["hybrid"] = hybrid.retrieve(
        question, k=config.TOP_K, candidate_k=config.CANDIDATE_K, rrf_k=config.RRF_K
    )
    show("hybrid (BM25 + dense + RRF)", results["hybrid"], time.perf_counter() - started)

    # 3. Hybrid then reranked — rerank the fused candidates, not just top-k,
    #    or the cross-encoder never sees anything the fusion ranked low.
    if use_reranker:
        started = time.perf_counter()
        candidates = hybrid.retrieve(
            question,
            k=config.CANDIDATE_K,
            candidate_k=config.CANDIDATE_K,
            rrf_k=config.RRF_K,
        )
        reranker = CrossEncoderReranker(config.RERANKER_MODEL)
        results["hybrid_rerank"] = reranker.rerank(question, candidates, config.TOP_K)
        show("hybrid + rerank", results["hybrid_rerank"], time.perf_counter() - started)
    else:
        print("\nℹ️  ข้ามขั้นตอน rerank — ใส่ --rerank เพื่อเปิด (ช้ามากบน CPU)")

    # 4. How much did each stage actually change the order?
    print("\n📊 สรุปความต่าง (เทียบ chunk_id ที่ติดอันดับ):")
    base = [h.get("chunk_id") for h in results["dense"]]
    for name, hits in results.items():
        ids = [h.get("chunk_id") for h in hits]
        overlap = len(set(ids) & set(base))
        print(f"   {name:16s} {ids}  ซ้ำกับ dense {overlap}/{len(base)}")

    payload = {
        "question": question,
        "top_k": config.TOP_K,
        "candidate_k": config.CANDIDATE_K,
        "rrf_k": config.RRF_K,
        "reranker": config.RERANKER_MODEL if use_reranker else None,
        "results": results,
    }
    with open(config.RETRIEVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 บันทึกไว้ที่ {config.RETRIEVAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
