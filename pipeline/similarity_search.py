"""Step 6/7 - Retrieve the top-k most similar chunks.

Loads the FAISS index, embeds a query, prints the k best matches with scores.

Run: python pipeline/similarity_search.py "your question"
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402
from src.retriever import DenseRetriever  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_QUESTION = "วิชาปฏิบัติการควบคุมเวอร์ชันมี CLO อะไรบ้าง"


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    if not config.FAISS_INDEX_FILE.exists():
        print(f"❌ ไม่พบ {config.FAISS_INDEX_FILE}")
        print("   กรุณารัน pipeline/create_vector_db.py ก่อน")
        sys.exit(1)

    store = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    embedder = EmbeddingModel(config.EMBEDDING_MODEL, normalize=config.NORMALIZE_EMBEDDINGS)
    retriever = DenseRetriever(store, embedder)

    print(f"📚 index: {len(store)} เวกเตอร์ มิติ {store.dim}")
    print(f"❓ คำถาม: {question}\n")

    hits = retriever.retrieve(question, config.TOP_K)
    if not hits:
        print("ไม่พบผลลัพธ์")
        return

    print(f"🔎 {len(hits)} อันดับแรก (cosine similarity):")
    for hit in hits:
        where = f"หน้า/บรรทัด {hit.get('line_start')}-{hit.get('line_end')}"
        print(f"\n  {hit['rank']}. score={hit['score']:.4f}  [{hit['source']}]  {where}")
        for line in hit["text"].splitlines()[:4]:
            print(f"       {line[:96]}")
        if len(hit["text"].splitlines()) > 4:
            print("       …")


if __name__ == "__main__":
    main()
