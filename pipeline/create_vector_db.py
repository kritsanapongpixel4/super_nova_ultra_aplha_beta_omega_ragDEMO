"""Step 4/7 - Build the FAISS index from the embeddings.

Reads outputs/embeddings.npy + outputs/chunks.json,
writes vector_db/document.index + vector_db/chunk_store.json.

Run: python pipeline/create_vector_db.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import numpy as np  # noqa: E402

import config  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    # 1. Load the two halves that have to stay aligned
    for path, hint in (
        (config.CHUNKS_FILE, "pipeline/chunking.py"),
        (config.EMBEDDINGS_FILE, "pipeline/create_embeddings.py"),
    ):
        if not path.exists():
            print(f"❌ ไม่พบไฟล์ {path}")
            print(f"   กรุณารัน {hint} ก่อน")
            sys.exit(1)

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(config.EMBEDDINGS_FILE)
    print(f"📄 chunks={len(chunks)}  embeddings={embeddings.shape}")

    # 2. Build
    store = VectorStore(config.EMBEDDING_DIM)
    store.build(embeddings, chunks)
    print(f"🔧 สร้าง FAISS index แล้ว (IndexFlatIP, {len(store)} เวกเตอร์, มิติ {store.dim})")

    # 3. Save
    store.save(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    index_mb = config.FAISS_INDEX_FILE.stat().st_size / 1024 / 1024
    store_mb = config.CHUNK_STORE_FILE.stat().st_size / 1024 / 1024
    print(f"💾 {config.FAISS_INDEX_FILE.name} ({index_mb:.1f} MB)")
    print(f"💾 {config.CHUNK_STORE_FILE.name} ({store_mb:.1f} MB)")

    # 4. Read it back and search with a vector we already have — proves the
    #    saved index is usable and still aligned with the chunk store.
    reloaded = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    hits = reloaded.search(embeddings[0], k=3)
    print(f"\n🔎 ทดสอบค้นหาด้วยเวกเตอร์ของ chunk #0 — ได้ {len(hits)} ผลลัพธ์:")
    for hit in hits:
        preview = hit["text"][:70].replace("\n", " ")
        print(f"   {hit['rank']}. score={hit['score']:.3f}  {preview}")


if __name__ == "__main__":
    main()
