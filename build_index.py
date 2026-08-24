"""Build every index the RAG system needs, from the raw source file.

Usage:
    python build_index.py            # build only if the dataset changed
    python build_index.py --force    # always rebuild
"""

import argparse
import json
import sys
import time

import config

# Windows consoles default to cp1252, which cannot encode Thai or emoji.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def build(force: bool = False) -> None:
    """Run the full offline pipeline: extract -> chunk -> embed -> index."""
    config.ensure_dirs()

    from src.hybrid_retriever import BM25Index
    from src.index_meta import is_stale, read_meta

    if not config.SOURCE_FILES:
        print(f"❌ ไม่พบไฟล์ที่รองรับใน {config.DATA_DIR}")
        print(f"   หมวดที่เลือกอยู่: {config.SOURCE_CATEGORIES}")
        sys.exit(1)

    meta = read_meta(config.INDEX_META_FILE)
    built = config.FAISS_INDEX_FILE.exists() and meta is not None
    if built and not force and not is_stale(config.SOURCE_FILES, config.INDEX_META_FILE):
        print(f"✅ index ตรงกับข้อมูลอยู่แล้ว ({meta['n_chunks']} chunks, สร้างเมื่อ {meta['built_at']})")
        print("   ใส่ --force ถ้าต้องการสร้างใหม่")
        return

    # Imported here, not at module scope: each one pulls in torch or PyMuPDF,
    # so `python build_index.py --help` would otherwise take several seconds.
    from pipeline import chunking, create_embeddings, create_vector_db, extract_text

    steps = [
        ("1/5 แยกข้อความ", extract_text.main),
        ("2/5 ตัด chunk", chunking.main),
        ("3/5 สร้าง embeddings", create_embeddings.main),
        ("4/5 สร้าง FAISS index", create_vector_db.main),
    ]
    started = time.perf_counter()
    for title, run in steps:
        print(f"\n{'='*60}\n▶ {title}\n{'='*60}")
        step_started = time.perf_counter()
        run()
        print(f"   ⏱️  {time.perf_counter() - step_started:.1f}s")

    # BM25 lives outside the FAISS path but has to describe the same chunks,
    # so build it here rather than leaving the first query to pay for it.
    print(f"\n{'='*60}\n▶ 5/5 สร้าง BM25 index\n{'='*60}")
    with open(config.CHUNK_STORE_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(config.BM25_INDEX_FILE)
    print(f"🔧 BM25 index {len(bm25)} chunks → {config.BM25_INDEX_FILE.name}")

    print(f"\n🎉 เสร็จทั้งหมดใน {time.perf_counter() - started:.1f}s")
    print(f"   ลองใช้งาน: python main.py \"คำถามของคุณ\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG indexes.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the index fingerprint still matches the dataset",
    )
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
