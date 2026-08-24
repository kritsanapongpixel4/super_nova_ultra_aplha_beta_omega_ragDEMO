"""Step 3/7 - Turn chunks into embedding vectors.

Reads outputs/chunks.json, writes outputs/embeddings.npy.
Slowest step - the model download happens here on first run.

Run: python pipeline/create_embeddings.py
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import numpy as np  # noqa: E402

import config  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    config.ensure_dirs()

    # 1. Load chunks
    if not config.CHUNKS_FILE.exists():
        print(f"❌ ไม่พบไฟล์ {config.CHUNKS_FILE}")
        print("   กรุณารัน pipeline/chunking.py ก่อน")
        sys.exit(1)

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    print(f"📄 โหลด {len(texts)} chunks จาก {config.CHUNKS_FILE.name}")

    # 2. Encode
    model = EmbeddingModel(config.EMBEDDING_MODEL, normalize=config.NORMALIZE_EMBEDDINGS)
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress=True,
    )
    elapsed = time.perf_counter() - started
    print(
        f"🧮 เข้ารหัสเสร็จใน {elapsed:.0f} วินาที "
        f"({len(texts) / elapsed:.1f} chunks/วินาที)"
    )

    # 3. Sanity-check the shape against config before anything depends on it
    if vectors.shape[0] != len(texts):
        print(f"❌ จำนวนเวกเตอร์ ({vectors.shape[0]}) ไม่ตรงกับ chunks ({len(texts)})")
        sys.exit(1)
    if vectors.shape[1] != config.EMBEDDING_DIM:
        print(
            f"❌ มิติของเวกเตอร์ ({vectors.shape[1]}) ไม่ตรงกับ "
            f"config.EMBEDDING_DIM ({config.EMBEDDING_DIM})"
        )
        print("   แก้ config.EMBEDDING_DIM ให้ตรงก่อนสร้าง FAISS index")
        sys.exit(1)

    norms = np.linalg.norm(vectors, axis=1)
    print(
        f"📐 shape={vectors.shape}  dtype={vectors.dtype}  "
        f"ความยาวเวกเตอร์ min={norms.min():.3f} max={norms.max():.3f}"
    )
    if config.NORMALIZE_EMBEDDINGS and not np.allclose(norms, 1.0, atol=1e-3):
        print("⚠️  ตั้ง NORMALIZE_EMBEDDINGS=True แต่เวกเตอร์ยาวไม่เท่ากับ 1")

    # 4. Write output
    np.save(config.EMBEDDINGS_FILE, vectors)
    size_mb = config.EMBEDDINGS_FILE.stat().st_size / 1024 / 1024
    print(f"💾 บันทึกไว้ที่ {config.EMBEDDINGS_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
