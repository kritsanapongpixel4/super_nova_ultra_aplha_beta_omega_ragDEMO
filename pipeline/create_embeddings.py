"""Step 3/7 - Turn chunks into embedding vectors.

Reads outputs/chunks.json, writes outputs/embeddings/<model>.npy.
Slowest step - the model download happens here on first run.

Run: python pipeline/create_embeddings.py
     python pipeline/create_embeddings.py --model qwen3-0.6b --device cuda
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import numpy as np  # noqa: E402

import config  # noqa: E402
from src import cli  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    spec, _ = cli.take_model_flag()
    config.ensure_dirs()
    print(f"🧬 โมเดล: {spec.key} ({spec.hf_id})")

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
    model = EmbeddingModel(
        spec.hf_id,
        normalize=config.NORMALIZE_EMBEDDINGS,
        device=config.EMBEDDING_DEVICE,
        spec=spec,
    )
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
    # A model outside the registry has no declared dimension (spec.dim == 0);
    # the vectors themselves are then the only source of truth, and there is
    # nothing to disagree with.
    if config.EMBEDDING_DIM and vectors.shape[1] != config.EMBEDDING_DIM:
        print(
            f"❌ มิติของเวกเตอร์ ({vectors.shape[1]}) ไม่ตรงกับที่ registry ประกาศไว้ "
            f"({config.EMBEDDING_DIM}) สำหรับ {spec.key}"
        )
        print("   แก้ dim ของโมเดลนี้ใน src/model_registry.py ให้ตรงก่อนสร้าง FAISS index")
        sys.exit(1)

    norms = np.linalg.norm(vectors, axis=1)
    print(
        f"📐 shape={vectors.shape}  dtype={vectors.dtype}  "
        f"ความยาวเวกเตอร์ min={norms.min():.3f} max={norms.max():.3f}"
    )
    if config.NORMALIZE_EMBEDDINGS and not np.allclose(norms, 1.0, atol=1e-3):
        print("⚠️  ตั้ง NORMALIZE_EMBEDDINGS=True แต่เวกเตอร์ยาวไม่เท่ากับ 1")

    # 4. Write output, plus a sidecar recording what it cost to produce.
    #    Without the sidecar the only record of "how long did qwen3-0.6b take
    #    to encode this corpus" is a line that scrolled off a terminal.
    np.save(config.EMBEDDINGS_FILE, vectors)
    size_mb = config.EMBEDDINGS_FILE.stat().st_size / 1024 / 1024
    print(f"💾 บันทึกไว้ที่ {config.EMBEDDINGS_FILE} ({size_mb:.1f} MB)")

    sidecar = config.EMBEDDINGS_FILE.with_suffix(".meta.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": spec.key,
                "hf_id": spec.hf_id,
                "device": model.device,
                "n_chunks": len(texts),
                "dim": int(vectors.shape[1]),
                "batch_size": config.EMBEDDING_BATCH_SIZE,
                "max_seq_length": spec.max_seq_length,
                "load_seconds": round(model.load_seconds or 0.0, 2),
                "encode_seconds": round(elapsed, 2),
                "chunks_per_second": round(len(texts) / elapsed, 2),
                "file_mb": round(size_mb, 1),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"💾 สถิติเวลา → {sidecar.name}")


if __name__ == "__main__":
    main()
