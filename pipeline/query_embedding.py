"""Step 5/7 - Embed a query.

Shows that a question becomes a vector in the same space as the chunks -
the whole reason similarity search works.

Run: python pipeline/query_embedding.py "your question"
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import numpy as np  # noqa: E402

import config  # noqa: E402
from src import cli  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_QUESTION = "วิชาปฏิบัติการควบคุมเวอร์ชันมี CLO อะไรบ้าง"


def main() -> None:
    spec, argv = cli.take_model_flag()
    question = " ".join(argv).strip() or DEFAULT_QUESTION
    print(f"🧬 โมเดล: {spec.key}")
    print(f"❓ คำถาม: {question}")

    model = EmbeddingModel(
        spec.hf_id,
        normalize=config.NORMALIZE_EMBEDDINGS,
        device=config.EMBEDDING_DEVICE,
        spec=spec,
    )
    vector = model.encode_query(question)

    norm = float(np.linalg.norm(vector))
    print(f"📐 shape={vector.shape}  dtype={vector.dtype}  ความยาว(norm)={norm:.4f}")
    preview = ", ".join(f"{value:+.4f}" for value in vector[:8])
    print(f"🔢 8 มิติแรก: [{preview}, ...]")

    if config.NORMALIZE_EMBEDDINGS and abs(norm - 1.0) > 1e-3:
        print("⚠️  ตั้ง NORMALIZE_EMBEDDINGS=True แต่เวกเตอร์ยาวไม่เท่ากับ 1")

    # The query has to live in the same space as the chunks, or the index is
    # searching with a ruler from a different universe.
    if config.EMBEDDING_DIM and vector.shape[0] != config.EMBEDDING_DIM:
        print(
            f"❌ มิติ {vector.shape[0]} ไม่ตรงกับ config.EMBEDDING_DIM "
            f"({config.EMBEDDING_DIM}) — ค้นหาใน FAISS index ไม่ได้"
        )
        sys.exit(1)
    print(f"✅ มิติตรงกับ index ({config.EMBEDDING_DIM}) — ใช้ค้นหาได้")


if __name__ == "__main__":
    main()
