"""Step 4/7 - Build the FAISS vector database.

Reads outputs/chunks.json + outputs/embeddings.npy,
writes vector_db/document.index and vector_db/chunk_store.json.

Run: python pipeline/create_vector_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: VectorStore(config.EMBEDDING_DIM).build(embeddings, chunks)
    # TODO: store.save(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    raise NotImplementedError


if __name__ == "__main__":
    main()
