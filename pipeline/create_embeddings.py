"""Step 3/7 - Turn chunks into embedding vectors.

Reads outputs/chunks.json, writes outputs/embeddings.npy.
Slowest step - the model download happens here on first run.

Run: python pipeline/create_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: load chunks
    # TODO: EmbeddingModel(config.EMBEDDING_MODEL).encode([c["text"] for c in chunks])
    # TODO: np.save(config.EMBEDDINGS_FILE, vectors) and print the array shape
    raise NotImplementedError


if __name__ == "__main__":
    main()
