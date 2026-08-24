"""Step 6/7 - Retrieve the top-k most similar chunks.

Loads the FAISS index, embeds a query, prints the k best matches with scores.

Run: python pipeline/similarity_search.py "your question"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    # TODO: store.search(query_vector, config.TOP_K)
    # TODO: print rank, score, line numbers, and a text preview per hit
    raise NotImplementedError


if __name__ == "__main__":
    main()
