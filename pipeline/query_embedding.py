"""Step 5/7 - Embed a query.

Shows that a question becomes a vector in the same space as the chunks -
the whole reason similarity search works.

Run: python pipeline/query_embedding.py "your question"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: read the question from sys.argv
    # TODO: EmbeddingModel(...).encode_query(question)
    # TODO: print the vector shape, its norm, and the first few dimensions
    raise NotImplementedError


if __name__ == "__main__":
    main()
