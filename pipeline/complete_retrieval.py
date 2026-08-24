"""Step 7/7 - The complete retrieval pipeline.

Combines everything: hybrid retrieval (BM25 + dense + RRF) plus cross-encoder
reranking, saved to outputs/retrieval_results.json for inspection.

Run: python pipeline/complete_retrieval.py "your question"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: build HybridRetriever + CrossEncoderReranker
    # TODO: compare dense-only vs hybrid vs hybrid+rerank on the same query
    # TODO: write outputs/retrieval_results.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
