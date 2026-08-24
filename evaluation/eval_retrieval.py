"""Compare retrieval configurations against the golden set.

Runs each configuration over every golden question and reports the metrics
side by side, so the choice of retriever is a measurement, not a guess.

Configurations compared:
    dense            - FAISS only
    bm25             - BM25 only
    hybrid           - BM25 + dense + RRF
    hybrid+rerank    - hybrid, reordered by the cross-encoder

Run: python evaluation/eval_retrieval.py   ->  outputs/eval_retrieval.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: load data/golden_set.json
    # TODO: for each config, retrieve for every question and score with metrics.py
    # TODO: print a comparison table and write outputs/eval_retrieval.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
