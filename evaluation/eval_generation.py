"""Evaluate answer quality.

Retrieval metrics only prove the right chunks were found; this scores what the
model did with them:

    faithfulness  - is every claim supported by the retrieved context?
    relevance     - does the answer address the question?
    completeness  - does it cover the reference answer?

Judged by an LLM against the golden set, so re-run it after prompt changes.

Run: python evaluation/eval_generation.py   ->  outputs/eval_generation.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: load data/golden_set.json and run the full pipeline per question
    # TODO: score each answer with an LLM judge (claude-opus-5)
    # TODO: write outputs/eval_generation.json with per-question and mean scores
    raise NotImplementedError


if __name__ == "__main__":
    main()
