"""Generate the evaluation set (data/golden_set.json).

Each entry pairs a question with the chunk ids that genuinely answer it:

    {"query_id": 1, "question": "...", "relevant_chunk_ids": ["12", "13"]}

Questions are drafted from the source Q&A pairs, then reviewed by hand -
an unreviewed golden set measures the generator, not the truth.

Run: python evaluation/build_golden_set.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: load outputs/chunks.json
    # TODO: draft one question per source Q&A pair, mapped to its chunk ids
    # TODO: write data/golden_set.json for manual review
    raise NotImplementedError


if __name__ == "__main__":
    main()
