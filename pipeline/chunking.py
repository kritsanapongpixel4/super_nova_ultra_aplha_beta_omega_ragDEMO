"""Step 2/7 - Split the extracted text into chunks.

Reads outputs/extracted_text.json, writes outputs/chunks.json.
Try different CHUNK_SIZE / CHUNK_OVERLAP values and watch the chunk count change.

Run: python pipeline/chunking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    # TODO: load extracted records
    # TODO: src.text_splitter.split_records(records, CHUNK_SIZE, CHUNK_OVERLAP)
    # TODO: write outputs/chunks.json and print the chunk count
    raise NotImplementedError


if __name__ == "__main__":
    main()
