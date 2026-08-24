"""Step 1/7 - Extract text from the source file.

Reads data/sex_q_a.txt, parses it into Q&A records with line numbers,
and writes outputs/extracted_text.json.

Run: python pipeline/extract_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    # TODO: src.document_loader.extract(config.SOURCE_FILE)
    # TODO: json.dump(records, config.EXTRACTED_TEXT_FILE)
    # TODO: print how many records were parsed
    raise NotImplementedError


if __name__ == "__main__":
    main()
