"""File loading and text extraction.

Reads the raw source file and turns it into structured Q&A records,
keeping the line numbers so every answer can be traced back to the file.
"""

from pathlib import Path
from typing import Any


def load_text(path: Path, encoding: str = "utf-8") -> str:
    """Return the raw contents of a text file."""
    raise NotImplementedError


def parse_qa_pairs(text: str) -> list[dict[str, Any]]:
    """Split raw text into Q&A records.

    Returns records shaped like:
        {"id": 0, "question": ..., "answer": ..., "line_start": 1, "line_end": 4}
    """
    raise NotImplementedError


def extract(path: Path) -> list[dict[str, Any]]:
    """Load a file and parse it into Q&A records (writes outputs/extracted_text.json)."""
    raise NotImplementedError
