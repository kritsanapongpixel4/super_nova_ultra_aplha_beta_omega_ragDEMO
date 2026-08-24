"""File loading and text extraction.

Reads all supported files from the data/ directory and turns them into
structured records, keeping source file names so every passage can be
traced back to the original document.

Supported formats: .pdf, .txt, .docx, .md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Format-specific readers ─────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import pymupdf  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF extraction.  "
            "Install it with:  pip install pymupdf"
        ) from exc

    text_parts: list[str] = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read a plain-text / markdown file."""
    return path.read_text(encoding=encoding)


def _read_docx(path: Path) -> str:
    """Extract text from a .docx file."""
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for .docx extraction.  "
            "Install it with:  pip install python-docx"
        ) from exc

    doc = docx.Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs)


_READERS: dict[str, Any] = {
    ".pdf": _read_pdf,
    ".txt": _read_text,
    ".md": _read_text,
    ".docx": _read_docx,
}


# ── Public API ──────────────────────────────────────────────────────────

def load_text(path: Path, encoding: str = "utf-8") -> str:
    """Return the raw text contents of a supported file."""
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported file type: {suffix!r} ({path.name})")
    if suffix in {".txt", ".md"}:
        return reader(path, encoding=encoding)
    return reader(path)


def parse_document(text: str, source: str = "") -> list[dict[str, Any]]:
    """Split raw text into passage records.

    For general documents (not Q&A), each non-empty paragraph becomes a
    record.  Returns records shaped like::

        {"id": 0, "text": "…", "source": "filename.pdf",
         "line_start": 1, "line_end": 4}
    """
    records: list[dict[str, Any]] = []
    lines = text.split("\n")

    current_passage: list[str] = []
    start_line = 1
    record_id = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped:
            if not current_passage:
                start_line = i
            current_passage.append(line)
        else:
            # Empty line → flush the current passage
            if current_passage:
                records.append({
                    "id": record_id,
                    "text": "\n".join(current_passage),
                    "source": source,
                    "line_start": start_line,
                    "line_end": i - 1,
                })
                record_id += 1
                current_passage = []

    # Flush any remaining passage
    if current_passage:
        records.append({
            "id": record_id,
            "text": "\n".join(current_passage),
            "source": source,
            "line_start": start_line,
            "line_end": len(lines),
        })

    return records


def extract_all(paths: list[Path]) -> list[dict[str, Any]]:
    """Load every file in *paths* and return a flat list of records.

    Each record carries a globally unique ``id`` and the ``source``
    filename so it can be traced back to the original document.
    """
    all_records: list[dict[str, Any]] = []
    global_id = 0

    for path in paths:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue

        try:
            text = load_text(path)
        except Exception:
            logger.exception("Failed to read %s — skipping", path.name)
            continue

        if not text.strip():
            logger.info("Empty content in %s — skipping", path.name)
            continue

        records = parse_document(text, source=path.name)
        # Re-number with globally unique IDs
        for rec in records:
            rec["id"] = global_id
            global_id += 1

        logger.info("  📄 %s → %d records", path.name, len(records))
        all_records.extend(records)

    return all_records


# Keep the old single-file API for backward compat
def extract(path: Path) -> list[dict[str, Any]]:
    """Load a single file and parse it into records."""
    return extract_all([path])
