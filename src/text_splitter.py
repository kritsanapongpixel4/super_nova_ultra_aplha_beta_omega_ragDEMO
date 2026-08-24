"""Text chunking – split documents into overlapping windows.

Uses **PyThaiNLP** for word tokenisation and LangChain's
``RecursiveCharacterTextSplitter`` so that chunk boundaries respect
Thai word boundaries instead of slicing mid‑word.
"""

from __future__ import annotations

import re

from typing import Any

from pythainlp.tokenize import word_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Thai‑aware separators (highest→lowest priority) ────────────────────
# The splitter tries each separator in order and falls back to the next
# one when a chunk is still too large.
# Thai or Latin letters only: digits and punctuation do not make a chunk
# worth embedding.  A page number, a lone bullet or a run of form dots
# all score zero here.
_LETTER_RE = re.compile(r"[ก-๙A-Za-z]")


_THAI_SEPARATORS: list[str] = [
    "\n\n",        # paragraph break
    "\n",          # line break
    "。",          # CJK period (sometimes found in mixed content)
    " ",           # space (common around English words inside Thai text)
    "ๆ",           # Thai repetition mark – natural phrase boundary
    "",            # character‑level fallback
]


def _thai_token_len(text: str) -> int:
    """Return the number of Thai tokens in *text*.

    Used as the ``length_function`` for the recursive splitter so that
    ``chunk_size`` is measured in **tokens** rather than raw characters.
    """
    return len(word_tokenize(text, engine="newmm"))


def _make_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Build a reusable splitter configured for Thai text."""
    return RecursiveCharacterTextSplitter(
        separators=_THAI_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_thai_token_len,
        is_separator_regex=False,
    )


# ── Public API ──────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a single string into overlapping chunks of ~*chunk_size* Thai tokens."""
    splitter = _make_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)


def is_noise(text: str, min_letters: int) -> bool:
    """True if *text* has too few real letters to be worth indexing."""
    return len(_LETTER_RE.findall(text)) < min_letters


def split_records(
    records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_letters: int = 0,
) -> list[dict[str, Any]]:
    """Chunk every Q&A record, carrying its metadata onto each chunk.

    Returns chunks shaped like::

        {
            "chunk_id": 0,
            "text": "…",
            "source_id": 3,
            "source": "filename.pdf",
            "line_start": 12,
            "line_end": 15,
        }
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for record in records:
        # Support both generic "text" records and Q&A records
        passage = record.get("text", "")
        if not passage:
            # Fallback: combine question + answer (Q&A format)
            parts_list: list[str] = []
            if record.get("question"):
                parts_list.append(record["question"].strip())
            if record.get("answer"):
                parts_list.append(record["answer"].strip())
            passage = "\n".join(parts_list)
        if not passage:
            continue

        # A record marked "atomic" is a structured unit that only makes sense
        # whole — a course and all its CLOs, say.  Splitting it would scatter
        # one answer across several chunks, where TOP_K can then cut it in
        # half without anything looking wrong.
        parts = [passage] if record.get("atomic") else splitter.split_text(passage)
        for part in parts:
            if is_noise(part, min_letters):
                continue
            chunk = {
                "chunk_id": chunk_id,
                "text": part,
                "source_id": record.get("id"),
                "source": record.get("source", ""),
                "line_start": record.get("line_start"),
                "line_end": record.get("line_end"),
            }
            if record.get("course_code"):
                chunk["course_code"] = record["course_code"]
            chunks.append(chunk)
            chunk_id += 1

    return chunks
