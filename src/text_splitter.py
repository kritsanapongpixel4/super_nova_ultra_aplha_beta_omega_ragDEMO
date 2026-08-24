"""Text chunking – split documents into overlapping windows.

Uses **PyThaiNLP** for word tokenisation and LangChain's
``RecursiveCharacterTextSplitter`` so that chunk boundaries respect
Thai word boundaries instead of slicing mid‑word.
"""

from __future__ import annotations

from typing import Any

from pythainlp.tokenize import word_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Thai‑aware separators (highest→lowest priority) ────────────────────
# The splitter tries each separator in order and falls back to the next
# one when a chunk is still too large.
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


def split_records(
    records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Chunk every Q&A record, carrying its metadata onto each chunk.

    Returns chunks shaped like::

        {
            "chunk_id": 0,
            "text": "…",
            "source_id": 3,
            "line_start": 12,
            "line_end": 15,
        }
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for record in records:
        # Combine question + answer into a single passage for chunking
        passage = ""
        if record.get("question"):
            passage += record["question"].strip()
        if record.get("answer"):
            if passage:
                passage += "\n"
            passage += record["answer"].strip()
        if not passage:
            # fallback: try a generic "text" key
            passage = record.get("text", "")
        if not passage:
            continue

        parts = splitter.split_text(passage)
        for part in parts:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": part,
                    "source_id": record.get("id"),
                    "line_start": record.get("line_start"),
                    "line_end": record.get("line_end"),
                }
            )
            chunk_id += 1

    return chunks
