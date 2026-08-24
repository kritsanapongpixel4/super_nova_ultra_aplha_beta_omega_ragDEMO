"""Text chunking - split documents into overlapping windows."""

from typing import Any


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a single string into overlapping chunks of ~chunk_size characters."""
    raise NotImplementedError


def split_records(
    records: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Chunk every Q&A record, carrying its metadata onto each chunk.

    Returns chunks shaped like:
        {"chunk_id": 0, "text": ..., "source_id": 3, "line_start": 12, "line_end": 15}
    """
    raise NotImplementedError
