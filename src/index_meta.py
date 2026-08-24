"""Detect when the index is stale relative to the dataset.

Stores a fingerprint (file hash + chunking/embedding settings) next to the
index so the pipeline can refuse to answer from an index built on old data.
"""

from pathlib import Path
from typing import Any


def file_fingerprint(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    raise NotImplementedError


def build_meta(source_path: Path, n_chunks: int) -> dict[str, Any]:
    """Describe the dataset and settings an index was built from."""
    raise NotImplementedError


def write_meta(meta: dict[str, Any], path: Path) -> None:
    """Write index_meta.json."""
    raise NotImplementedError


def is_stale(source_path: Path, meta_path: Path) -> bool:
    """True when the index needs rebuilding (missing meta, or fingerprint changed)."""
    raise NotImplementedError
