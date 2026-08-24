"""Index fingerprinting - tell whether the built index still matches the data.

An index that silently drifts out of sync with data/ is the worst kind of
bug here: retrieval keeps working and keeps returning confident answers, just
from documents that no longer exist or without the ones that were added.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CHUNK_BYTES = 1024 * 1024


def file_fingerprint(path: Path) -> str:
    """Return a content hash for one file.

    Content rather than size+mtime: copying the project or checking it out
    fresh rewrites every mtime, which would report a perfectly good index as
    stale every time.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def sources_fingerprint(paths: list[Path]) -> str:
    """One hash covering the whole source set, including which files it holds.

    File names go into the digest as well, so renaming or removing a document
    counts as a change even when the remaining bytes are identical.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(file_fingerprint(path).encode("ascii"))
    return digest.hexdigest()


def build_meta(source_paths: list[Path], n_chunks: int) -> dict[str, Any]:
    """Describe the data an index was built from."""
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sources": len(source_paths),
        "sources": sorted(p.name for p in source_paths),
        "sources_fingerprint": sources_fingerprint(source_paths),
        "n_chunks": n_chunks,
    }


def write_meta(meta: dict[str, Any], path: Path) -> None:
    """Persist the fingerprint next to the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_meta(path: Path) -> dict[str, Any] | None:
    """Load a previously written fingerprint, or None if there is none."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_stale(source_paths: list[Path], meta_path: Path) -> bool:
    """True if the data no longer matches what the index was built from.

    A missing meta file is *not* stale — older indexes were built before this
    check existed, and refusing to serve them would be worse than not knowing.
    Callers that care should check ``read_meta`` for None themselves.
    """
    meta = read_meta(meta_path)
    if meta is None:
        return False
    return meta.get("sources_fingerprint") != sources_fingerprint(source_paths)
