"""Load the golden set and bind it to the chunks that are on disk right now.

Every entry carries its answer twice.  ``relevant_chunk_ids`` is the position
the chunk held when the set was built — short, readable, and what
``metrics.score_one`` compares against.  ``relevant_chunk_keys`` is a hash of
the chunk's source file and text, which does not move when the corpus grows.

Both, because only the second one survives a rebuild.  chunk_id is positional:
adding the fourteen registrar PDFs to ``data/curriculum/`` put them ahead of
the CLO table in ``config.SOURCE_FILES``, every chunk after them was
renumbered, and the ids that had pointed at course cards came to point at
whatever now sat at that offset — id "0" went from a CLO card to the cover
page of ``00_สารบัญ-ภาพรวมงานทะเบียน.pdf``.

Nothing failed.  ``score_one`` found no overlap and returned zero, for all 128
questions, and the benchmark recorded that as a model that could not retrieve.
The existing guard in ``build_golden_set.py`` did not catch it either: it
checks that every id *exists*, and "0" still existed.

So ``load()`` re-resolves the keys against the chunks it is given and says out
loud how many entries had to move.  If the keys themselves stop resolving, the
chunk text changed — that is a real rebuild, not a renumbering, and it raises
rather than scoring zero in silence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# Bump when the key recipe changes, so an old set is rebuilt rather than
# silently failing to resolve against a new hash.
KEY_VERSION = 1


class GoldenSetError(RuntimeError):
    """The golden set cannot be trusted against the current chunks."""


def chunk_key(chunk: dict[str, Any]) -> str:
    """A stable identity for *chunk*: its source file plus its text.

    Not the chunk_id, which is a position, and not the text alone — the same
    boilerplate line ("คำสำคัญ", a page header) turns up in several documents,
    and an answer that could be either of them is not an answer.
    """
    payload = f"{chunk.get('source', '')}\n{chunk.get('text', '')}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Identify a corpus by content, so two runs can be compared honestly.

    Order-independent: the digest is over the sorted keys, so re-ordering
    ``SOURCE_CATEGORIES`` does not read as a different corpus when the same
    text came out the other end.
    """
    keys = sorted(chunk_key(chunk) for chunk in chunks)
    sources = {chunk.get("source") for chunk in chunks}
    digest = hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return {
        "n_chunks": len(keys),
        "n_sources": len(sources),
        "chunks_sha1": digest,
        "key_version": KEY_VERSION,
    }


def _entries(raw: Any) -> tuple[list[dict], dict]:
    """Accept both the current ``{meta, queries}`` file and the bare list."""
    if isinstance(raw, dict):
        return list(raw.get("queries", [])), dict(raw.get("meta", {}))
    return list(raw), {}


def load(
    chunks: list[dict[str, Any]],
    path: Path | None = None,
    *,
    quiet: bool = False,
    include_unanswerable: bool = False,
) -> tuple[list[dict], dict]:
    """Return (entries, report) with every ``relevant_chunk_ids`` re-resolved.

    Args:
        chunks: the chunks the retriever under test actually indexed.
        path:   golden set file; defaults to ``data/golden_set.json``.
        quiet:  suppress the summary line.
        include_unanswerable: keep the questions that have no gold chunk.
            Off by default because every retrieval metric scores them 0 by
            construction — ``recall_at_k`` divides by a relevant set that is
            empty — which would drag a retriever's numbers down for answering
            correctly.  ``eval_generation.py`` turns them on: they are the
            only questions that can show whether the system admits it does
            not know.

    Raises:
        GoldenSetError: the file is missing, predates content keys, or its
            keys no longer match anything in *chunks*.  All three mean any
            score computed from it would be fiction.
    """
    import config

    path = path or (config.DATA_DIR / "golden_set.json")
    if not path.exists():
        raise GoldenSetError(
            f"ไม่พบ {path} — รัน python evaluation/build_golden_set.py ก่อน"
        )

    with open(path, "r", encoding="utf-8") as f:
        entries, meta = _entries(json.load(f))
    if not entries:
        raise GoldenSetError(f"{path.name} ว่างเปล่า")

    if any("relevant_chunk_keys" not in entry for entry in entries):
        raise GoldenSetError(
            f"{path.name} เป็นรูปแบบเก่าที่ไม่มี relevant_chunk_keys — "
            "เฉลยผูกกับตำแหน่ง chunk จึงเชื่อถือไม่ได้เมื่อข้อมูลเปลี่ยน\n"
            "   รัน python evaluation/build_golden_set.py เพื่อสร้างใหม่"
        )

    unanswerable = [e for e in entries if not e["relevant_chunk_keys"]]
    if not include_unanswerable:
        entries = [e for e in entries if e["relevant_chunk_keys"]]

    by_key = {chunk_key(chunk): str(chunk["chunk_id"]) for chunk in chunks}
    rebound, unresolved = 0, []

    for entry in entries:
        keys = entry["relevant_chunk_keys"]
        missing = [key for key in keys if key not in by_key]
        if missing:
            unresolved.append(entry.get("query_id"))
            continue
        resolved = [by_key[key] for key in keys]
        if resolved != entry.get("relevant_chunk_ids"):
            entry["relevant_chunk_ids"] = resolved
            rebound += 1

    if unresolved:
        raise GoldenSetError(
            f"{len(unresolved)} จาก {len(entries)} คำถามหา chunk เฉลยไม่เจอ "
            f"(เช่น query_id {unresolved[:5]})\n"
            "   แปลว่าเนื้อความของ chunk เปลี่ยนไป ไม่ใช่แค่เลื่อนตำแหน่ง — "
            "การตั้งค่า chunking หรือตัวเอกสารเปลี่ยน\n"
            "   รัน python evaluation/build_golden_set.py เพื่อสร้างใหม่"
        )

    now = fingerprint(chunks)
    built_on = meta.get("corpus", {})
    report = {
        "n_queries": len(entries),
        "n_unanswerable": len(unanswerable),
        "categories": dict(Counter(e.get("category", "?") for e in entries)),
        # Which questions were asked, not just which corpus they were asked
        # about.  Two runs can share a corpus fingerprint and still be
        # incomparable because the question set grew between them — which is
        # exactly what happened when the FAQ questions were added.
        "questions_sha1": hashlib.sha1(
            "\n".join(sorted(e["question"] for e in entries)).encode("utf-8")
        ).hexdigest()[:16],
        "rebound": rebound,
        "corpus_now": now,
        "corpus_built_on": built_on or None,
        "corpus_changed": bool(built_on) and built_on.get("chunks_sha1") != now["chunks_sha1"],
    }

    if not quiet:
        spread = " + ".join(f"{n} {c}" for c, n in sorted(report["categories"].items()))
        print(f"🏆 golden set {len(entries)} คำถาม ({spread}) · corpus "
              f"{now['n_chunks']} chunks จาก {now['n_sources']} ไฟล์ ({now['chunks_sha1']})")
        if unanswerable and not include_unanswerable:
            print(f"   ตัดคำถามที่ไม่มีเฉลยออก {len(unanswerable)} ข้อ "
                  "(วัด retrieval ไม่ได้ — ใช้ include_unanswerable=True ถ้าต้องการ)")
        if rebound:
            print(f"   ↻ ผูกเฉลยใหม่ {rebound} คำถาม — chunk_id เลื่อนตำแหน่งตั้งแต่สร้าง set")
        if report["corpus_changed"]:
            print(f"   ⚠️  corpus ต่างจากตอนสร้าง set "
                  f"({built_on.get('n_chunks')} → {now['n_chunks']} chunks) — "
                  "เทียบกับผลวัดรอบก่อนได้เฉพาะเมื่อ fingerprint ตรงกัน")

    return entries, report
