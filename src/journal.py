"""Experiment journal — an append-only record of what was tried and when.

Two files, written together on every call:

    logs/EXPERIMENTS.md    dated entries, meant to be read by a person
    logs/experiments.jsonl one JSON object per entry, meant to be queried

Both because they answer different questions.  "What did I do on Tuesday and
why" is a reading question; "chart encode speed against model size across
every run" is a query.  Keeping only the markdown means re-parsing prose
later; keeping only the JSONL means nobody ever reads it.

Nothing here overwrites: entries are appended, so a re-run adds to the record
rather than replacing it, and a run that failed stays in the history next to
the one that worked.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
MD_PATH = LOGS_DIR / "EXPERIMENTS.md"
JSONL_PATH = LOGS_DIR / "experiments.jsonl"

_HEADER = """# สมุดบันทึกการทดลอง

บันทึกอัตโนมัติจาก `src/journal.py` — ทุกครั้งที่รัน pipeline หรือ benchmark
จะต่อท้ายไฟล์นี้ ไม่มีการเขียนทับ เรียงจากเก่าไปใหม่
"""


def append(
    title: str,
    *,
    ok: bool = True,
    what: str = "",
    how: str = "",
    result: str = "",
    **data: Any,
) -> None:
    """Add one dated entry to both journal files.

    Args:
        title:  short name of the experiment, e.g. "encode: qwen3-0.6b"
        ok:     did it work — a failed attempt is still worth recording
        what:   what was being tested
        how:    the settings that made this run different from the last
        result: the outcome in one line, with the numbers that matter
        data:   any structured measurements, kept verbatim in the JSONL
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()

    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "title": title,
        "ok": ok,
        "what": what,
        "how": how,
        "result": result,
        **data,
    }
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    if not MD_PATH.exists():
        MD_PATH.write_text(_HEADER, encoding="utf-8")

    mark = "✅" if ok else "❌"
    lines = [f"\n## {mark} {now:%Y-%m-%d %H:%M} — {title}\n"]
    for label, text in (
        ("ทดลองอะไร", what),
        ("ทำอย่างไร", how),
        ("ผลลัพธ์", result),
    ):
        if text:
            lines.append(f"- **{label}:** {text}")
    if data:
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
    lines.append("")

    with open(MD_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def entries() -> list[dict[str, Any]]:
    """Every entry recorded so far, oldest first — for building comparisons."""
    if not JSONL_PATH.exists():
        return []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
