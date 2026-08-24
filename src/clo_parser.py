"""Parse the PLO × CLO matrix PDF into retrievable sentences.

``CLOs-Computer_Engineering-RMUTT.pdf`` is a wide landscape table: each row
is a course or one of its CLOs, and eight PLO columns carry ✓ marks.  Plain
text extraction throws the geometry away, leaving hundreds of chunks whose
entire content is ``'✓'`` — real information that no embedding model can
use.

This module rebuilds the relationships from word coordinates and writes
them as sentences instead::

    รายวิชา 04-620-201 ปฏิบัติการควบคุมเวอร์ชัน สอดคล้องกับ PLO2, PLO4, PLO5, PLO8
    รายวิชา 04-620-201 ปฏิบัติการควบคุมเวอร์ชัน — CLO 1 สามารถประยุกต์ใช้งาน… สอดคล้องกับ PLO2

``page.get_text("words")`` is used rather than ``page.find_tables()``: the
table extractor mis-assembles Thai combining marks (``ความสัมพนั ธ์`` instead
of ``ความสัมพันธ์``), while the word extractor returns them correctly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .thai_normalize import normalize_text

logger = logging.getLogger(__name__)

_COURSE_RE = re.compile(r"^(\d{2}-\d{3}-\d{3})\s*(.*)$", re.DOTALL)
_CLO_RE = re.compile(r"^CLO\s*(\d+)\s*(.*)$", re.DOTALL)
_PLO_RE = re.compile(r"^PLO\d+$")
_MARKS = {"✓", "✔", "√", "P"}

# Words on the same table row can sit a point or two apart vertically —
# the ✓ marks are consistently ~1pt below their text.  Row pitch is ~19pt,
# so a 4pt window merges them without ever merging two rows.
_ROW_TOLERANCE = 4.0


def _plo_columns(words: list[tuple]) -> dict[str, float]:
    """Map each PLO name to the x position of its column header."""
    return {
        text: x0
        for x0, _y0, _x1, _y1, text, *_ in words
        if _PLO_RE.match(text)
    }


def _group_rows(words: list[tuple], tolerance: float = _ROW_TOLERANCE) -> list[list[tuple]]:
    """Cluster words into visual rows by their y coordinate."""
    rows: list[list[tuple]] = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        if rows and abs(word[1] - rows[-1][0][1]) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(r, key=lambda w: w[0]) for r in rows]


def _marked_plos(row: list[tuple], columns: dict[str, float]) -> list[str]:
    """Return the PLOs whose column carries a ✓ on this row."""
    hits: list[str] = []
    for x0, _y0, _x1, _y1, text, *_ in row:
        if text.strip() not in _MARKS:
            continue
        # The glyph is drawn a few points right of the header's left edge,
        # so attribute it to the nearest column rather than an exact match.
        nearest = min(columns, key=lambda name: abs(columns[name] - x0))
        if abs(columns[nearest] - x0) < 20:
            hits.append(nearest)
    return sorted(set(hits), key=lambda name: columns[name])


def _row_text(row: list[tuple], first_plo_x: float) -> str:
    """Join the label words that sit left of the PLO columns."""
    return " ".join(
        text for x0, _y0, _x1, _y1, text, *_ in row if x0 < first_plo_x - 5
    ).strip()


def looks_like_clo_matrix(page) -> bool:
    """True if *page* carries the PLO column headers this parser needs."""
    text = page.get_text()
    return "PLO1" in text and "PLO8" in text and "CLO" in text


def parse(path: Path) -> list[dict[str, Any]] | None:
    """Turn a PLO × CLO matrix PDF into sentence records.

    Returns ``None`` when *path* is not such a matrix, so callers can fall
    back to normal text extraction.
    """
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        if not doc.page_count or not looks_like_clo_matrix(doc[0]):
            return None

        records: list[dict[str, Any]] = []
        course = ""          # "04-620-201 ปฏิบัติการควบคุมเวอร์ชัน"
        pending: dict | None = None   # entry still collecting wrapped lines

        def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            label = normalize_text(" ".join(pending["parts"]).strip())
            plos = pending["plos"]
            if not label:
                pending = None
                return
            suffix = f" สอดคล้องกับ {', '.join(plos)}" if plos else ""
            records.append({
                "text": f"{pending['prefix']}{label}{suffix}",
                "page": pending["page"],
            })
            pending = None

        for page_no, page in enumerate(doc, start=1):
            columns = _plo_columns(page.get_text("words"))
            if not columns:
                continue
            first_plo_x = min(columns.values())

            for row in _group_rows(page.get_text("words")):
                label = _row_text(row, first_plo_x)
                plos = _marked_plos(row, columns)

                if not label:
                    # A ✓-only row belongs to the entry above it.
                    if pending is not None:
                        pending["plos"] = sorted(
                            set(pending["plos"]) | set(plos),
                            key=lambda name: columns[name],
                        )
                    continue

                if _PLO_RE.match(label.split(" ")[0]) or "ความสัมพันธ์ระหว่าง" in label:
                    continue  # repeated page header

                course_match = _COURSE_RE.match(label)
                clo_match = _CLO_RE.match(label)

                if course_match:
                    flush()
                    code, name = course_match.groups()
                    course = normalize_text(f"{code} {name}".strip())
                    pending = {
                        "prefix": "รายวิชา ",
                        "parts": [label],
                        "plos": plos,
                        "page": page_no,
                    }
                elif clo_match:
                    flush()
                    number, body = clo_match.groups()
                    pending = {
                        "prefix": f"รายวิชา {course} — CLO {number} ",
                        "parts": [body],
                        "plos": plos,
                        "page": page_no,
                    }
                elif pending is not None:
                    # Wrapped continuation of the previous label.
                    pending["parts"].append(label)
                    pending["plos"] = sorted(
                        set(pending["plos"]) | set(plos),
                        key=lambda name: columns[name],
                    )

        flush()

    for index, record in enumerate(records):
        record["id"] = index
        record["source"] = path.name
        # Downstream expects line_start/line_end; this table has no lines,
        # so both carry the page number the relationship was read from.
        record["line_start"] = record["line_end"] = record.pop("page")

    logger.info("  📊 %s → %d records (ตาราง PLO-CLO)", path.name, len(records))
    return records
