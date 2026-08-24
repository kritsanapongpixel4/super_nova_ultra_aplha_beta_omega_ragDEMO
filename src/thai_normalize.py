"""Repair Thai text extracted from PDFs that use legacy (non-Unicode) fonts.

Many Thai PDFs embed fonts whose ``ToUnicode`` map points vowels, tone marks
and symbols into the **Private Use Area** (U+E000–U+F8FF) instead of real
Unicode code points.  PyMuPDF faithfully returns those PUA characters, so
``ข้าพเจ้า`` arrives as ``ข\uf70bาพเจ\uf70bา`` — a word no embedding model or
BM25 tokeniser will ever match.

This module maps them back before anything downstream sees the text.
"""

from __future__ import annotations

import unicodedata

from pythainlp.util import normalize as thai_normalize


# ── Thai combining marks (Adobe PUA set, U+F700–U+F71A) ─────────────────
# Legacy Thai fonts carry several *positional variants* of the same mark
# (raised / lowered / left-shifted) so it can sit correctly on tall or
# descending consonants.  They all collapse back to one real code point.
#
# Verified against this corpus: F702→ปีการศึกษา, F705→ฝ่ายวิชาการ,
# F70A→ไม่, F70B→เข้า, F70E→อาจารย์, F710→ปัจจุบัน, F712→เป็น.
_THAI_MARKS: dict[int, str] = {
    0xF700: "\u0E39",  # ◌ู  sara uu
    0xF701: "\u0E38",  # ◌ุ  sara u
    0xF702: "\u0E35",  # ◌ี  sara ii
    0xF703: "\u0E37",  # ◌ื  sara uee
    0xF704: "\u0E36",  # ◌ึ  sara ue
    0xF705: "\u0E48",  # ◌่  mai ek
    0xF706: "\u0E49",  # ◌้  mai tho
    0xF707: "\u0E4A",  # ◌๊  mai tri
    0xF708: "\u0E4B",  # ◌๋  mai chattawa
    0xF709: "\u0E4C",  # ◌์  thanthakhat
    0xF70A: "\u0E48",  # ◌่
    0xF70B: "\u0E49",  # ◌้
    0xF70C: "\u0E4A",  # ◌๊
    0xF70D: "\u0E4B",  # ◌๋
    0xF70E: "\u0E4C",  # ◌์
    0xF70F: "\u0E4D",  # ◌ํ  nikhahit
    0xF710: "\u0E31",  # ◌ั  mai han akat
    0xF711: "\u0E34",  # ◌ิ  sara i
    0xF712: "\u0E47",  # ◌็  mai taikhu
    0xF713: "\u0E48",  # ◌่
    0xF714: "\u0E49",  # ◌้
    0xF715: "\u0E4A",  # ◌๊
    0xF716: "\u0E4B",  # ◌๋
    0xF717: "\u0E4C",  # ◌์
    0xF718: "\u0E4D",  # ◌ํ
    0xF719: "\u0E4E",  # ◌๎  yamakkan
    0xF71A: "\u0E3A",  # ◌ฺ  phinthu
}

# ── Wingdings / Symbol glyphs (U+F020–U+F0FF) ───────────────────────────
# Those fonts are mapped as ASCII + 0xF000.  In these documents they are
# only ever form furniture — checkboxes in front of options, bullets in
# front of list items — so they collapse to two neutral markers rather
# than being deleted, which would glue neighbouring words together.
_SYMBOLS: dict[int, str] = {
    0xF0FC: "✔",   # Wingdings check mark ("ทำเครื่องหมาย ✔ ลงใน")
    0xF035: "☐",
    0xF050: "☐",
    0xF052: "☐",
    0xF063: "☐",   # "อาจารย์ที่ปรึกษา ☐ เห็นควรอนุมัติ"
    0xF06C: "☐",
    0xF06F: "☐",
    0xF0A3: "☐",   # "Comment ☐ อนุมัติ / ☐ ไม่อนุมัติ"
    0xF0A8: "☐",
    0xF03D: "•",
    0xF097: "•",
    0xF0B7: "•",   # Symbol middle dot → bullet
}

_PUA_MAP: dict[int, str] = {**_THAI_MARKS, **_SYMBOLS}

# Anything still left in the PUA after the table above is an unmapped
# decorative glyph.  A space keeps word boundaries intact; dropping it
# outright would fuse the words on either side.
_PUA_FALLBACK = " "


def has_pua(text: str) -> bool:
    """True if *text* still contains Private Use Area characters."""
    return any("\uE000" <= ch <= "\uF8FF" for ch in text)


def fix_pua(text: str) -> str:
    """Replace every Private Use Area character with its real counterpart."""
    if not has_pua(text):
        return text
    return "".join(
        _PUA_MAP.get(ord(ch), _PUA_FALLBACK) if "\uE000" <= ch <= "\uF8FF" else ch
        for ch in text
    )


def normalize_text(text: str) -> str:
    """Full clean-up for one extracted passage.

    1. Map PUA characters back to real Unicode.
    2. NFC-compose so combining marks sit in canonical order.
    3. Run PyThaiNLP's normaliser (removes duplicate tone marks and
       reorders misplaced ones — common in text recovered from PDFs).

    Step 3 runs **line by line**: ``thai_normalize`` collapses runs of
    newlines, and ``parse_document`` needs the blank lines intact to tell
    one passage from the next.
    """
    if not text:
        return text
    text = fix_pua(text)
    text = unicodedata.normalize("NFC", text)
    lines = text.split("\n")
    return "\n".join(
        thai_normalize(line) if line.strip() else line
        for line in lines
    )
