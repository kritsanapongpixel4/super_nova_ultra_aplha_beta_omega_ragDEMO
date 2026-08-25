"""Generate the evaluation set (data/golden_set.json).

Each entry pairs a question with the chunk that genuinely answers it, named
two ways:

    {"query_id": 1, "question": "...",
     "relevant_chunk_ids":  ["12"],           # position today
     "relevant_chunk_keys": ["a3f1c8..."]}    # hash of source + text

The id is what the metrics compare against; the key is what survives the
corpus growing.  ``evaluation/golden_set.py`` re-resolves one from the other
at load time — see its docstring for the renumbering bug that made this
necessary.

The questions come from the CLO table, because that is the one part of this
corpus where the right answer is not a judgement call: each course card is a
single chunk, it is the only chunk that describes that course, and both the
course name and its code are written on it.  So each course is asked twice —
once by name, once by code — and the card is the sole correct answer.

That gives a set that can be built and rebuilt automatically, with no hand
labelling to go stale when the chunking settings change.  What it cannot do
is speak for the rest of the corpus: 64 course cards out of 3,128 chunks,
all with the same shape.  A model that wins here has been shown to be better
at finding course cards, and nothing more.  See the README for that caveat
in full.

Run: python evaluation/build_golden_set.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from evaluation.golden_set import chunk_key, fingerprint  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# "รายวิชา 04-620-201 ปฏิบัติการควบคุมเวอร์ชัน สอดคล้องกับ PLO2, ..."
_COURSE_LINE = re.compile(
    r"^รายวิชา\s+(\d{2}-\d{3}-\d{3})\s+(.+?)\s+สอดคล้องกับ", re.MULTILINE
)


def build(chunks: list[dict]) -> list[dict]:
    """One question per phrasing per course card."""
    entries: list[dict] = []
    skipped: list[str] = []

    for chunk in chunks:
        code = chunk.get("course_code")
        if not code:
            continue
        match = _COURSE_LINE.search(chunk["text"])
        if not match:
            # A card whose first line does not parse cannot be asked about by
            # name; say so rather than quietly emitting half a pair.
            skipped.append(code)
            continue

        name = match.group(2).strip()
        chunk_id = str(chunk["chunk_id"])
        key = chunk_key(chunk)
        for phrasing, question in (
            ("by_name", f"วิชา{name}มี CLO อะไรบ้าง"),
            ("by_code", f"CLO ของวิชา {code} มีอะไรบ้าง"),
        ):
            entries.append(
                {
                    "query_id": len(entries) + 1,
                    "question": question,
                    "relevant_chunk_ids": [chunk_id],
                    "relevant_chunk_keys": [key],
                    "phrasing": phrasing,
                    "course_code": code,
                    "course_name": name,
                }
            )

    if skipped:
        print(f"⚠️  ข้ามไป {len(skipped)} วิชาที่อ่านชื่อจากบรรทัดแรกไม่ได้: {skipped[:5]}")
    return entries


def main() -> None:
    if not config.CHUNKS_FILE.exists():
        print(f"❌ ไม่พบ {config.CHUNKS_FILE} — รัน pipeline/chunking.py ก่อน")
        sys.exit(1)

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    entries = build(chunks)
    if not entries:
        print("❌ ไม่พบการ์ดวิชาใน chunks.json — golden set สร้างไม่ได้")
        sys.exit(1)

    # The whole set is worthless if an answer does not resolve, and that
    # failure is invisible at scoring time: every metric simply reads 0.
    #
    # Checking the id is not enough — that was the guard that let the
    # renumbering through, because id "0" still existed, it just meant a
    # different chunk.  Check the content key, which cannot be right by
    # accident, and check that no two cards hash the same: duplicate keys
    # would make one question answerable by two chunks and the metric a lie.
    keys = {chunk_key(chunk) for chunk in chunks}
    dangling = [
        entry for entry in entries if set(entry["relevant_chunk_keys"]) - keys
    ]
    if dangling:
        print(f"❌ มี {len(dangling)} คำถามที่ชี้ไป chunk ที่ไม่มีอยู่จริง")
        sys.exit(1)

    answer_keys = [k for entry in entries for k in entry["relevant_chunk_keys"]]
    collisions = {k for k in answer_keys if answer_keys.count(k) > 2}
    if collisions:
        print(f"❌ การ์ดวิชา {len(collisions)} ใบมีเนื้อความซ้ำกันจนแยกไม่ออก")
        sys.exit(1)

    corpus = fingerprint(chunks)
    payload = {
        "meta": {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(entries),
            # What the set was measured against.  Two benchmark runs are
            # comparable only if this matches; golden_set.load() says so when
            # it does not.
            "corpus": corpus,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "queries": entries,
    }

    path = config.DATA_DIR / "golden_set.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    courses = len({entry["course_code"] for entry in entries})
    print(f"🏆 สร้าง golden set {len(entries)} คำถาม จาก {courses} วิชา")
    print(f"   สำนวน: ถามด้วยชื่อวิชา {sum(e['phrasing'] == 'by_name' for e in entries)}, "
          f"ถามด้วยรหัสวิชา {sum(e['phrasing'] == 'by_code' for e in entries)}")
    print(f"   corpus: {corpus['n_chunks']} chunks จาก {corpus['n_sources']} ไฟล์ "
          f"({corpus['chunks_sha1']})")
    print(f"💾 บันทึกไว้ที่ {path}")


if __name__ == "__main__":
    main()
