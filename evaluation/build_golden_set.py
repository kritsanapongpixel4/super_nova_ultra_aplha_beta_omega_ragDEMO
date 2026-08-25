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

Three sources, all parsed out of the documents themselves so the whole set
can be rebuilt whenever the chunking changes — no hand labelling to go stale:

``clo``
    The CLO table.  Each course card is a single chunk, it is the only chunk
    describing that course, and both the name and the code are written on it,
    so the right answer is not a judgement call.  Each course is asked twice,
    once by name and once by code.

``faq``
    The ``ถาม:``/``ตอบ:`` pairs the registrar documents carry in their own
    "คำถามที่พบบ่อย" section — real questions, written by the people who
    wrote the documents.  These are what reach the corpus beyond the CLO
    table: 13 documents instead of one.

``unanswerable``
    The topics each document lists under "ข้อมูลที่เอกสารนี้ไม่ได้ระบุ".
    They have no relevant chunk on purpose — they exist to catch a system
    that invents an answer instead of saying it does not know.  Lines that
    point at another REG document are skipped: those *are* answerable from
    the corpus, just not from that one file.

An answer that spans a chunk boundary belongs to both chunks, so entries can
carry more than one key.  Score those with **hit@k**, not recall@k — recall
divides by the number of gold chunks, so a question whose answer sits in two
of them could never score above 0.5 however well retrieval did.  For the CLO
entries, which have exactly one gold chunk each, the two are identical.

Run: python evaluation/build_golden_set.py
"""

import json
import re
import sys
from collections import Counter
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

def _flat(text: str) -> str:
    """Collapse whitespace so a probe matches across the PDF's line breaks."""
    return " ".join(text.split())


# "ถาม: ยื่นลาพักการศึกษาที่ไหน / ตอบ: ยื่นผ่านคำร้องออนไลน์..."  The answer
# runs until the next question, or until a short line that is a new heading.
_FAQ_PAIR = re.compile(
    r"ถาม:\s*(.+?)\s*\nตอบ:\s*(.+?)"
    r"(?=\nถาม:|\n[ก-๙A-Za-z][^\n]{0,40}\n|\Z)",
    re.S,
)

# The block listing what the document deliberately does not cover.
_NOT_COVERED = re.compile(
    r"ข้อมูลที่เอกสารนี้ไม่ได้ระบุ[^\n]*\n(.+?)"
    r"(?=\nเอกสารที่เกี่ยวข้อง|\nที่มาและสถานะ|\nคำสำคัญ|\Z)",
    re.S,
)

# "...ไม่ได้ระบุในเอกสารฉบับนี้ แต่เอกสาร REG-13 ระบุไว้ว่า 1,000 บาท" — the
# corpus does answer this one, so it is not a test of refusing to answer.
_CROSS_REFERENCE = re.compile(r"REG-\d+")


# "รายวิชา 04-620-201 ปฏิบัติการควบคุมเวอร์ชัน สอดคล้องกับ PLO2, ..."
_COURSE_LINE = re.compile(
    r"^รายวิชา\s+(\d{2}-\d{3}-\d{3})\s+(.+?)\s+สอดคล้องกับ", re.MULTILINE
)


def build_clo(chunks: list[dict]) -> list[dict]:
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
                    "question": question,
                    "relevant_chunk_ids": [chunk_id],
                    "relevant_chunk_keys": [key],
                    "category": "clo",
                    "phrasing": phrasing,
                    "source": chunk.get("source", ""),
                    "course_code": code,
                    "course_name": name,
                }
            )

    if skipped:
        print(f"⚠️  ข้ามไป {len(skipped)} วิชาที่อ่านชื่อจากบรรทัดแรกไม่ได้: {skipped[:5]}")
    return entries


def build_faq(records: list[dict], chunks: list[dict]) -> list[dict]:
    """One question per ``ถาม:``/``ตอบ:`` pair the documents already carry.

    The gold chunk is located by searching for the question text, not the
    answer: the question is a distinctive sentence that appears once, while a
    short answer like "500 บาท" turns up in several places.  Chunks overlap by
    ``CHUNK_OVERLAP`` tokens, so a pair sitting on a boundary lands in two
    chunks and both are correct — hence a list, and hence hit@k.
    """
    by_source: dict[str, list[tuple[dict, str]]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.get("source", ""), []).append(
            (chunk, _flat(chunk["text"]))
        )

    entries: list[dict] = []
    unlocated: list[str] = []

    for record in records:
        source = record.get("source", "")
        for match in _FAQ_PAIR.finditer(record.get("text", "")):
            question = _flat(match.group(1))
            answer = _flat(match.group(2))
            found = [
                chunk for chunk, flat in by_source.get(source, []) if question in flat
            ]
            if not found:
                # The pair survived extraction but not chunking — most likely
                # is_noise() dropped it.  Emitting it anyway would be a
                # question no retriever could ever get right.
                unlocated.append(f"{source}: {question[:40]}")
                continue
            entries.append(
                {
                    "question": question,
                    "relevant_chunk_ids": [str(c["chunk_id"]) for c in found],
                    "relevant_chunk_keys": [chunk_key(c) for c in found],
                    "category": "faq",
                    "phrasing": "faq",
                    "source": source,
                    "reference_answer": answer,
                }
            )

    if unlocated:
        print(f"⚠️  หา chunk ของ {len(unlocated)} คำถาม FAQ ไม่เจอ: {unlocated[:3]}")
    return entries


def build_unanswerable(records: list[dict]) -> list[dict]:
    """Questions the documents say outright they do not answer.

    No gold chunk, on purpose.  Retrieval metrics cannot score these — they
    are here for ``evaluation/eval_generation.py``, which checks that the
    system says it does not know rather than inventing a number.
    """
    entries: list[dict] = []
    for record in records:
        match = _NOT_COVERED.search(record.get("text", ""))
        if not match:
            continue
        for line in match.group(1).split("\n"):
            topic = _flat(line)
            if len(topic) < 8 or _CROSS_REFERENCE.search(topic):
                continue
            # Drop the document's own aside — "(เอกสารนี้ระบุเฉพาะค่าปรับ 500
            # บาท)" is an explanation to the reader, not part of the topic.
            topic = _flat(re.sub(r"\(.*?\)", "", topic))
            entries.append(
                {
                    "question": topic,
                    "relevant_chunk_ids": [],
                    "relevant_chunk_keys": [],
                    "category": "unanswerable",
                    "phrasing": "unanswerable",
                    "source": record.get("source", ""),
                }
            )
    return entries


def main() -> None:
    if not config.CHUNKS_FILE.exists():
        print(f"❌ ไม่พบ {config.CHUNKS_FILE} — รัน pipeline/chunking.py ก่อน")
        sys.exit(1)

    if not config.EXTRACTED_TEXT_FILE.exists():
        print(f"❌ ไม่พบ {config.EXTRACTED_TEXT_FILE} — รัน pipeline/extract_text.py ก่อน")
        sys.exit(1)

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(config.EXTRACTED_TEXT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    entries = (
        build_clo(chunks)
        + build_faq(records, chunks)
        + build_unanswerable(records)
    )
    if not entries:
        print("❌ ไม่พบคำถามที่สร้างได้เลย — golden set สร้างไม่ได้")
        sys.exit(1)
    for number, entry in enumerate(entries, start=1):
        entry["query_id"] = number

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

    # Each course card is asked exactly twice, by name and by code, so a key
    # appearing more often means two cards hashed the same — one question
    # would then have two "right" answers and the metric would be a lie.
    clo_keys = [
        k
        for entry in entries
        if entry["category"] == "clo"
        for k in entry["relevant_chunk_keys"]
    ]
    collisions = {k for k in clo_keys if clo_keys.count(k) > 2}
    if collisions:
        print(f"❌ การ์ดวิชา {len(collisions)} ใบมีเนื้อความซ้ำกันจนแยกไม่ออก")
        sys.exit(1)

    by_category = Counter(entry["category"] for entry in entries)
    covered = {k for entry in entries for k in entry["relevant_chunk_keys"]}

    corpus = fingerprint(chunks)
    payload = {
        "meta": {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(entries),
            "by_category": dict(by_category),
            # How much of the corpus any question actually points at.  The
            # honest headline for what this set can and cannot detect.
            "chunks_covered": len(covered),
            "coverage": round(len(covered) / len(chunks), 4),
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

    answerable = sum(1 for e in entries if e["relevant_chunk_keys"])
    sources = len({e["source"] for e in entries if e["source"]})
    print(f"🏆 สร้าง golden set {len(entries)} คำถาม จาก {sources} เอกสาร")
    print(f"   clo          {by_category['clo']:3d}  ตาราง CLO "
          f"(ชื่อวิชา {sum(e['phrasing'] == 'by_name' for e in entries)}, "
          f"รหัสวิชา {sum(e['phrasing'] == 'by_code' for e in entries)})")
    print(f"   faq          {by_category['faq']:3d}  คำถามที่พบบ่อยในเอกสารทะเบียน")
    print(f"   unanswerable {by_category['unanswerable']:3d}  เอกสารระบุเองว่าไม่มีคำตอบ "
          f"(ไม่มี chunk เฉลย ใช้กับ eval_generation)")
    print(f"   วัด retrieval ได้ {answerable} คำถาม · ครอบคลุม {len(covered)}/{len(chunks)} "
          f"chunks ({len(covered) / len(chunks):.1%})")
    print(f"   corpus: {corpus['n_chunks']} chunks จาก {corpus['n_sources']} ไฟล์ "
          f"({corpus['chunks_sha1']})")
    print(f"💾 บันทึกไว้ที่ {path}")


if __name__ == "__main__":
    main()
