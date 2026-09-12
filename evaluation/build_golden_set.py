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

# A line break inside the list is not an item break.  These lists wrap:
#
#     จำนวนเงินค่าธรรมเนียมลาพักการศึกษา ... แต่เอกสาร REG-13 ระบุไว้ว่า
#     1,000 บาท
#
# Splitting on the newline made "1,000 บาท" a question of its own, and worse,
# it escaped the REG- filter that had just excluded the line it belongs to —
# so a cross-reference became an "unanswerable" question whose answer sits one
# line above it.  The same wrap split "...อย่างน้อย 2 / สัปดาห์)" into a
# truncated question plus the fragment "สัปดาห์)".
#
# A line continues the one before it when that line cannot have ended an item:
# it leaves a bracket open, or trails off on a connective or a bare number.
# Checked against all 25 lines the 13 registrar documents actually carry — it
# joins exactly the three real wraps and splits every genuine item.
_CONTINUES = re.compile(r"(?:ว่า|ที่|และ|หรือ|เช่น|คือ|\d)$")


def _unwrap(block: str) -> list[str]:
    """Split a "not covered" list into items, honouring wrapped lines."""
    items: list[str] = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if items and (
            _CONTINUES.search(items[-1])
            or items[-1].count("(") > items[-1].count(")")
        ):
            items[-1] = f"{items[-1]} {line}"
        else:
            items.append(line)
    return items

# Written by evaluation/paraphrase_faq.py — see build_faq() for why.
_PARAPHRASE_FILE = config.DATA_DIR / "faq_paraphrases.json"


# "รายวิชา 04-620-201 ปฏิบัติการควบคุมเวอร์ชัน สอดคล้องกับ PLO2, ..."
_COURSE_LINE = re.compile(
    r"^รายวิชา\s+(\d{2}-\d{3}-\d{3})\s+(.+?)\s+สอดคล้องกับ", re.MULTILINE
)

# How the two หลักสูตร documents list a course — four lines, always in this
# order:
#
#     01-110-004 สังคมกับสิ่งแวดล้อม
#     Society and Environment
#     3(3-0)
#
# Regular enough to build questions from without an LLM, which keeps this
# script offline and its output reproducible.
# "3(3-0)" / "3(2-2-5)" — the credit figure.  Its presence is what separates a
# chunk that answers "which code, how many credits" from one that only names
# the course in a prerequisite list.
_CREDIT_FIGURE = re.compile(r"\d+\s*\(")
_COURSE_CODE = re.compile(r"\d{2}-\d{3}-\d{3}")

_CURRICULUM_COURSE = re.compile(
    r"^(\d{2}-\d{3}-\d{3})[ \t]+([^\n]{3,80})\n"
    r"([A-Za-z][^\n]{2,90})\n"
    r"(\d+)\(([^)\n]{1,12})\)",
    re.MULTILINE,
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


def build_curriculum(chunks: list[dict]) -> list[dict]:
    """One question per phrasing per course listed in the หลักสูตร documents.

    Those two PDFs are 1,718 of the corpus's 2,202 chunks and had no question
    pointing at them at all, so every retrieval number measured so far said
    nothing about 78% of what the system can be asked.  The CLO table covers
    the same courses' learning outcomes but is a different document with
    different wording, and 64 chunks of it.

    A course is gold wherever it is defined, not in one chosen chunk: the same
    code is listed in both the 2563 and 2568 curricula, and the chunk overlap
    repeats it again inside each.  Naming one of those the right answer and
    the identical text beside it wrong would measure nothing but which copy
    the retriever happened to reach.  Note that this caps recall@k — four gold
    chunks cannot fit in top-1 — so read hit@k here, not recall.

    "Defined" is wider than the four-line entry.  The first version of this
    builder marked only those, and the generation run then scored thirteen
    curriculum answers as retrieval misses while the system had in fact
    answered every one of them correctly, out of the semester plan or the
    course list — chunks that state the code beside its credit figure and
    answer "which code, how many credits" exactly as well as the entry does.
    Measured over the 288 questions, that labelling alone was most of the
    apparent failure:

                    เฉลยแบบแคบ        เฉลยที่ตอบได้จริง
        by_code    13.9% / 100.0%     52.8% / 100.0%
        by_name     7.6% /  35.4%     31.2% /  61.8%   (Hit@1 / Hit@5)

    Widening it is not grading on a curve: a chunk reading "04-621-302
    อินเตอร์เน็ตของสรรพสิ่ง 3(3-0-6)" answers the question, and calling it
    wrong measures the label, not the retriever.  by_name stays the weak
    spot either way, which is the finding worth keeping.
    """
    by_code: dict[str, dict] = {}

    for chunk in chunks:
        for code, thai, english, credits, hours in _CURRICULUM_COURSE.findall(
            chunk["text"]
        ):
            card = by_code.setdefault(
                code,
                {
                    "name": thai.strip(),
                    "english": english.strip(),
                    "credits": credits,
                    "hours": hours,
                    "keys": [],
                    "ids": [],
                    "sources": set(),
                },
            )
            key = chunk_key(chunk)
            if key not in card["keys"]:
                card["keys"].append(key)
                card["ids"].append(str(chunk["chunk_id"]))
            card["sources"].add(chunk.get("source", ""))

    # Second pass: everything else that answers the same question.  Runs after
    # the entries are known so a course nobody defines never gets invented
    # here out of a passing mention.
    for chunk in chunks:
        text = chunk.get("text", "")
        if not _CREDIT_FIGURE.search(text):
            continue
        key = chunk_key(chunk)
        for code in set(_COURSE_CODE.findall(text)):
            card = by_code.get(code)
            if card is None or key in card["keys"]:
                continue
            card["keys"].append(key)
            card["ids"].append(str(chunk["chunk_id"]))
            card["sources"].add(chunk.get("source", ""))

    entries: list[dict] = []
    for code, card in sorted(by_code.items()):
        for phrasing, question in (
            ("by_name", f"วิชา{card['name']}รหัสวิชาอะไร และกี่หน่วยกิต"),
            ("by_code", f"วิชา {code} ชื่อวิชาอะไร และกี่หน่วยกิต"),
        ):
            entries.append(
                {
                    "question": question,
                    "relevant_chunk_ids": card["ids"],
                    "relevant_chunk_keys": card["keys"],
                    "category": "curriculum",
                    "phrasing": phrasing,
                    # Several documents can define one course, so the single
                    # "source" field the other builders fill would be a
                    # coin flip.  doc_hit@k reads this list instead.
                    "source": sorted(card["sources"])[0],
                    "sources": sorted(card["sources"]),
                    "course_code": code,
                    "course_name": card["name"],
                    "credits": card["credits"],
                }
            )
    return entries


def build_faq(records: list[dict], chunks: list[dict]) -> list[dict]:
    """One question per ``ถาม:``/``ตอบ:`` pair the documents already carry.

    The gold chunk is located by searching for the question text, not the
    answer: the question is a distinctive sentence that appears once, while a
    short answer like "500 บาท" turns up in several places.  Chunks overlap by
    ``CHUNK_OVERLAP`` tokens, so a pair sitting on a boundary lands in two
    chunks and both are correct — hence a list, and hence hit@k.

    That search is also why the *asked* question has to come from
    ``data/faq_paraphrases.json`` rather than from the document.  Locating the
    chunk by its text guarantees the text is in the chunk, so using it as the
    query measures string matching: ``tfidf-char`` scored 94.5% on these
    against dense's 63.6%, and it is the weakest retriever in the set.  The
    document's wording still finds the chunk; the paraphrase is what gets
    asked.  Without the file the pairs are skipped — a contaminated number is
    worse than a missing one.
    """
    paraphrases: dict[str, str] = {}
    if _PARAPHRASE_FILE.exists():
        with open(_PARAPHRASE_FILE, "r", encoding="utf-8") as f:
            paraphrases = json.load(f)
    by_source: dict[str, list[tuple[dict, str]]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.get("source", ""), []).append(
            (chunk, _flat(chunk["text"]))
        )

    entries: list[dict] = []
    unlocated: list[str] = []
    unparaphrased: list[str] = []

    for record in records:
        source = record.get("source", "")
        for match in _FAQ_PAIR.finditer(record.get("text", "")):
            question = _flat(match.group(1))
            answer = _flat(match.group(2))
            # Both the question and the answer locate the pair, and either
            # chunk answers it.  Searching only for the question missed the
            # case where an answer runs past a chunk boundary: the tail chunk
            # holds the actual answer text but not the question, so a
            # retriever that found it scored 0.  Exact substring both ways —
            # no fuzzy matching, because a golden set built on guesses is the
            # bug this whole file exists to avoid.
            probe = answer[:35]
            found = [
                chunk
                for chunk, flat in by_source.get(source, [])
                if question in flat or probe in flat
            ]
            if not found:
                # The pair survived extraction but not chunking — most likely
                # is_noise() dropped it.  Emitting it anyway would be a
                # question no retriever could ever get right.
                unlocated.append(f"{source}: {question[:40]}")
                continue
            asked = paraphrases.get(question)
            if not asked:
                unparaphrased.append(question)
                continue
            entries.append(
                {
                    "question": asked,
                    "relevant_chunk_ids": [str(c["chunk_id"]) for c in found],
                    "relevant_chunk_keys": [chunk_key(c) for c in found],
                    "category": "faq",
                    "phrasing": "faq",
                    "source": source,
                    # Kept so the pair can be traced back to the document, and
                    # so a rebuild finds the same chunk after re-chunking.
                    "document_question": question,
                    "reference_answer": answer,
                }
            )

    if unlocated:
        print(f"⚠️  หา chunk ของ {len(unlocated)} คำถาม FAQ ไม่เจอ: {unlocated[:3]}")
    if unparaphrased:
        print(f"⚠️  ข้าม {len(unparaphrased)} คำถาม FAQ ที่ยังไม่มีคำถามเขียนใหม่ — "
              "รัน python evaluation/paraphrase_faq.py")
    return entries


# Audited 2026-09-12 against all 2,202 chunks.  A document's own "ไม่ได้ระบุ"
# list means "this file does not say"; the chatbot answers from all eighteen,
# and for these the corpus does say — so grading a correct answer as a refusal
# failure would measure the label.  Quoted evidence, so the call can be checked:
#
#   ...แจ้งสำเร็จการศึกษา   คู่มือนักศึกษา69: "ยื่นภายใน 15 วัน นับตั้งแต่วันเปิด
#                            ภาคการศึกษา"
#   ...ขอคืนสภาพภายในกี่วัน  คู่มือนักศึกษา69: "ต้องไม่พ้นกำหนดระยะเวลา 1 ปี นับจาก
#                            วันที่ถูกประกาศถอนชื่อ"
#   ...ลาพักต่อเนื่องได้     คู่มือนักศึกษา69: "จะลาพักการศึกษาเกินกว่า 2 ภาค
#                            การศึกษาปกติติดต่อกันไม่ได้"
#   ...กลับเข้าศึกษา         REG-13 itself: "ควรยื่นคำร้องก่อนวันลงทะเบียนอย่างน้อย
#                            2 สัปดาห์" — the document's own aside said it gives
#                            only this and not a hard deadline, but that aside is
#                            a parenthetical, and parentheses are stripped from
#                            the question.  What is left asks for something the
#                            document does answer.
#
# The last one is excluded for a second reason: "ค่าธรรมเนียมของคำร้องนี้" has no
# referent once it leaves its document, so no retriever can be expected to know
# which form "นี้" means.  It tests the question, not the system.
_ANSWERED_ELSEWHERE = {
    "วันสุดท้ายของการแจ้งสำเร็จการศึกษาในระบบ ให้ดูจากปฏิทินการศึกษาของแต่ละภาคการศึกษา",
    "กำหนดเวลาว่าต้องยื่นขอคืนสภาพภายในกี่วันหลังถูกถอนชื่อ",
    "จำนวนภาคการศึกษาสูงสุดที่ลาพักต่อเนื่องได้",
    "กำหนดเวลาที่ต้องยื่นขอกลับเข้าศึกษาอย่างช้าที่สุด",
    "ค่าธรรมเนียมของคำร้องนี้",
}


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
        for line in _unwrap(match.group(1)):
            topic = _flat(line)
            if len(topic) < 8 or _CROSS_REFERENCE.search(topic):
                continue
            # Drop the document's own aside — "(เอกสารนี้ระบุเฉพาะค่าปรับ 500
            # บาท)" is an explanation to the reader, not part of the topic.
            topic = _flat(re.sub(r"\(.*?\)", "", topic))
            if topic in _ANSWERED_ELSEWHERE:
                continue
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
        + build_curriculum(chunks)
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
    def _phrasings(category: str) -> str:
        same = [e for e in entries if e["category"] == category]
        return (f"(ชื่อวิชา {sum(e['phrasing'] == 'by_name' for e in same)}, "
                f"รหัสวิชา {sum(e['phrasing'] == 'by_code' for e in same)})")

    print(f"   clo          {by_category['clo']:3d}  ตาราง CLO {_phrasings('clo')}")
    print(f"   curriculum   {by_category['curriculum']:3d}  รายวิชาในเล่มหลักสูตร "
          f"{_phrasings('curriculum')}")
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
