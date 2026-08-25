"""Rewrite the FAQ questions so they are not copied out of the corpus.

``build_golden_set.py`` locates an FAQ pair's gold chunk by searching for the
question text inside it.  That works, but it guarantees the question appears
in its own answer word for word — and a benchmark built that way measures
string matching, not retrieval.  The evidence: ``tfidf-char``, a bag of
character n-grams with no notion of meaning and the worst retriever in the
comparison (6.2% on course names), scores **94.5%** on those questions.  The
dense retriever scores 63.6% on the same ones.  When the dumbest method wins
by thirty points, the task is not the one anybody meant to measure.

So each question is rewritten the way a student would actually type it —
same thing being asked, different words — and the rewrite is what goes into
the golden set.  The original still does the job of finding the gold chunk;
it just stops being the query.

Written to a file rather than generated during the build on purpose:

- ``build_golden_set.py`` stays offline and deterministic.  Rebuilding after
  a chunking change must not depend on an API key or a daily quota.
- An LLM rewriting test questions can drift off the meaning.  A file can be
  read and corrected by a person; a call inside the build cannot.

Run once, then commit the result.  Re-run only when documents are added.

    python evaluation/paraphrase_faq.py            # เขียนคำถามที่ยังไม่มีเท่านั้น
    python evaluation/paraphrase_faq.py --force    # เขียนใหม่ทั้งหมด

Output: data/faq_paraphrases.json  {original question: rewritten question}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from evaluation.build_golden_set import _FAQ_PAIR, _flat  # noqa: E402
from src.generator import Generator  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PARAPHRASE_FILE = config.DATA_DIR / "faq_paraphrases.json"
BATCH = 10

_PROMPT = """เขียนคำถามต่อไปนี้ใหม่ ให้เป็นแบบที่นักศึกษาจริงจะพิมพ์ถามแชทบอท

กติกา:
- ถามเรื่องเดิมทุกประการ ห้ามเปลี่ยนความหมายหรือเพิ่มเงื่อนไขใหม่
- ใช้คำให้ต่างจากเดิมมากที่สุดเท่าที่ยังสื่อความเดิมได้ เลี่ยงการลอกวลีเดิมมาทั้งท่อน
- เขียนแบบภาษาพูดสั้น ๆ ได้ ไม่ต้องเป็นทางการ
- ห้ามตอบคำถาม ให้เขียนเฉพาะคำถามใหม่

ตอบเป็น JSON array ของสตริง เรียงตามลำดับเดิม ความยาวเท่าเดิม ไม่ต้องอธิบาย:

{questions}"""


def rewrite(generator: Generator, questions: list[str]) -> list[str] | None:
    """Rewrite one batch, or None if the reply cannot be used."""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    try:
        text = generator.complete(_PROMPT.format(questions=numbered), max_tokens=2000)
    except Exception as exc:
        print(f"   ❌ {type(exc).__name__}: {exc}")
        return None

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        print(f"   ❌ ไม่ใช่ JSON array: {text[:100]}")
        return None
    try:
        out = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        print(f"   ❌ JSON เสีย: {exc}")
        return None

    if len(out) != len(questions) or not all(isinstance(q, str) and q.strip() for q in out):
        # A short batch would silently pair questions with the wrong rewrite.
        print(f"   ❌ ได้ {len(out)} ข้อ แต่ส่งไป {len(questions)}")
        return None
    return [_flat(q) for q in out]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase the FAQ questions.")
    parser.add_argument("--force", action="store_true", help="เขียนใหม่ทั้งหมด")
    args = parser.parse_args()

    with open(config.EXTRACTED_TEXT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    questions = [
        _flat(m.group(1))
        for record in records
        for m in _FAQ_PAIR.finditer(record.get("text", ""))
    ]

    existing: dict[str, str] = {}
    if PARAPHRASE_FILE.exists() and not args.force:
        with open(PARAPHRASE_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    todo = [q for q in questions if q not in existing]
    print(f"❓ คำถาม FAQ {len(questions)} ข้อ · มีอยู่แล้ว {len(existing)} · ต้องเขียนใหม่ {len(todo)}")
    if not todo:
        print("✅ ครบแล้ว ไม่ต้องเรียก API")
        return

    generator = Generator(
        model=config.LLM_MODEL, fallback_models=config.LLM_FALLBACK_MODELS
    )
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        print(f"✍️  {start + 1}-{start + len(batch)} จาก {len(todo)} ...", flush=True)
        result = rewrite(generator, batch)
        if result is None:
            continue
        existing.update(dict(zip(batch, result)))

    with open(PARAPHRASE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    missing = [q for q in questions if q not in existing]
    print(f"\n💾 {PARAPHRASE_FILE} — {len(existing)} คำถาม")
    if missing:
        print(f"⚠️  ยังขาด {len(missing)} ข้อ (โควตาหมดหรือคำตอบใช้ไม่ได้) — รันซ้ำได้ ไม่เขียนทับของเดิม")
    print("👀 ควรอ่านไฟล์ตรวจสักรอบว่าคำถามใหม่ยังถามเรื่องเดิมจริง")


if __name__ == "__main__":
    main()
