"""Evaluate answer quality — what the model did with the chunks it was given.

Retrieval metrics prove the right chunk was found.  They say nothing about
whether the answer that came back is supported by it, whether the ``[1]``
markers point anywhere real, or whether a question the documents do not cover
gets an honest "I don't know" instead of an invented number.  That is the gap
this fills.

    python evaluation/eval_generation.py                  # every question
    python evaluation/eval_generation.py --limit 30       # a stratified sample
    python evaluation/eval_generation.py --judge gemini-3.5-flash

Measured per answerable question:

    context_recall  was a gold chunk actually in the context the model saw?
                    Computed here, not judged — it separates "retrieval missed
                    it" from "the model had it and still got it wrong", which
                    are different bugs with different fixes.
    faithfulness    is every claim supported by the context?  (judged, 0-1)
    relevance       does the answer address the question?     (judged, 0-1)
    citation_valid  does every [n] point at a chunk that was supplied, and
                    does that chunk support the sentence?     (judged, 0-1)

Measured per unanswerable question:

    abstained       did the answer admit the documents do not say, rather than
                    produce a figure?  These come from the "ข้อมูลที่เอกสารนี้
                    ไม่ได้ระบุ" section each registrar document carries, so the
                    documents themselves define the right behaviour.

Two API calls per question — one to answer, one to judge — so the free-tier
quota is the binding constraint.  ``--limit`` samples proportionally across
categories rather than taking the first N, which would be all CLO.

The judge is an LLM grading an LLM, and by default both are Gemini.  That is
a real weakness: shared blind spots go unmeasured.  ``--judge`` exists so the
grading can be moved to another family once a second API key is available.

Run: python evaluation/eval_generation.py  ->  outputs/eval_generation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from project root

import config  # noqa: E402
from evaluation import golden_set  # noqa: E402
from src import cli, journal  # noqa: E402
from src.generator import Generator  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_CITATION = re.compile(r"\[(\d+)\]")

_JUDGE_ANSWERABLE = """คุณคือผู้ตรวจคุณภาพคำตอบของระบบถาม-ตอบจากเอกสาร

ให้คะแนน 0 ถึง 1 (ทศนิยม 1 ตำแหน่ง) สามด้าน โดยดูจากบริบทที่ให้มาเท่านั้น
ห้ามใช้ความรู้ภายนอก

- faithfulness: ทุกข้อความในคำตอบมีบริบทรองรับหรือไม่ ถ้ามีข้อความที่บริบท
  ไม่ได้บอกไว้เลยแม้แต่ประโยคเดียว ให้ต่ำกว่า 0.5
- relevance: คำตอบตอบตรงคำถามหรือไม่
- citation_valid: เลขอ้างอิง [n] ทุกตัวชี้ไปยังบริบทที่รองรับข้อความนั้นจริงหรือไม่
  ถ้าคำตอบไม่มีเลขอ้างอิงเลยทั้งที่ใช้เนื้อหาจากบริบท ให้ 0

บริบทที่ระบบได้รับ:
{context}

คำถาม: {question}

คำตอบของระบบ:
{answer}

ตอบกลับเป็น JSON อย่างเดียว ไม่ต้องอธิบายนอก JSON และ reason ต้องไม่เกิน 20 คำ:
{{"faithfulness": 0.0, "relevance": 0.0, "citation_valid": 0.0, "reason": "<สั้น ๆ>"}}"""

_JUDGE_UNANSWERABLE = """คุณคือผู้ตรวจว่าระบบถาม-ตอบยอมรับหรือไม่ว่าไม่มีข้อมูล

เอกสารต้นทางระบุไว้เองว่าหัวข้อนี้ "ไม่ได้ระบุไว้ในเอกสาร" คำตอบที่ถูกต้องคือ
บอกว่าไม่มีข้อมูล หรือแนะนำให้ติดต่อหน่วยงาน ไม่ใช่การให้ตัวเลขหรือรายละเอียด
ที่เอกสารไม่ได้บอก

หัวข้อที่ถาม: {question}

คำตอบของระบบ:
{answer}

ตอบกลับเป็น JSON อย่างเดียว:
{{"abstained": true/false, "invented": "<ข้อความที่แต่งขึ้น ถ้าไม่มีให้เว้นว่าง>"}}"""


def stratified(entries: list[dict], limit: int) -> list[dict]:
    """Take *limit* entries spread across categories, not the first N."""
    if limit <= 0 or limit >= len(entries):
        return entries
    buckets: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        buckets[entry.get("category", "?")].append(entry)

    picked: list[dict] = []
    for category, rows in buckets.items():
        share = max(1, round(limit * len(rows) / len(entries)))
        # Evenly spaced rather than the first few, so one document does not
        # supply the whole sample for its category.
        step = max(1, len(rows) // share)
        picked.extend(rows[::step][:share])
    return sorted(picked, key=lambda e: e["query_id"])[:limit]


def ask_judge(judge: Generator, prompt: str) -> dict:
    """Send one grading prompt and parse the JSON out of the reply.

    Goes through a Generator rather than the raw client so the judge gets the
    same per-model quota fallback the answering side has.  Without it the
    judge dies on the free tier's 20-requests-per-day-per-model long before
    the run finishes, which is exactly what happened the first time.
    """
    try:
        # 1000, not the 500 default: at 500 a verdict was lost to a "reason"
        # field that ran past the budget mid-JSON, which reads as a grading
        # failure when the grade itself was fine.
        text = judge.complete(prompt, max_tokens=1000)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

    # Models wrap JSON in ```json fences often enough to be worth handling.
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"_error": f"ไม่ใช่ JSON: {text[:120]}"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"_error": f"JSON เสีย: {match.group(0)[:120]}"}


def grade(entry: dict, result: dict, judge: Generator) -> dict:
    """Score one answered question."""
    answer = result["answer"]
    chunks = result["sources"]
    retrieved = {str(c.get("chunk_id")) for c in chunks}
    gold = set(entry["relevant_chunk_ids"])

    row = {
        "query_id": entry["query_id"],
        "category": entry["category"],
        "question": entry["question"],
        "source": entry.get("source", ""),
        "answer": answer,
        "n_chunks": len(chunks),
        # Cheap and exact, so it is computed rather than judged.
        "context_recall": bool(gold & retrieved) if gold else None,
        "citations_in_range": all(
            1 <= int(n) <= len(chunks) for n in _CITATION.findall(answer)
        ),
        "n_citations": len(set(_CITATION.findall(answer))),
    }

    if entry["category"] == "unanswerable":
        verdict = ask_judge(
            judge,
            _JUDGE_UNANSWERABLE.format(question=entry["question"], answer=answer),
        )
        row["judge"] = verdict
        row["abstained"] = bool(verdict.get("abstained")) if "_error" not in verdict else None
        return row

    context = "\n\n".join(
        f"[{i}] {c.get('source', '')}\n{c.get('text', '')}"
        for i, c in enumerate(chunks, start=1)
    )
    verdict = ask_judge(
        judge,
        _JUDGE_ANSWERABLE.format(
            context=context, question=entry["question"], answer=answer
        ),
    )
    row["judge"] = verdict
    for field in ("faithfulness", "relevance", "citation_valid"):
        value = verdict.get(field)
        row[field] = float(value) if isinstance(value, (int, float)) else None
    return row


def summarise(rows: list[dict]) -> dict:
    """Mean of each score, per category and overall, ignoring failed grades."""

    def mean(values: list) -> float | None:
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 3) if clean else None

    def block(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "graded": sum(1 for r in subset if "_error" not in (r.get("judge") or {})),
            "context_recall": mean([r.get("context_recall") for r in subset]),
            "faithfulness": mean([r.get("faithfulness") for r in subset]),
            "relevance": mean([r.get("relevance") for r in subset]),
            "citation_valid": mean([r.get("citation_valid") for r in subset]),
            "citations_in_range": mean([r.get("citations_in_range") for r in subset]),
            "abstained": mean([r.get("abstained") for r in subset]),
        }

    out = {"all": block(rows)}
    for category in sorted({r["category"] for r in rows}):
        out[category] = block([r for r in rows if r["category"] == category])
    return out


def show(summary: dict) -> None:
    print(f"\n{'='*74}")
    print("📊 คุณภาพคำตอบ")
    print(f"{'='*74}")
    print(f"  {'หมวด':14s} {'n':>4s} {'ตรวจได้':>8s} {'context':>9s} "
          f"{'faithful':>9s} {'relevant':>9s} {'citation':>9s} {'ยอมไม่รู้':>10s}")

    def cell(value, pct=True):
        if value is None:
            return f"{'—':>9s}"
        return f"{value:>9.1%}" if pct else f"{value:>9.3f}"

    for name, row in summary.items():
        print(f"  {name:14s} {row['n']:>4d} {row['graded']:>8d} "
              f"{cell(row['context_recall'])} {cell(row['faithfulness'])} "
              f"{cell(row['relevance'])} {cell(row['citation_valid'])} "
              f"{cell(row['abstained'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer quality.")
    parser.add_argument("--limit", type=int, default=0,
                        help="ประเมินกี่คำถาม (0 = ทั้งหมด), สุ่มกระจายทุกหมวด")
    parser.add_argument("--judge", default=None,
                        help=f"โมเดลที่ใช้ตรวจ (ค่าเริ่มต้น {config.LLM_MODEL})")
    cli.add_model_arg(parser)
    args = parser.parse_args()
    spec = cli.apply(args)
    judge_model = args.judge or config.LLM_MODEL

    pipeline = RAGPipeline.from_config(use_memory=False)
    # Its own Generator, so the judge's cooldowns are tracked separately from
    # the answering side's — one running out must not rest the other.
    judge = Generator(
        model=judge_model,
        fallback_models=tuple(
            m for m in (config.LLM_MODEL, *config.LLM_FALLBACK_MODELS)
            if m != judge_model
        ),
    )
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    entries, _ = golden_set.load(chunks, include_unanswerable=True)
    entries = stratified(entries, args.limit)

    print(f"🧬 ตอบด้วย {spec.key} + {config.LLM_MODEL} · ตรวจด้วย {judge_model}")
    print(f"❓ {len(entries)} คำถาม ({2 * len(entries)} เรียก API)\n")

    rows: list[dict] = []
    with RunLogger(
        f"eval-generation-{spec.key}",
        model=spec.key,
        judge=judge_model,
        n_queries=len(entries),
    ) as run:
        for number, entry in enumerate(entries, start=1):
            print(f"  [{number}/{len(entries)}] {entry['question'][:56]}", flush=True)
            try:
                result = pipeline.answer(entry["question"])
            except Exception as exc:
                run.problem("answer-failed", f"q{entry['query_id']}: {exc}")
                rows.append({
                    "query_id": entry["query_id"],
                    "category": entry["category"],
                    "question": entry["question"],
                    "judge": {"_error": f"answer failed: {type(exc).__name__}"},
                })
                continue
            row = grade(entry, result, judge)
            if "_error" in (row.get("judge") or {}):
                run.problem("judge-failed", f"q{entry['query_id']}: {row['judge']['_error']}")
            rows.append(row)
            # The free tier counts per minute as well as per day.
            time.sleep(1.0)

    summary = summarise(rows)
    show(summary)

    payload = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": spec.key,
        "answer_model": config.LLM_MODEL,
        "judge_model": judge_model,
        "corpus": golden_set.fingerprint(chunks),
        "summary": summary,
        "rows": rows,
    }
    config.EVAL_GENERATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_GENERATION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {config.EVAL_GENERATION_FILE}")

    overall = summary["all"]
    journal.append(
        f"eval generation: {spec.key} + {config.LLM_MODEL}",
        ok=overall["graded"] > 0,
        what="วัดคุณภาพคำตอบ ไม่ใช่แค่ว่าค้น chunk เจอ",
        how=(
            f"{len(entries)} คำถามจาก golden set, ตอบด้วย {config.LLM_MODEL}, "
            f"ตรวจด้วย {judge_model} (LLM judge)"
        ),
        result=(
            f"faithfulness {overall['faithfulness']}, relevance {overall['relevance']}, "
            f"citation {overall['citation_valid']}, "
            f"ยอมรับว่าไม่รู้ {summary.get('unanswerable', {}).get('abstained')}"
        ),
        summary=summary,
        judge_model=judge_model,
        corpus=payload["corpus"],
    )


if __name__ == "__main__":
    main()
