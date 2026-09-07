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
quota is the binding constraint: 20 requests per model per day across five
models is about 100, and 493 questions need 986.  ``--limit`` samples
proportionally across categories rather than taking the first N, which here
would be 60 CLO questions and nothing else — the one category the system
already answers well, so the number would come back flattering and mean
nothing.

So the run is **resumable**.  Every answer and every verdict is written to
``outputs/eval_cache.json`` as it arrives, and a later run reuses whatever is
already there.  Without that the script was all-or-nothing: quota ran out
partway, the process kept going but recorded errors, and the next attempt
started from zero — which is why the first real run produced six graded
questions out of 205.  Now each day's quota adds to the pile.

Run it again after the quota resets and it picks up where it stopped:

    python evaluation/eval_generation.py            # ทำต่อจากที่ค้างไว้
    python evaluation/eval_generation.py --fresh    # ทิ้งแคช เริ่มใหม่

The cache is keyed by the question, the corpus fingerprint and the model that
produced the text, so re-chunking or switching models invalidates the
affected entries by itself rather than silently mixing two systems' results.
By the question and not by query_id: that is a position in the golden set,
and when the set grew from 205 to 493 the numbering shifted under seven
already-graded rows.

The judge is an LLM grading an LLM.  With both sides on Gemini that is a real
weakness — shared blind spots go unmeasured, and the first 30-question run
came back 100% on faithfulness, relevance and citations, which says more
about the grader than the system.  ``--judge ollama:<model>`` moves the
grading to a model running locally: a different family, and no quota.

    python evaluation/eval_generation.py --judge ollama:qwen3:8b

Measured 2026-09-07, that is not usable on this machine: qwen3:8b needs an
8192-token window for a grading prompt, which puts the KV cache past the
8GB card and spills layers to the CPU, and one grade then takes over 900
seconds — 51 hours for the set.  A hosted judge from another vendor buys the
same independence without the wait; whichever is used has to pass the
calibration in the notes before its numbers are quoted.

Run: python evaluation/eval_generation.py  ->  outputs/eval_generation.json
"""

from __future__ import annotations

import argparse
import hashlib
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

CACHE_FILE = config.OUTPUTS_DIR / "eval_cache.json"


def cache_key(question: str, corpus_sha1: str) -> str:
    """Identify cached work by the question, never by its position.

    The key used to be ``query_id``, which is an index into the golden set —
    and the golden set grows.  Adding the 288 curriculum questions renumbered
    everything from 129 on, so seven cached rows would have been handed back
    as the answer to a question that did not produce them, silently, with no
    error anywhere.  That is the same failure the golden set itself had with
    positional chunk ids; this is the second place it was hiding.
    """
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{corpus_sha1}"


def load_cache(fresh: bool) -> dict:
    """Load the cache, converting anything still keyed by position.

    A slot that never got as far as being graded has no question recorded, so
    there is nothing to re-key it by and it is dropped — one wasted answer
    costs a single call, and guessing costs the run's integrity.
    """
    if fresh or not CACHE_FILE.exists():
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    migrated: dict = {}
    dropped = 0
    for key, slot in raw.items():
        head, _, corpus_sha1 = key.partition(":")
        if not head.isdigit():
            migrated[key] = slot
            continue
        question = ((slot.get("row") or {}).get("question") or "").strip()
        if not question:
            dropped += 1
            continue
        migrated[cache_key(question, corpus_sha1)] = slot

    moved = sum(1 for k in raw if k.partition(":")[0].isdigit()) - dropped
    if moved or dropped:
        print(f"↻ ย้ายแคช {moved} รายการมาผูกกับตัวคำถามแทนเลขลำดับ"
              + (f" · ทิ้ง {dropped} รายการที่ไม่รู้ว่าเป็นคำถามข้อไหน" if dropped else ""))
        save_cache(migrated)
    return migrated


def save_cache(cache: dict) -> None:
    """Write after every question — a run that dies must keep its progress."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

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
    parser.add_argument("--fresh", action="store_true",
                        help="ทิ้งแคชแล้วเริ่มใหม่ (ค่าเริ่มต้นคือทำต่อจากที่ค้างไว้)")
    parser.add_argument("--judge", default=None,
                        help=f"โมเดลที่ใช้ตรวจ — ชื่อโมเดล Gemini หรือ ollama:<model> "
                             f"เพื่อใช้โมเดลในเครื่อง (ค่าเริ่มต้น {config.LLM_MODEL})")
    cli.add_model_arg(parser)
    args = parser.parse_args()
    spec = cli.apply(args)
    judge_model = args.judge or config.LLM_MODEL

    pipeline = RAGPipeline.from_config(use_memory=False)
    # Its own Generator, so the judge's cooldowns are tracked separately from
    # the answering side's — one running out must not rest the other.
    if judge_model.startswith("ollama:"):
        # No quota and a different model family from the one being graded —
        # the two reasons the judge is worth moving off Gemini.
        from src.local_llm import OllamaGenerator

        judge = OllamaGenerator(judge_model.split(":", 1)[1])
        try:
            judge.check()
        except Exception as exc:
            print(f"❌ {exc}")
            sys.exit(1)
    else:
        # Its own Generator, so the judge's cooldowns are tracked separately
        # from the answering side's — one running out must not rest the other.
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

    corpus = golden_set.fingerprint(chunks)
    cache = load_cache(args.fresh)
    done = sum(
        1
        for e in entries
        if (cache.get(cache_key(e["question"], corpus["chunks_sha1"])) or {}).get(
            "judge_model"
        )
        == judge_model
    )

    print(f"🧬 ตอบด้วย {spec.key} + {config.LLM_MODEL} · ตรวจด้วย {judge_model}")
    print(f"❓ {len(entries)} คำถาม · ตรวจไปแล้ว {done} · เหลือ {len(entries) - done} "
          f"(~{2 * (len(entries) - done)} เรียก API)")
    print(f"💾 แคช: {CACHE_FILE.name}\n")

    rows: list[dict] = []
    with RunLogger(
        f"eval-generation-{spec.key}",
        model=spec.key,
        judge=judge_model,
        n_queries=len(entries),
    ) as run:
        called = reused = 0
        for number, entry in enumerate(entries, start=1):
            key = cache_key(entry["question"], corpus["chunks_sha1"])
            slot = cache.get(key) or {}
            head = f"  [{number}/{len(entries)}] {entry['question'][:52]}"

            # 1. The answer.  Belongs to the answering model, not the judge —
            #    changing --judge must not throw away work that cost quota.
            result = slot.get("result") if slot.get("answer_model") == config.LLM_MODEL else None
            if result is None:
                try:
                    result = pipeline.answer(entry["question"])
                except Exception as exc:
                    run.problem("answer-failed", f"q{entry['query_id']}: {exc}")
                    print(f"{head}  ❌ ตอบไม่ได้ (จะลองใหม่รอบหน้า)", flush=True)
                    continue
                called += 1
                slot = {
                    "result": {"answer": result["answer"], "sources": result["sources"]},
                    "answer_model": config.LLM_MODEL,
                    # Recorded here, not only inside the graded row, so a slot
                    # that dies before grading can still be identified later.
                    "question": entry["question"],
                }
                cache[key] = slot
                save_cache(cache)
                result = slot["result"]

            # 2. The verdict.  Re-judged when the judge changes, since a score
            #    from a different grader is not comparable.
            if slot.get("judge_model") == judge_model and "row" in slot:
                rows.append(slot["row"])
                reused += 1
                print(f"{head}  ↩ ใช้ของเดิม", flush=True)
                continue

            row = grade(entry, result, judge)
            called += 1
            if "_error" in (row.get("judge") or {}):
                run.problem("judge-failed", f"q{entry['query_id']}: {row['judge']['_error']}")
                print(f"{head}  ⚠️  ตรวจไม่ได้ (จะลองใหม่รอบหน้า)", flush=True)
                # Not cached: an ungraded row must not look done next time.
                continue

            slot["row"] = row
            slot["judge_model"] = judge_model
            cache[key] = slot
            save_cache(cache)
            rows.append(row)
            print(f"{head}  ✓", flush=True)
            # The free tier counts per minute as well as per day.  A local
            # judge has no such limit, and 205 one-second sleeps is 3 minutes
            # of doing nothing.
            if not judge_model.startswith("ollama:"):
                time.sleep(1.0)

        print(f"\n📞 เรียก API {called} ครั้ง · ใช้ผลเดิม {reused} ข้อ")

    summary = summarise(rows)
    show(summary)

    payload = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": spec.key,
        "answer_model": config.LLM_MODEL,
        "judge_model": judge_model,
        "corpus": corpus,
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
