"""How fast does each generation model answer?

    python benchmarks/bench_llm_apis.py                 # every Gemini model
    python benchmarks/bench_llm_apis.py --repeat 5      # more samples per model
    python benchmarks/bench_llm_apis.py --models gemini-3.5-flash,gemini-3.1-flash-lite

The questions are real ones, retrieved against the real index, so each model
sees a prompt the size the system actually sends — a few thousand tokens of
Thai context, not a toy string.  Anything else measures the network.

Three numbers per call, because they answer different questions:

    TTFT     time to the first token.  What a user in a chat UI perceives as
             "did it hear me", and the only one a streaming interface can
             hide latency behind.
    total    time until the answer is complete.
    tok/s    output tokens per second after the first — the throughput that
             decides how long a long answer takes.

**On comparing providers:** only GEMINI_API_KEY is configured on this
machine, so what follows compares the five Gemini models the fallback chain
uses, which is the comparison that changes anything here — free-tier quota is
counted per model, and the system already switches between them.  A second
provider needs its key in .env and a client call here; nothing else in the
file assumes Gemini.  Prices for the other providers are in the README, from
their published pricing pages.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import cli, journal, prompt_templates  # noqa: E402
from src.generator import load_api_key  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = Path(__file__).resolve().parent / "results"

QUESTIONS = [
    "วิชา 04-620-201 มี CLO อะไรบ้าง",
    "หลักสูตรวิศวกรรมคอมพิวเตอร์มีกี่หน่วยกิต",
    "ขั้นตอนการลาพักการศึกษาต้องทำอย่างไร",
]


def build_prompts() -> list[tuple[str, str]]:
    """(question, prompt) pairs built from real retrieval, or plain questions.

    Falling back to the bare question when no index exists keeps the API
    timing usable on a machine that has not built one — the numbers are then
    for a short prompt, which the report says.
    """
    if not config.FAISS_INDEX_FILE.exists():
        print("⚠️  ไม่พบ index — วัดด้วยคำถามเปล่า (prompt สั้นกว่าการใช้งานจริงมาก)")
        return [(q, q) for q in QUESTIONS]

    from src.embedding_model import EmbeddingModel
    from src.hybrid_retriever import BM25Index, HybridRetriever
    from src.vector_store import VectorStore

    store = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    embedder = EmbeddingModel(
        config.EMBEDDING_MODEL,
        normalize=config.NORMALIZE_EMBEDDINGS,
        device=config.EMBEDDING_DEVICE,
        spec=config.EMBEDDING_SPEC,
    )
    bm25 = (
        BM25Index.load(config.BM25_INDEX_FILE)
        if config.BM25_INDEX_FILE.exists()
        else None
    )
    if bm25 is None or len(bm25) != len(store.chunks):
        bm25 = BM25Index()
        bm25.build(store.chunks)
        bm25.save(config.BM25_INDEX_FILE)

    retriever = HybridRetriever(store, embedder, bm25)
    pairs = []
    for question in QUESTIONS:
        chunks = retriever.retrieve(
            question, k=config.TOP_K, candidate_k=config.CANDIDATE_K, rrf_k=config.RRF_K
        )
        pairs.append((question, prompt_templates.build_prompt(question, chunks)))
    return pairs


def call_gemini(client, model: str, prompt: str) -> dict:
    """One streamed call, timed.  Raises nothing — failures are results too."""
    from google.genai import types

    settings = types.GenerateContentConfig(
        system_instruction=prompt_templates.SYSTEM_PROMPT,
        max_output_tokens=config.LLM_MAX_TOKENS,
    )

    started = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    usage = None
    try:
        stream = client.models.generate_content_stream(
            model=model, contents=prompt, config=settings
        )
        for piece in stream:
            if ttft is None:
                # First chunk with actual text, not the first empty frame —
                # an empty keepalive would report a TTFT the user never saw.
                if getattr(piece, "text", None):
                    ttft = time.perf_counter() - started
            if getattr(piece, "text", None):
                text_parts.append(piece.text)
            if getattr(piece, "usage_metadata", None):
                usage = piece.usage_metadata
        total = time.perf_counter() - started
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "total_s": round(time.perf_counter() - started, 3),
        }

    answer = "".join(text_parts)
    out_tokens = getattr(usage, "candidates_token_count", None)
    in_tokens = getattr(usage, "prompt_token_count", None)
    # Throughput measured after the first token: including TTFT would fold
    # queueing time into a number that is meant to describe generation speed.
    streaming_s = (total - ttft) if (ttft is not None and total > ttft) else None
    return {
        "ok": True,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 3),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_chars": len(answer),
        "tokens_per_s": (
            round(out_tokens / streaming_s, 1)
            if out_tokens and streaming_s and streaming_s > 0
            else None
        ),
        "preview": answer[:120].replace("\n", " "),
    }


def summarise(samples: list[dict]) -> dict:
    """Aggregate the successful calls; keep the failures visible."""
    ok = [s for s in samples if s.get("ok")]
    failed = [s for s in samples if not s.get("ok")]

    def stat(field: str):
        values = [s[field] for s in ok if s.get(field) is not None]
        if not values:
            return None
        return {
            "mean": round(statistics.mean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "calls": len(samples),
        "ok": len(ok),
        "failed": len(failed),
        "errors": sorted({s.get("error", "") for s in failed if s.get("error")}),
        "ttft_s": stat("ttft_s"),
        "total_s": stat("total_s"),
        "tokens_per_s": stat("tokens_per_s"),
        "input_tokens": stat("input_tokens"),
        "output_tokens": stat("output_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generation APIs.")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model ids (default: LLM_MODEL + LLM_FALLBACK_MODELS)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="calls per model per question"
    )
    cli.add_model_arg(parser)
    args = parser.parse_args()
    cli.apply(args)

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [config.LLM_MODEL, *config.LLM_FALLBACK_MODELS]
    )

    # Say plainly which providers could be measured and which could not,
    # rather than quietly reporting a one-provider table as a comparison.
    from dotenv import load_dotenv

    load_dotenv(config.ROOT_DIR / ".env")
    providers = {
        name: bool(os.environ.get(f"{name.upper()}_API_KEY"))
        for name in ("gemini", "openai", "anthropic")
    }
    print("🔑 คีย์ที่พบ: " + ", ".join(
        f"{name}={'มี' if found else 'ไม่มี'}" for name, found in providers.items()
    ))
    if not providers["gemini"]:
        print("❌ ไม่พบ GEMINI_API_KEY — วัดไม่ได้")
        sys.exit(1)

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=load_api_key(),
        http_options=types.HttpOptions(
            timeout=120_000,
            # No retries at all here: a retried call reports the latency of
            # the second attempt, which is not what happened.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    prompts = build_prompts()
    print(f"❓ {len(prompts)} คำถาม × {args.repeat} ครั้ง × {len(models)} โมเดล\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    with RunLogger("bench-llm-apis", models=models, repeat=args.repeat) as run:
        for model in models:
            print(f"── {model}")
            samples: list[dict] = []
            for question, prompt in prompts:
                for _ in range(args.repeat):
                    sample = call_gemini(client, model, prompt)
                    sample["question"] = question
                    samples.append(sample)
                    if sample["ok"]:
                        print(f"   ✓ TTFT {sample['ttft_s']}s  รวม {sample['total_s']}s  "
                              f"{sample['output_tokens']} tok  {sample['tokens_per_s']} tok/s")
                    else:
                        run.problem("llm-call-failed", f"{model}: {sample['error']}")
                        print(f"   ✗ {sample['error'][:100]}")
            results[model] = {"samples": samples, **summarise(samples)}
            summary = results[model]
            if summary["ok"]:
                print(f"   → เฉลี่ย TTFT {summary['ttft_s']['mean']}s, "
                      f"รวม {summary['total_s']['mean']}s\n")
            else:
                print(f"   → ล้มเหลวทั้ง {summary['calls']} ครั้ง\n")

    lines = [
        "| โมเดล | สำเร็จ | TTFT เฉลี่ย | TTFT ต่ำ-สูง | รวมเฉลี่ย | tok/s | input tok | output tok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, r in results.items():
        if not r["ok"]:
            lines.append(f"| {model} | 0/{r['calls']} | — | — | — | — | — | {r['errors'][:1]} |")
            continue
        lines.append(
            f"| {model} | {r['ok']}/{r['calls']} | {r['ttft_s']['mean']:.2f}s | "
            f"{r['ttft_s']['min']:.2f}-{r['ttft_s']['max']:.2f}s | "
            f"{r['total_s']['mean']:.2f}s | "
            f"{r['tokens_per_s']['mean'] if r['tokens_per_s'] else '—'} | "
            f"{r['input_tokens']['mean']:.0f} | {r['output_tokens']['mean']:.0f} |"
        )
    text = "\n".join(lines)
    print("\n" + text)

    payload = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "providers_with_keys": providers,
        "embedding_model": config.EMBEDDING_KEY,
        "repeat": args.repeat,
        "questions": [q for q, _ in prompts],
        "results": results,
    }
    with open(RESULTS_DIR / "llm-api-comparison.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    (RESULTS_DIR / "llm-api-comparison.md").write_text(
        f"# เปรียบเทียบความเร็ว API ของโมเดลภาษา\n\n"
        f"วัดเมื่อ {datetime.now():%Y-%m-%d %H:%M} · "
        f"{len(prompts)} คำถาม × {args.repeat} ครั้ง · prompt สร้างจาก retrieval จริง\n\n"
        f"คีย์ที่มี: {', '.join(k for k, v in providers.items() if v) or 'ไม่มี'}\n\n"
        + text
        + "\n",
        encoding="utf-8",
    )
    print(f"\n💾 {RESULTS_DIR / 'llm-api-comparison.md'}")

    working = {m: r for m, r in results.items() if r["ok"]}
    journal.append(
        "benchmark LLM API latency",
        ok=bool(working),
        what="วัดเวลาตอบกลับของโมเดลภาษาแต่ละตัว (TTFT, เวลารวม, tokens/วินาที)",
        how=(
            f"{len(prompts)} คำถามจริง × {args.repeat} ครั้ง/โมเดล, "
            f"prompt สร้างจาก hybrid retrieval บน index ของ {config.EMBEDDING_KEY}, "
            f"เรียกแบบ streaming ไม่มี retry"
        ),
        result=(
            ", ".join(
                f"{m}: TTFT {r['ttft_s']['mean']}s รวม {r['total_s']['mean']}s"
                for m, r in working.items()
            )
            or "ไม่มีโมเดลใดตอบสำเร็จ"
        ),
        providers_with_keys=providers,
        summary={m: {k: v for k, v in r.items() if k != "samples"} for m, r in results.items()},
    )


if __name__ == "__main__":
    main()
