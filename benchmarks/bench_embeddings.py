"""Benchmark one embedding model end to end, then compare them all.

    python benchmarks/bench_embeddings.py --model e5-base     # one model
    python benchmarks/bench_embeddings.py --report            # the comparison

One model per process, on purpose.  A 4B model that runs the machine out of
memory takes its own process down and nothing else; the results already
written stay on disk, and the next model starts clean.  It is also what makes
"try them one at a time" literally true — each invocation is a complete,
recorded experiment.

The run is not a simulation of the pipeline, it *is* the pipeline: the
vectors and the FAISS index it produces are written to the same paths
``pipeline/create_embeddings.py`` and ``create_vector_db.py`` write, so a
benchmarked model is immediately usable with ``main.py --model <key>``.

Measured per model:

    load        seconds to bring the weights up (cold cache = download too)
    encode      seconds for the whole corpus, and chunks/second
    index       seconds to build FAISS from the vectors
    query       per-question latency over the golden set: mean, p50, p95
    quality     Recall@1/5, MRR and nDCG@10 on the golden set, dense-only,
                split by whether the question names the course or its code

Quality is measured dense-only, with no BM25 and no code pinning, because
this is the one place the embedding model is the only variable.  The hybrid
numbers belong to bench_retrievers.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config  # noqa: E402
from evaluation import metrics  # noqa: E402
from src import cli, journal  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402
from src.index_meta import build_meta, write_meta  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def peak_rss_gb() -> float | None:
    """Resident memory of this process, or None if psutil is not installed."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**3, 2)
    except Exception:
        return None


def load_inputs() -> tuple[list[dict], list[dict]]:
    """The chunks to encode and the questions to score against."""
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_FILE} — รัน run_pipeline.py --steps 1-2 ก่อน"
        )
    golden_path = config.DATA_DIR / "golden_set.json"
    if not golden_path.exists():
        raise FileNotFoundError(
            f"{golden_path} — รัน evaluation/build_golden_set.py ก่อน"
        )
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    return chunks, golden


def evaluate(
    store: VectorStore, model: EmbeddingModel, golden: list[dict]
) -> tuple[dict, list[float]]:
    """Dense-only retrieval over every golden question.

    Queries are encoded one at a time rather than as a batch, because that is
    what actually happens when somebody asks a question — a batched figure
    would flatter every model equally and describe none of them.
    """
    k_values = config.EVAL_K_VALUES
    per_query: list[dict[str, float]] = []
    by_phrasing: dict[str, list[dict[str, float]]] = {}
    latencies: list[float] = []

    for entry in golden:
        started = time.perf_counter()
        vector = model.encode_query(entry["question"])
        hits = store.search(vector, max(k_values))
        latencies.append(time.perf_counter() - started)

        retrieved = [str(hit["chunk_id"]) for hit in hits]
        relevant = set(entry["relevant_chunk_ids"])
        scores = metrics.score_one(retrieved, relevant, k_values)
        per_query.append(scores)
        by_phrasing.setdefault(entry.get("phrasing", "all"), []).append(scores)

    result = {"all": metrics.aggregate(per_query)}
    for phrasing, rows in by_phrasing.items():
        result[phrasing] = metrics.aggregate(rows)
    return result, latencies


def run_one(spec, chunks: list[dict], golden: list[dict]) -> dict:
    """Encode, index, evaluate — and leave the index behind for real use."""
    texts = [chunk["text"] for chunk in chunks]
    rss_before = peak_rss_gb()

    model = EmbeddingModel(
        spec.hf_id,
        normalize=config.NORMALIZE_EMBEDDINGS,
        device=config.EMBEDDING_DEVICE,
        spec=spec,
    )

    print(f"\n⏳ โหลด {spec.key} ...")
    model.load()

    print(f"🧮 เข้ารหัส {len(texts)} chunks ...")
    started = time.perf_counter()
    vectors = model.encode(
        texts, batch_size=config.EMBEDDING_BATCH_SIZE, show_progress=True
    )
    encode_seconds = time.perf_counter() - started
    rss_after = peak_rss_gb()

    started = time.perf_counter()
    store = VectorStore(int(vectors.shape[1]))
    store.build(vectors, chunks)
    index_seconds = time.perf_counter() - started

    print("🔎 ประเมินด้วย golden set ...")
    quality, latencies = evaluate(store, model, golden)
    latency_ms = np.array(latencies) * 1000

    # Persist to the real pipeline paths, so benchmarking a model is also
    # how it becomes available to main.py — no second encode pass.
    config.ensure_dirs()
    config.EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, vectors)
    store.save(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    write_meta(
        build_meta(
            config.SOURCE_FILES,
            len(chunks),
            embedding_model=spec.hf_id,
            embedding_key=spec.key,
            embedding_dim=int(vectors.shape[1]),
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        ),
        config.INDEX_META_FILE,
    )

    result = {
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "params_m": spec.params_m,
        "dim": int(vectors.shape[1]),
        "max_seq_length": spec.max_seq_length,
        "device": model.device,
        "batch_size": config.EMBEDDING_BATCH_SIZE,
        "n_chunks": len(texts),
        "n_queries": len(golden),
        "load_seconds": round(model.load_seconds or 0.0, 2),
        "encode_seconds": round(encode_seconds, 2),
        "chunks_per_second": round(len(texts) / encode_seconds, 2),
        "index_seconds": round(index_seconds, 3),
        "query_ms_mean": round(float(latency_ms.mean()), 2),
        "query_ms_p50": round(float(np.percentile(latency_ms, 50)), 2),
        "query_ms_p95": round(float(np.percentile(latency_ms, 95)), 2),
        "embeddings_mb": round(config.EMBEDDINGS_FILE.stat().st_size / 1024**2, 1),
        "rss_before_gb": rss_before,
        "rss_after_gb": rss_after,
        "quality": quality,
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }
    model.unload()
    return result


def show(result: dict) -> None:
    q = result["quality"]
    print(f"\n{'='*62}")
    print(f"📊 {result['model_key']}  ({result['params_m']}M, มิติ {result['dim']}, {result['device']})")
    print(f"{'='*62}")
    print(f"  โหลดโมเดล        {result['load_seconds']:>9.1f} s")
    print(f"  เข้ารหัสคลัง      {result['encode_seconds']:>9.1f} s   "
          f"({result['chunks_per_second']} chunks/s)")
    print(f"  สร้าง FAISS       {result['index_seconds']:>9.3f} s")
    print(f"  ต่อคำถาม          {result['query_ms_mean']:>9.1f} ms  "
          f"(p50 {result['query_ms_p50']}, p95 {result['query_ms_p95']})")
    print(f"  {'':16s} {'Recall@1':>9s} {'Recall@5':>9s} {'MRR':>7s} {'nDCG@10':>8s}")
    for name in ("all", "by_name", "by_code"):
        if name in q:
            row = q[name]
            print(f"  {name:16s} {row['recall@1']:>8.1%} {row['recall@5']:>9.1%} "
                  f"{row['mrr']:>7.3f} {row['ndcg@10']:>8.3f}")


def report() -> None:
    """Compare every result written so far."""
    files = sorted(RESULTS_DIR.glob("embed-*.json"))
    if not files:
        print("❌ ยังไม่มีผลใน benchmarks/results/ — รันทีละโมเดลก่อน")
        sys.exit(1)

    results = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    results.sort(key=lambda r: r["quality"]["all"]["recall@1"], reverse=True)

    lines = [
        "| โมเดล | params | มิติ | อุปกรณ์ | โหลด | เข้ารหัส 3,128 chunks | chunks/s | ต่อคำถาม (p50) | Recall@1 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        q = r["quality"]["all"]
        lines.append(
            f"| {r['model_key']} | {r['params_m']}M | {r['dim']} | {r['device']} | "
            f"{r['load_seconds']:.0f}s | {r['encode_seconds']:.0f}s | "
            f"{r['chunks_per_second']:.1f} | {r['query_ms_p50']:.1f} ms | "
            f"{q['recall@1']:.1%} | {q['recall@5']:.1%} | {q['mrr']:.3f} |"
        )

    split = [
        "",
        "แยกตามสำนวนคำถาม (Recall@1)",
        "",
        "| โมเดล | ถามด้วยชื่อวิชา | ถามด้วยรหัสวิชา |",
        "|---|---|---|",
    ]
    for r in results:
        q = r["quality"]
        name = q.get("by_name", {}).get("recall@1", 0)
        code = q.get("by_code", {}).get("recall@1", 0)
        split.append(f"| {r['model_key']} | {name:.1%} | {code:.1%} |")

    text = "\n".join(lines + split)
    print(text)

    out = RESULTS_DIR / "embedding-comparison.md"
    out.write_text(
        f"# เปรียบเทียบ embedding model\n\n"
        f"วัดเมื่อ {datetime.now():%Y-%m-%d %H:%M} · "
        f"คลัง {results[0]['n_chunks']} chunks · "
        f"golden set {results[0]['n_queries']} คำถาม · dense-only\n\n" + text + "\n",
        encoding="utf-8",
    )
    print(f"\n💾 {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models.")
    parser.add_argument(
        "--report", action="store_true", help="compare every result written so far"
    )
    cli.add_model_arg(parser)
    args = parser.parse_args()

    if args.report:
        report()
        return

    spec = cli.apply(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chunks, golden = load_inputs()

    with RunLogger(
        f"bench-embed-{spec.key}",
        model=spec.key,
        hf_id=spec.hf_id,
        device=config.EMBEDDING_DEVICE or "auto",
        n_chunks=len(chunks),
        n_queries=len(golden),
    ) as run:
        try:
            result = run_one(spec, chunks, golden)
        except Exception as exc:
            # A model that cannot be downloaded, cannot be loaded, or runs the
            # machine out of memory is a result too — the whole point of the
            # exercise is knowing which ones this machine can actually run.
            run.problem(
                "model-failed", f"{spec.key}: {type(exc).__name__}: {exc}", model=spec.key
            )
            journal.append(
                f"benchmark embedding: {spec.key}",
                ok=False,
                what=f"วัดความเร็วและคุณภาพของ {spec.hf_id}",
                how=f"device={config.EMBEDDING_DEVICE or 'auto'}, "
                f"max_seq={spec.max_seq_length}, chunks={len(chunks)}",
                result=f"ล้มเหลว: {type(exc).__name__}: {str(exc)[:300]}",
                model=spec.key,
                error_type=type(exc).__name__,
            )
            raise

        show(result)
        path = RESULTS_DIR / f"embed-{spec.key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 {path}")
        run.event("benchmark-done", **{
            k: v for k, v in result.items() if k != "quality"
        })

    q = result["quality"]["all"]
    journal.append(
        f"benchmark embedding: {spec.key}",
        what=f"วัดความเร็วและคุณภาพของ {spec.hf_id} ({spec.params_m}M, มิติ {result['dim']})",
        how=(
            f"เข้ารหัส {result['n_chunks']} chunks บน {result['device']}, "
            f"batch={result['batch_size']}, max_seq={spec.max_seq_length}; "
            f"ประเมิน dense-only ด้วย golden set {result['n_queries']} คำถาม"
        ),
        result=(
            f"โหลด {result['load_seconds']}s, เข้ารหัส {result['encode_seconds']}s "
            f"({result['chunks_per_second']} chunks/s), ต่อคำถาม p50 "
            f"{result['query_ms_p50']}ms, Recall@1 {q['recall@1']:.1%}, MRR {q['mrr']:.3f}"
        ),
        **{k: v for k, v in result.items() if k != "quality"},
        quality=result["quality"],
    )


if __name__ == "__main__":
    main()
