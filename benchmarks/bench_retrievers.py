"""Is BM25 the right sparse retriever?  Measure the alternatives.

    python benchmarks/bench_retrievers.py                  # default model's index
    python benchmarks/bench_retrievers.py --model e5-base

Three questions, in order:

1. Among sparse retrievers alone, which finds the right chunk most often?
2. Fused with the dense retriever by RRF, which combination wins?  A sparse
   retriever that loses on its own can still fuse better, because what
   fusion rewards is finding chunks the dense side missed — not being right.
3. Does pinning exact course codes still earn its place once a
   character-level retriever is in the mix?  That pin exists because BM25
   splits ``04-620-201`` into three useless tokens; a char n-gram index
   never had that problem, so the pin may be solving a problem that the
   right retriever simply does not have.

Everything is scored on the same golden set, with the same dense index, so
the sparse method is the only thing that changes.
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
from evaluation import golden_set, metrics  # noqa: E402
from src import cli, journal, sparse_retrievers  # noqa: E402
from src.embedding_model import EmbeddingModel  # noqa: E402
from src.hybrid_retriever import exact_code_matches, reciprocal_rank_fusion  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = list(sparse_retrievers.BUILDERS)


def score_all(runner, golden: list[dict]) -> dict:
    """Run *runner* over every golden question and aggregate the metrics."""
    k_values = config.EVAL_K_VALUES
    per_query: list[dict[str, float]] = []
    by_phrasing: dict[str, list[dict[str, float]]] = {}
    latencies: list[float] = []

    for entry in golden:
        started = time.perf_counter()
        hits = runner(entry["question"])
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved = [str(hit["chunk_id"]) for hit in hits]
        scores = metrics.score_one(
            retrieved, set(entry["relevant_chunk_ids"]), k_values
        )
        per_query.append(scores)
        by_phrasing.setdefault(entry.get("phrasing", "all"), []).append(scores)

    latency = np.array(latencies)
    result = {
        "all": metrics.aggregate(per_query),
        "ms_mean": round(float(latency.mean()), 2),
        "ms_p50": round(float(np.percentile(latency, 50)), 2),
        "ms_p95": round(float(np.percentile(latency, 95)), 2),
    }
    for phrasing, rows in by_phrasing.items():
        result[phrasing] = metrics.aggregate(rows)
    return result


def table(rows: list[tuple[str, dict]], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| วิธี | Recall@1 | Recall@5 | MRR | nDCG@10 | ถามด้วยชื่อ | ถามด้วยรหัส | ต่อคำถาม (p50) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(
        rows, key=lambda item: item[1]["all"]["recall@1"], reverse=True
    ):
        a = r["all"]
        lines.append(
            f"| {name} | {a['recall@1']:.1%} | {a['recall@5']:.1%} | {a['mrr']:.3f} | "
            f"{a['ndcg@10']:.3f} | {r.get('by_name', {}).get('recall@1', 0):.1%} | "
            f"{r.get('by_code', {}).get('recall@1', 0):.1%} | {r['ms_p50']:.1f} ms |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sparse retrievers.")
    cli.add_model_arg(parser)
    args = parser.parse_args()
    spec = cli.apply(args)

    if not config.FAISS_INDEX_FILE.exists():
        print(f"❌ ไม่พบ index ของ {spec.key} — รัน benchmarks/bench_embeddings.py --model {spec.key} ก่อน")
        sys.exit(1)

    store = VectorStore.load(config.FAISS_INDEX_FILE, config.CHUNK_STORE_FILE)
    # Against the indexed chunks, which may lag outputs/chunks.json if the
    # index was built before the last re-chunk.
    try:
        golden, golden_report = golden_set.load(store.chunks)
    except golden_set.GoldenSetError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    embedder = EmbeddingModel(
        spec.hf_id,
        normalize=config.NORMALIZE_EMBEDDINGS,
        device=config.EMBEDDING_DEVICE,
        spec=spec,
    )
    embedder.load()
    print(f"🧬 dense: {spec.key} | 📚 {len(store)} chunks | ❓ {len(golden)} คำถาม\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "embedding_model": spec.key,
        "n_chunks": len(store),
        "n_queries": len(golden),
        "corpus": golden_report["corpus_now"],
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "build_seconds": {},
        "sparse_only": {},
        "hybrid": {},
    }

    with RunLogger(f"bench-retrievers-{spec.key}", model=spec.key) as run:
        # 1. Build every sparse index once, and time that too — a retriever
        #    that scores well but takes ten minutes to build is a different
        #    proposition from one that takes two seconds.
        indexes = {}
        for name in METHODS:
            print(f"🔧 สร้าง {name} ...", end=" ", flush=True)
            started = time.perf_counter()
            try:
                indexes[name] = sparse_retrievers.build(name, store.chunks)
            except Exception as exc:
                run.problem("sparse-build-failed", f"{name}: {type(exc).__name__}: {exc}")
                print(f"❌ {type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - started
            payload["build_seconds"][name] = round(elapsed, 2)
            print(f"{elapsed:.1f}s")

        # 2. Sparse alone.
        print("\n── ค้นด้วย sparse อย่างเดียว")
        for name, index in indexes.items():
            payload["sparse_only"][name] = score_all(
                lambda q, ix=index: ix.search(q, max(config.EVAL_K_VALUES)), golden
            )
            print(f"   {name:14s} Recall@1 {payload['sparse_only'][name]['all']['recall@1']:.1%}")

        # 3. Dense alone, and dense fused with each sparse method.
        print("\n── dense และ hybrid (RRF)")
        top = max(config.EVAL_K_VALUES)

        def dense_only(question: str) -> list[dict]:
            return store.search(embedder.encode_query(question), top)

        payload["dense_only"] = score_all(dense_only, golden)
        print(f"   {'dense':14s} Recall@1 {payload['dense_only']['all']['recall@1']:.1%}")

        def hybrid(question: str, index, pin: bool = False) -> list[dict]:
            dense_hits = store.search(
                embedder.encode_query(question), config.CANDIDATE_K
            )
            sparse_hits = index.search(question, config.CANDIDATE_K)
            fused = reciprocal_rank_fusion(
                [dense_hits, sparse_hits], k=config.CANDIDATE_K, rrf_k=config.RRF_K
            )
            if not pin:
                return fused[:top]
            pinned = exact_code_matches(question, store.chunks)
            seen, out = set(), []
            for chunk in [*pinned, *fused]:
                if chunk.get("chunk_id") in seen:
                    continue
                seen.add(chunk.get("chunk_id"))
                out.append(chunk)
                if len(out) >= top:
                    break
            return out

        for name, index in indexes.items():
            payload["hybrid"][name] = score_all(
                lambda q, ix=index: hybrid(q, ix), golden
            )
            print(f"   dense+{name:9s} Recall@1 {payload['hybrid'][name]['all']['recall@1']:.1%}")

        # 4. Does pinning still help, for each fusion?
        print("\n── hybrid + ปักหมุดรหัสวิชา")
        payload["hybrid_pinned"] = {}
        for name, index in indexes.items():
            payload["hybrid_pinned"][name] = score_all(
                lambda q, ix=index: hybrid(q, ix, pin=True), golden
            )
            print(f"   dense+{name:9s}+pin Recall@1 "
                  f"{payload['hybrid_pinned'][name]['all']['recall@1']:.1%}")

    text = "\n\n".join(
        [
            f"# เปรียบเทียบวิธีค้นคืนแบบ sparse\n",
            f"วัดเมื่อ {datetime.now():%Y-%m-%d %H:%M} · dense = {spec.key} · "
            f"{len(store)} chunks · golden set {len(golden)} คำถาม",
            "เวลาสร้าง index: "
            + ", ".join(f"{k} {v}s" for k, v in payload["build_seconds"].items()),
            table(list(payload["sparse_only"].items()), "1. sparse อย่างเดียว"),
            table(
                [("dense", payload["dense_only"])]
                + [(f"dense+{k}", v) for k, v in payload["hybrid"].items()],
                "2. dense และ hybrid (RRF)",
            ),
            table(
                [(f"dense+{k}+pin", v) for k, v in payload["hybrid_pinned"].items()],
                "3. hybrid + ปักหมุดรหัสวิชา",
            ),
        ]
    )
    out_md = RESULTS_DIR / "retriever-comparison.md"
    out_md.write_text(text + "\n", encoding="utf-8")
    out_json = RESULTS_DIR / "retriever-comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {out_md}\n💾 {out_json}")

    best_sparse = max(
        payload["sparse_only"].items(), key=lambda kv: kv[1]["all"]["recall@1"]
    )
    best_hybrid = max(
        payload["hybrid"].items(), key=lambda kv: kv[1]["all"]["recall@1"]
    )
    journal.append(
        f"benchmark sparse retrievers (dense={spec.key})",
        what=f"เทียบ {len(indexes)} วิธี sparse นอกเหนือจาก BM25 ทั้งแบบเดี่ยวและ fuse กับ dense",
        how=(
            f"golden set {len(golden)} คำถาม, dense index จาก {spec.key}, "
            f"RRF k={config.RRF_K}, candidate_k={config.CANDIDATE_K}"
        ),
        result=(
            f"sparse เดี่ยวดีสุด: {best_sparse[0]} Recall@1 "
            f"{best_sparse[1]['all']['recall@1']:.1%}; "
            f"hybrid ดีสุด: dense+{best_hybrid[0]} Recall@1 "
            f"{best_hybrid[1]['all']['recall@1']:.1%}"
        ),
        methods=METHODS,
        build_seconds=payload["build_seconds"],
        sparse_recall_at_1={
            k: round(v["all"]["recall@1"], 4) for k, v in payload["sparse_only"].items()
        },
        hybrid_recall_at_1={
            k: round(v["all"]["recall@1"], 4) for k, v in payload["hybrid"].items()
        },
        hybrid_pinned_recall_at_1={
            k: round(v["all"]["recall@1"], 4)
            for k, v in payload["hybrid_pinned"].items()
        },
    )


if __name__ == "__main__":
    main()
