"""Retrieval quality of the pipeline as it actually runs, per question type.

The other two benchmarks answer narrower questions: bench_embeddings.py
compares embedders with every other layer switched off, bench_retrievers.py
compares sparse methods.  Neither says what a user gets, because what a user
gets is all the layers at once — dense + sparse + RRF + pinning + rerank.

This runs every answerable question in the golden set through RAGPipeline
itself, so the numbers describe the shipped system and can be quoted as such.
Quoting the other two side by side instead is how the README ended up with a
summary table assembled from three different runs, two of which had the
reranker switched off.

Read Hit@k here, not recall: a curriculum question has several gold chunks and
recall divides by all of them, so it cannot reach 1.0 at k=1 however well
retrieval did.

Run: python benchmarks/bench_pipeline.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluation import golden_set, metrics  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

K = (1, 5)


def main() -> None:
    pipeline = RAGPipeline.from_config(use_memory=False)
    chunks = pipeline.retriever.store.chunks
    entries, report = golden_set.load(chunks, quiet=True)
    fp = report["corpus_now"]

    print(f"คลัง {fp['n_chunks']} chunks / {fp['n_sources']} ไฟล์  {fp['chunks_sha1']}")
    print(f"คำถาม {len(entries)} ข้อ  {report['questions_sha1']}")
    print(f"ค่าที่ใช้: {config.EMBEDDING_MODEL} · sparse={config.SPARSE_METHOD}"
          f" · rerank={config.USE_RERANKER} · top_k={config.TOP_K}"
          f" · candidate_k={config.CANDIDATE_K}\n")

    groups = defaultdict(list)
    t0 = time.perf_counter()
    for n, entry in enumerate(entries, start=1):
        got = [str(c["chunk_id"]) for c in pipeline.retrieve(entry["question"])]
        score = metrics.score_one(got, set(entry["relevant_chunk_ids"]), K)
        label = entry["category"]
        if entry.get("phrasing") in ("by_name", "by_code"):
            label = f"{label}/{entry['phrasing']}"
        groups[label].append(score)
        groups["__all__"].append(score)
        if n % 50 == 0:
            print(f"  {n}/{len(entries)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nเสร็จใน {elapsed:.0f}s  ({elapsed/len(entries)*1000:.0f} ms/คำถาม)\n")
    print(f"{'หมวด':26s}{'n':>5s}{'Hit@1':>9s}{'Hit@5':>9s}{'MRR':>8s}")
    for label in sorted(groups):
        if label == "__all__":
            continue
        agg = metrics.aggregate(groups[label])
        print(f"{label:26s}{len(groups[label]):>5d}"
              f"{agg['hit@1']:>8.1%}{agg['hit@5']:>9.1%}{agg['mrr']:>8.3f}")
    agg = metrics.aggregate(groups["__all__"])
    print(f"{'รวม':26s}{len(groups['__all__']):>5d}"
          f"{agg['hit@1']:>8.1%}{agg['hit@5']:>9.1%}{agg['mrr']:>8.3f}")


if __name__ == "__main__":
    main()
