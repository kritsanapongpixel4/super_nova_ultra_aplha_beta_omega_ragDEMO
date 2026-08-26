"""Retrieval metrics.

Every function takes the ranked chunk ids the retriever returned and the set of
ids that are actually relevant, and returns a score in [0, 1].
"""

import math


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of all relevant chunks that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the top k that is relevant."""
    if k <= 0:
        return 0.0
    # Divided by k, not by len(retrieved[:k]): a retriever that returns three
    # chunks when asked for ten has not earned the precision of one that
    # returned three good ones out of three.
    return len(set(retrieved[:k]) & relevant) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant chunk (0.0 if none)."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain - rewards ranking hits higher."""
    if not relevant:
        return 0.0
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    # The ideal ranking puts every relevant chunk first — but no more of them
    # than k slots allow.
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return gain / ideal if ideal else 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric across all queries."""
    if not per_query:
        return {}
    names = per_query[0].keys()
    return {
        name: sum(row[name] for row in per_query) / len(per_query) for name in names
    }


def score_one(
    retrieved: list[str],
    relevant: set[str],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    *,
    retrieved_sources: list[str] | None = None,
    relevant_source: str | None = None,
) -> dict[str, float]:
    """Every metric for one query, keyed the way the report tables want them.

    Pass the source filenames too and ``doc_hit@k`` is added: did the top k
    include *any* chunk from the document the answer lives in?

    It is reported next to ``hit@k``, not instead of it, because the two
    disagree for a reason worth seeing.  Several registrar documents carry the
    same sentence — "ส่งที่ภาควิชาฯ", "7 วันทำการ" — so a question can have a
    genuinely correct answer in a chunk the golden set does not name, and
    ``hit@k`` scores that 0.  ``doc_hit@k`` is the looser reading; the gap
    between them is how much of the miss is labelling rather than retrieval.
    """
    scores: dict[str, float] = {"mrr": mrr(retrieved, relevant)}
    for k in k_values:
        scores[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    if retrieved_sources is not None and relevant_source:
        for k in k_values:
            scores[f"doc_hit@{k}"] = (
                1.0 if relevant_source in retrieved_sources[:k] else 0.0
            )
    return scores
