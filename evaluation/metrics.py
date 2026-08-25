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
) -> dict[str, float]:
    """Every metric for one query, keyed the way the report tables want them."""
    scores: dict[str, float] = {"mrr": mrr(retrieved, relevant)}
    for k in k_values:
        scores[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return scores
