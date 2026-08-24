"""Retrieval metrics.

Every function takes the ranked chunk ids the retriever returned and the set of
ids that are actually relevant, and returns a score in [0, 1].
"""


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0."""
    raise NotImplementedError


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of all relevant chunks that appear in the top k."""
    raise NotImplementedError


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the top k that is relevant."""
    raise NotImplementedError


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant chunk (0.0 if none)."""
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain - rewards ranking hits higher."""
    raise NotImplementedError


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric across all queries."""
    raise NotImplementedError
