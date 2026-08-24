"""Query transformations that run before retrieval.

- rewrite      : turn a context-dependent follow-up into a standalone question
- multi_query  : generate several phrasings and union their results
- hyde         : draft a hypothetical answer and search with that embedding
"""


def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    """Rewrite a follow-up ("what about that one?") into a standalone question."""
    raise NotImplementedError


def multi_query(query: str, n: int = 3) -> list[str]:
    """Return n alternative phrasings of the query (the original included)."""
    raise NotImplementedError


def hyde(query: str) -> str:
    """Generate a hypothetical answer document to embed instead of the question."""
    raise NotImplementedError
