"""Sparse retrievers other than BM25-Okapi, for comparison.

BM25 is not one algorithm, and it is not the only sparse family.  This module
implements the alternatives worth measuring on a Thai corpus, all behind the
same interface as ``hybrid_retriever.BM25Index`` (``build``/``search``/
``save``/``load``/``__len__``) so any of them can be dropped into
``HybridRetriever`` without touching it.

Three lineages, chosen because they fail differently:

**BM25 variants** — ``BM25L`` and ``BM25+`` change only how document length is
penalised.  Okapi's length normalisation over-penalises long documents, and
both variants patch that in different ways.  This corpus mixes 60-token form
lines with 700-token course cards, so the fix is not academic.

**Vector-space TF-IDF** — a different scoring family: cosine over tf-idf
weights rather than a probabilistic relevance score.  Two variants, and the
difference between them is the interesting one.  The word-level version
depends on PyThaiNLP segmenting correctly; the character n-gram version does
not tokenize into words at all, which on Thai is a genuine structural
advantage — no segmenter means no segmenter errors, and a course code like
``04-620-201`` survives as overlapping character runs instead of being split
into ``04``/``620``/``201``.

**Query-likelihood with Dirichlet smoothing** — the other classic
probabilistic model, scoring a document by how likely it was to have
generated the query.  Its smoothing behaves quite differently from BM25's on
short queries, which is what this system mostly gets.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from .hybrid_retriever import tokenize


class SparseIndex:
    """Shared plumbing: hold the chunks, persist, report length.

    Subclasses implement ``_fit`` (build whatever structure they score with)
    and ``_scores`` (return one score per chunk for a query).
    """

    name = "sparse"

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    # ── To implement ────────────────────────────────────────────────────

    def _fit(self, corpus: list[list[str]]) -> None:
        raise NotImplementedError

    def _scores(self, query: str, tokens: list[str]) -> list[float]:
        """One score per chunk.  Both forms of the query are supplied:
        *query* is the raw text, *tokens* the segmented form.  A character
        n-gram index needs the raw text — segmenting it and rejoining changes
        the whitespace, and whitespace is exactly what ``char_wb`` uses to
        find word boundaries."""
        raise NotImplementedError

    # ── Common ──────────────────────────────────────────────────────────

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        if not corpus:
            raise ValueError(f"cannot build a {self.name} index from zero chunks")
        self._fit(corpus)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return the k best chunks, best first, each with score and rank."""
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []

        scores = self._scores(query, tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order[: max(1, k)], start=1):
            if scores[index] <= 0:
                break  # shares no term with the query
            hit = dict(self.chunks[index])
            hit["score"] = float(scores[index])
            hit["rank"] = rank
            results.append(hit)
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — build the index first")
        with open(path, "rb") as f:
            return pickle.load(f)


class RankBM25Index(SparseIndex):
    """BM25L or BM25+ from rank_bm25, whichever ``variant`` names."""

    def __init__(self, variant: str = "bm25l") -> None:
        super().__init__()
        self.variant = variant
        self.name = variant
        self.bm25 = None

    def _fit(self, corpus: list[list[str]]) -> None:
        from rank_bm25 import BM25L, BM25Okapi, BM25Plus

        classes = {"bm25": BM25Okapi, "bm25l": BM25L, "bm25plus": BM25Plus}
        if self.variant not in classes:
            raise ValueError(f"unknown BM25 variant {self.variant!r}")
        self.bm25 = classes[self.variant](corpus)

    def _scores(self, query: str, tokens: list[str]) -> list[float]:
        return list(self.bm25.get_scores(tokens))


class TfidfIndex(SparseIndex):
    """Cosine similarity over tf-idf weights.

    ``analyzer="char_wb"`` is the variant that matters for Thai: it never
    calls a word segmenter, so segmentation mistakes cannot cost a match,
    and a run of digits and hyphens stays recognisable as itself.
    """

    def __init__(self, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1)):
        super().__init__()
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.name = f"tfidf-{analyzer}"
        self.vectorizer = None
        self.matrix = None

    def _fit(self, corpus: list[list[str]]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.analyzer == "word":
            # Pre-tokenized by PyThaiNLP, so hand the tokens straight through
            # rather than letting sklearn's whitespace splitter loose on Thai.
            documents = [" ".join(tokens) for tokens in corpus]
            self.vectorizer = TfidfVectorizer(
                analyzer="word", tokenizer=str.split, lowercase=False,
                token_pattern=None,
            )
        else:
            documents = [chunk["text"].lower() for chunk in self.chunks]
            self.vectorizer = TfidfVectorizer(
                analyzer=self.analyzer, ngram_range=self.ngram_range, lowercase=True
            )
        self.matrix = self.vectorizer.fit_transform(documents)

    def _scores(self, query: str, tokens: list[str]) -> list[float]:
        # The word index was fitted on space-joined tokens, so a query has to
        # be presented the same way.  The character index was fitted on raw
        # chunk text, so the query must stay raw — rejoining tokens without
        # spaces would erase the word boundaries char_wb keys off, and the
        # query would be n-grammed differently from every document it is
        # being compared against.
        prepared = " ".join(tokens) if self.analyzer == "word" else query.lower()
        vector = self.vectorizer.transform([prepared])
        # Both sides are L2-normalised by TfidfVectorizer, so the dot product
        # is already the cosine.
        return list((self.matrix @ vector.T).toarray().ravel())


class DirichletLMIndex(SparseIndex):
    """Query-likelihood language model with Dirichlet smoothing.

    score(d, q) = Σ_t log( (tf(t,d) + mu * P(t|C)) / (|d| + mu) )

    The collection model P(t|C) is what rescues a document that is missing
    one query term — under raw maximum likelihood that term contributes
    log(0) and the document is eliminated outright, however well it matches
    everything else.
    """

    name = "dirichlet-lm"

    def __init__(self, mu: float = 2000.0) -> None:
        super().__init__()
        self.mu = mu
        self.term_freqs: list[Counter] = []
        self.lengths: list[int] = []
        self.collection: dict[str, float] = {}

    def _fit(self, corpus: list[list[str]]) -> None:
        self.term_freqs = [Counter(tokens) for tokens in corpus]
        self.lengths = [len(tokens) for tokens in corpus]
        total = sum(self.lengths) or 1
        collection_counts: Counter = Counter()
        for tokens in corpus:
            collection_counts.update(tokens)
        self.collection = {
            term: count / total for term, count in collection_counts.items()
        }

    def _scores(self, query: str, tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freqs, length in zip(self.term_freqs, self.lengths):
            total = 0.0
            for term in tokens:
                prior = self.collection.get(term, 0.0)
                if prior == 0.0:
                    # A term nowhere in the collection tells the ranking
                    # nothing — every document is equally without it.
                    continue
                total += math.log(
                    (freqs.get(term, 0) + self.mu * prior) / (length + self.mu)
                )
            scores.append(total)

        # Log-probabilities are all negative, and SparseIndex.search() stops
        # at the first non-positive score.  Shift so the ordering survives
        # into a range the shared code can read.
        if not scores:
            return scores
        floor = min(scores)
        return [score - floor + 1e-9 for score in scores]


#: Everything bench_retrievers.py knows how to build, by name.
BUILDERS = {
    "bm25": lambda: RankBM25Index("bm25"),
    "bm25l": lambda: RankBM25Index("bm25l"),
    "bm25plus": lambda: RankBM25Index("bm25plus"),
    "tfidf-word": lambda: TfidfIndex("word"),
    "tfidf-char": lambda: TfidfIndex("char_wb", (3, 5)),
    "dirichlet-lm": lambda: DirichletLMIndex(),
}


def build(name: str, chunks: list[dict[str, Any]]) -> SparseIndex:
    """Build one named sparse index over *chunks*."""
    if name not in BUILDERS:
        raise KeyError(f"ไม่รู้จัก sparse retriever {name!r} — มี: {', '.join(BUILDERS)}")
    index = BUILDERS[name]()
    index.build(chunks)
    return index
