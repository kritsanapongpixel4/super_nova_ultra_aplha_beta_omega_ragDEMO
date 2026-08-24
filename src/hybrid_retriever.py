"""Hybrid retrieval - BM25 (exact tokens) + Dense (semantics), fused with RRF.

BM25 catches literal terms the embedder blurs away; dense catches paraphrases
BM25 misses. Reciprocal Rank Fusion merges the two rankings without needing
their scores to be on a comparable scale.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from pythainlp.tokenize import word_tokenize

from .embedding_model import EmbeddingModel
from .vector_store import VectorStore

# A token has to carry at least one Thai or Latin letter, or a digit — this
# drops the whitespace and punctuation that newmm hands back as its own
# tokens, which would otherwise match every document equally.
_MEANINGFUL = re.compile(r"[ก-๙A-Za-z0-9]")

# newmm splits "04-620-201" into "04", "620", "201", so a course code stops
# being the precise term it is — "620" alone matches half the curriculum.
# The full code is emitted as an extra token so an exact code query still has
# something exact to hit, which is the whole reason BM25 is in the mix.
_COURSE_CODE = re.compile(r"\d{2}-\d{3}-\d{3}")


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25 (Thai needs a word segmenter, not whitespace splitting)."""
    lowered = text.lower()
    tokens = [
        token
        for raw in word_tokenize(lowered, engine="newmm")
        if (token := raw.strip()) and _MEANINGFUL.search(token)
    ]
    tokens.extend(_COURSE_CODE.findall(lowered))
    return tokens


class BM25Index:
    """Sparse, exact-token index persisted as bm25_index.pkl."""

    def __init__(self) -> None:
        self.bm25 = None
        self.chunks: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        # BM25Okapi divides by the average document length, so an empty
        # corpus would blow up here rather than at query time.
        if not corpus:
            raise ValueError("cannot build a BM25 index from zero chunks")
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        if self.bm25 is None:
            raise RuntimeError("index is empty — call build() or load() first")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order[: max(1, k)], start=1):
            if scores[index] <= 0:
                break  # nothing below this shares a single term with the query
            hit = dict(self.chunks[index])
            hit["score"] = float(scores[index])
            hit["rank"] = rank
            results.append(hit)
        return results

    def save(self, path: Path) -> None:
        if self.bm25 is None:
            raise RuntimeError("nothing to save — call build() first")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run pipeline/complete_retrieval.py to build it"
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        index = cls()
        index.bm25 = payload["bm25"]
        index.chunks = payload["chunks"]
        return index


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse several ranked lists: score = sum of 1 / (rrf_k + rank).

    Chunks are matched across lists by chunk_id.
    """
    scores: dict[Any, float] = {}
    seen: dict[Any, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk.get("chunk_id")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            # Keep the first copy we saw, but remember each retriever's own
            # score so the fused result can still be explained afterwards.
            entry = seen.setdefault(chunk_id, dict(chunk))
            if "score" in chunk:
                entry.setdefault("component_scores", []).append(chunk["score"])

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    fused: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ordered[: max(1, k)], start=1):
        hit = seen[chunk_id]
        hit["score"] = scores[chunk_id]
        hit["rank"] = rank
        fused.append(hit)
    return fused


def exact_code_matches(
    query: str, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Chunks whose ``course_code`` is written literally in *query*.

    A course code is an exact identifier, not a topic.  Neither retriever
    handles it well once every course card shares the same shape: the code is
    ten characters inside ~200 tokens, so the embedder barely sees it, and
    while BM25 does find it, RRF then buries the result — fusion rewards
    chunks both lists agree on, and a chunk dense ranked nowhere loses to
    mediocre-but-agreed-upon ones.  Measured 2026-08-24: for the query
    "CLO ของวิชา 04-620-201 มีกี่ข้อ" the right card sat at BM25 rank 6 and
    hybrid rank 14.

    Looking the code up directly sidesteps all of that.
    """
    codes = set(_COURSE_CODE.findall(query.lower()))
    if not codes:
        return []
    return [chunk for chunk in chunks if chunk.get("course_code") in codes]


class HybridRetriever:
    """BM25 + Dense + RRF fusion, with exact course codes pinned to the front."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingModel,
        bm25: BM25Index,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        k: int,
        candidate_k: int = 20,
        rrf_k: int = 60,
        pin_exact_codes: bool = True,
    ) -> list[dict[str, Any]]:
        """Pull candidate_k from each retriever, fuse the rankings, return the top k.

        Set ``pin_exact_codes=False`` to measure plain fusion — the evaluation
        needs an unpinned baseline to show what the pinning is worth.
        """
        dense_hits = self.store.search(self.embedder.encode_query(query), candidate_k)
        sparse_hits = self.bm25.search(query, candidate_k)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=candidate_k, rrf_k=rrf_k)

        pinned = exact_code_matches(query, self.store.chunks) if pin_exact_codes else []

        results: list[dict[str, Any]] = []
        taken: set[Any] = set()
        for chunk in [*pinned, *fused]:
            chunk_id = chunk.get("chunk_id")
            if chunk_id in taken:
                continue
            taken.add(chunk_id)
            hit = dict(chunk)
            hit["rank"] = len(results) + 1
            # Pinned chunks never went through fusion, so they carry no score.
            # Say so rather than inventing a number that looks comparable.
            hit.setdefault("score", float("nan"))
            hit["pinned"] = chunk in pinned
            results.append(hit)
            if len(results) >= max(1, k):
                break
        return results
