"""Online retrievers (Ф3.1 dense, Ф3.2 BM25).

Both retrievers normalize `RetrievedCandidate.rank` to a 1-based rank in the
order returned by the underlying store/index, regardless of whether that
store already set `.rank` (and regardless of its base, 0 or 1) -- Phase 5's
RRF fusion needs a consistent, guaranteed 1-based per-retriever rank to
compute `1 / (k + rank)`, so this is normalized here once rather than
trusted from each backend.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.core.types import RetrievedCandidate
from app.embedding.base import Embedder
from app.lexical.base import LexicalIndex
from app.vector_store.base import VectorStore


def _with_one_based_rank(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    return [replace(candidate, rank=i + 1) for i, candidate in enumerate(candidates)]


class DenseRetriever:
    """Ф3.1: ANN top-k over cosine similarity via `VectorStore.search`,
    against a query vector produced by the SAME embedder instance used at
    indexing time (Ф2.5)."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        query_vector = self._embedder.encode_query(query)
        candidates = self._vector_store.search(collection, query_vector, top_k, metadata_filter)
        return _with_one_based_rank(candidates)


class BM25Retriever:
    """Ф3.2: top-k by BM25 score via `LexicalIndex.search`."""

    def __init__(self, lexical_index: LexicalIndex) -> None:
        self._lexical_index = lexical_index

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        candidates = self._lexical_index.search(query, top_k)
        return _with_one_based_rank(candidates)
