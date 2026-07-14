"""Unit tests for `DenseRetriever`/`BM25Retriever` (Ф3.1/Ф3.2) against tiny
fake `VectorStore`/`LexicalIndex` doubles -- no real Qdrant/BM25 needed to
confirm top_k is respected and rank is assigned starting at 1."""

from __future__ import annotations

from typing import Any

from app.core.types import RetrievedCandidate, Vector
from app.embedding.base import Embedder
from app.lexical.base import LexicalIndex
from app.search.retrievers import BM25Retriever, DenseRetriever
from app.vector_store.base import VectorStore


class FakeEmbedder(Embedder):
    def encode_documents(self, texts: list[str]) -> list[Vector]:
        return [[0.0] for _ in texts]

    def encode_query(self, text: str) -> Vector:
        return [1.0, 2.0, 3.0]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "fake-embedder"


class FakeVectorStore(VectorStore):
    """Returns a fixed candidate list, unranked (rank=None), to confirm the
    retriever assigns rank itself rather than trusting the store."""

    def __init__(self, candidates: list[RetrievedCandidate]) -> None:
        self._candidates = candidates
        self.last_query_vector: Vector | None = None
        self.last_top_k: int | None = None
        self.last_metadata_filter: dict[str, Any] | None = None

    def create_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        raise NotImplementedError

    def upsert(self, collection: str, chunks, vectors) -> None:
        raise NotImplementedError

    def search(
        self,
        collection: str,
        query_vector: Vector,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        self.last_query_vector = query_vector
        self.last_top_k = top_k
        self.last_metadata_filter = metadata_filter
        return self._candidates[:top_k]

    def delete_collection(self, name: str) -> None:
        raise NotImplementedError

    def collection_exists(self, name: str) -> bool:
        raise NotImplementedError

    def health(self) -> bool:
        return True


class FakeLexicalIndex(LexicalIndex):
    def __init__(self, candidates: list[RetrievedCandidate]) -> None:
        self._candidates = candidates
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def build(self, chunks) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.last_query = query
        self.last_top_k = top_k
        return self._candidates[:top_k]

    def contains_all(self, chunk_id: str, terms: list[str]) -> bool:
        raise NotImplementedError

    def contains_any(self, chunk_id: str, terms: list[str]) -> bool:
        raise NotImplementedError

    def vocabulary(self) -> set[str]:
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError

    def load(self, path) -> None:
        raise NotImplementedError


def _candidates(n: int) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"text {i}", score=1.0 / (i + 1))
        for i in range(n)
    ]


def test_dense_retriever_respects_top_k_and_assigns_one_based_rank():
    store = FakeVectorStore(_candidates(5))
    retriever = DenseRetriever(FakeEmbedder(), store)

    results = retriever.retrieve("запрос", collection="chunks", top_k=3)

    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert store.last_top_k == 3
    assert store.last_query_vector == [1.0, 2.0, 3.0]


def test_dense_retriever_passes_metadata_filter_through():
    store = FakeVectorStore(_candidates(2))
    retriever = DenseRetriever(FakeEmbedder(), store)

    retriever.retrieve("запрос", collection="chunks", top_k=2, metadata_filter={"section": "intro"})

    assert store.last_metadata_filter == {"section": "intro"}


def test_bm25_retriever_respects_top_k_and_assigns_one_based_rank():
    lexical_index = FakeLexicalIndex(_candidates(4))
    retriever = BM25Retriever(lexical_index)

    results = retriever.retrieve("договор аренды", top_k=2)

    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]
    assert lexical_index.last_query == "договор аренды"
    assert lexical_index.last_top_k == 2


def test_retrievers_overwrite_existing_rank_to_stay_one_based():
    """Even when the backend already sets a (possibly 0-based) `.rank`, the
    retriever normalizes it to a 1-based sequence."""

    pre_ranked = [
        RetrievedCandidate(chunk_id="c0", doc_id="d0", text="t0", score=0.9, rank=0),
        RetrievedCandidate(chunk_id="c1", doc_id="d1", text="t1", score=0.5, rank=1),
    ]
    store = FakeVectorStore(pre_ranked)
    retriever = DenseRetriever(FakeEmbedder(), store)

    results = retriever.retrieve("запрос", collection="chunks", top_k=2)

    assert [r.rank for r in results] == [1, 2]
