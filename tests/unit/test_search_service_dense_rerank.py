"""Unit tests for `dense_rerank` mode in `SearchService`, against tiny fakes
(no real Qdrant/embedder/cross-encoder). Mirrors the intent of the slow
`test_search_hybrid_rerank.py` integration tests but at the service level:

- `dense_rerank` runs dense retrieval -> must_contain/exclude filter ->
  reranker, and does NOT touch the BM25 lexical index (no fusion step).
- a reranker failure degrades gracefully to the pre-rerank dense order plus a
  `warnings` entry (NFR "Надёжность").
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.types import RetrievedCandidate, Vector
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.lexical.base import LexicalIndex
from app.rerank.base import Reranker
from app.search.active_index import ActiveIndexContext
from app.search.service import SearchService


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


class FakeVectorStore:
    def __init__(self, candidates: list[RetrievedCandidate]) -> None:
        self._candidates = candidates

    def search(
        self, collection: str, query_vector: Vector, top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        return self._candidates[:top_k]


class FakeLexicalIndex(LexicalIndex):
    """Records whether `search` (the BM25 path) is ever called, so the test
    can assert `dense_rerank` never runs BM25."""

    def __init__(self) -> None:
        self.search_called = False

    def build(self, chunks) -> None:  # pragma: no cover - not used here
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.search_called = True
        return []

    def contains_all(self, chunk_id: str, terms: list[str]) -> bool:
        return True

    def contains_any(self, chunk_id: str, terms: list[str]) -> bool:
        return False

    def vocabulary(self) -> set[str]:
        return set()

    def save(self, path) -> None:  # pragma: no cover - not used here
        raise NotImplementedError

    def load(self, path) -> None:  # pragma: no cover - not used here
        raise NotImplementedError


class FakeResolver:
    def __init__(self, lexical_index: LexicalIndex) -> None:
        self._lexical_index = lexical_index

    def resolve(self) -> ActiveIndexContext:
        return ActiveIndexContext(
            index_version="v1",
            collection_name="chunks",
            lexical_index=self._lexical_index,
        )


class UnusedHybridizer(Hybridizer):
    def fuse(self, dense, lexical):  # pragma: no cover - must not be called
        raise AssertionError("dense_rerank must not fuse with BM25")


class ReversingReranker(Reranker):
    """Returns candidates in reversed order so the test can prove reranking
    actually reordered the dense result."""

    def rerank(self, query, candidates, top_n):
        return list(reversed(candidates[:top_n]))

    @property
    def model_name(self) -> str:
        return "reversing-fake"


class RaisingReranker(Reranker):
    def rerank(self, query, candidates, top_n):
        raise RuntimeError("boom: reranker backend unavailable")

    @property
    def model_name(self) -> str:
        return "raising-fake"


class NoopLogger:
    def log(self, entry: dict[str, Any]) -> None:
        pass


def _candidates(n: int) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"text {i}", score=1.0 / (i + 1))
        for i in range(n)
    ]


def _service(reranker: Reranker | None, lexical_index: FakeLexicalIndex) -> SearchService:
    return SearchService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(_candidates(4)),
        active_index_resolver=FakeResolver(lexical_index),
        hybridizer=UnusedHybridizer(),
        settings=Settings(),
        query_logger=NoopLogger(),
        typo_corrector=None,
        term_expander=None,
        reranker=reranker,
    )


def test_dense_rerank_reorders_dense_result_and_skips_bm25():
    lexical_index = FakeLexicalIndex()
    service = _service(ReversingReranker(), lexical_index)

    dense = service.search("запрос", mode="dense", top_k=4)
    reranked = service.search("запрос", mode="dense_rerank", top_k=4)

    dense_order = [h.chunk_id for h in dense.hits]
    reranked_order = [h.chunk_id for h in reranked.hits]

    assert reranked.mode == "dense_rerank"
    assert dense_order == ["c0", "c1", "c2", "c3"]
    assert reranked_order == ["c3", "c2", "c1", "c0"]
    assert not lexical_index.search_called


def test_dense_rerank_degrades_gracefully_when_reranker_fails():
    lexical_index = FakeLexicalIndex()
    service = _service(RaisingReranker(), lexical_index)

    result = service.search("запрос", mode="dense_rerank", top_k=4)

    assert [h.chunk_id for h in result.hits] == ["c0", "c1", "c2", "c3"]
    assert result.warnings
    assert any("rerank" in w.lower() for w in result.warnings)


def test_dense_rerank_without_reranker_returns_plain_dense_order():
    lexical_index = FakeLexicalIndex()
    service = _service(None, lexical_index)

    result = service.search("запрос", mode="dense_rerank", top_k=4)

    assert [h.chunk_id for h in result.hits] == ["c0", "c1", "c2", "c3"]
    assert not lexical_index.search_called
