"""Unit tests for `apply_must_contain_exclude` (Ф3.4), against a stub
`LexicalIndex` double whose `contains_all`/`contains_any` return values are
controlled directly by the test -- exercising the filter's AND semantics,
not the real BM25/tokenizer's matching behavior (see `test_lexical_bm25.py`
for that)."""

from __future__ import annotations

from pathlib import Path

from app.core.types import RetrievedCandidate
from app.lexical.base import LexicalIndex
from app.search.filters import apply_must_contain_exclude


class StubLexicalIndex(LexicalIndex):
    """Test double where `contains_all`/`contains_any` are driven by
    per-chunk_id lookup tables set directly by the test, instead of real
    tokenization/lemmatization."""

    def __init__(
        self,
        contains_all_result: dict[str, bool] | None = None,
        contains_any_result: dict[str, bool] | None = None,
    ) -> None:
        self._contains_all_result = contains_all_result or {}
        self._contains_any_result = contains_any_result or {}

    def build(self, chunks) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        raise NotImplementedError

    def contains_all(self, chunk_id: str, terms: list[str]) -> bool:
        return self._contains_all_result.get(chunk_id, False)

    def contains_any(self, chunk_id: str, terms: list[str]) -> bool:
        return self._contains_any_result.get(chunk_id, False)

    def vocabulary(self) -> set[str]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


def _candidate(chunk_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text="text", score=1.0)


def test_both_empty_is_a_no_op_returns_same_list_object():
    candidates = [_candidate("c1"), _candidate("c2")]
    lexical_index = StubLexicalIndex()

    result = apply_must_contain_exclude(candidates, [], [], lexical_index)

    assert result is candidates


def test_must_contain_keeps_only_matching_chunks():
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    lexical_index = StubLexicalIndex(contains_all_result={"c1": True, "c3": True})

    result = apply_must_contain_exclude(candidates, ["термин"], [], lexical_index)

    assert [c.chunk_id for c in result] == ["c1", "c3"]


def test_must_exclude_drops_matching_chunks():
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    lexical_index = StubLexicalIndex(contains_any_result={"c2": True})

    result = apply_must_contain_exclude(candidates, [], ["запрещено"], lexical_index)

    assert [c.chunk_id for c in result] == ["c1", "c3"]


def test_must_contain_and_must_exclude_compose_with_and_semantics():
    # c1: satisfies must_contain, does NOT match must_exclude -> kept
    # c2: satisfies must_contain, but ALSO matches must_exclude -> dropped
    # c3: does not satisfy must_contain -> dropped regardless of must_exclude
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    lexical_index = StubLexicalIndex(
        contains_all_result={"c1": True, "c2": True, "c3": False},
        contains_any_result={"c1": False, "c2": True, "c3": False},
    )

    result = apply_must_contain_exclude(candidates, ["термин"], ["запрещено"], lexical_index)

    assert [c.chunk_id for c in result] == ["c1"]


def test_no_matches_returns_empty_list():
    candidates = [_candidate("c1"), _candidate("c2")]
    lexical_index = StubLexicalIndex(contains_all_result={})

    result = apply_must_contain_exclude(candidates, ["термин"], [], lexical_index)

    assert result == []
