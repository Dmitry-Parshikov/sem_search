"""Unit tests for `RRFHybridizer` (Ф3.3)."""

from __future__ import annotations

from app.core.types import RetrievedCandidate
from app.hybrid.rrf import RRFHybridizer


def _candidate(chunk_id: str, rank: int, score: float = 1.0) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text=f"text {chunk_id}", score=score, rank=rank
    )


def test_rrf_fuses_overlapping_and_disjoint_lists_by_hand_computed_score():
    # dense: c1 rank1, c2 rank2
    # lexical: c2 rank1, c3 rank2
    dense = [_candidate("c1", rank=1), _candidate("c2", rank=2)]
    lexical = [_candidate("c2", rank=1), _candidate("c3", rank=2)]
    k = 60

    hybridizer = RRFHybridizer(k=k)
    fused = hybridizer.fuse(dense, lexical)

    expected_scores = {
        "c1": 1.0 / (k + 1),
        "c2": 1.0 / (k + 2) + 1.0 / (k + 1),
        "c3": 1.0 / (k + 2),
    }

    assert {c.chunk_id for c in fused} == {"c1", "c2", "c3"}
    scores_by_id = {c.chunk_id: c.score for c in fused}
    for chunk_id, expected in expected_scores.items():
        assert scores_by_id[chunk_id] == expected

    # c2 (in both lists) must rank first, since it has the highest fused score.
    assert fused[0].chunk_id == "c2"
    # c1 and c3 both come from a single list at rank1/rank2 respectively --
    # c1's contribution (1/(k+1)) beats c3's (1/(k+2)), so c1 > c3.
    assert [c.chunk_id for c in fused] == ["c2", "c1", "c3"]
    assert [c.rank for c in fused] == [1, 2, 3]


def test_rrf_chunk_present_in_only_one_list_still_gets_a_score():
    dense = [_candidate("only_dense", rank=1)]
    lexical: list[RetrievedCandidate] = []

    fused = RRFHybridizer(k=60).fuse(dense, lexical)

    assert len(fused) == 1
    assert fused[0].chunk_id == "only_dense"
    assert fused[0].score == 1.0 / (60 + 1)


def test_rrf_prefers_dense_candidate_fields_when_present_in_both_lists():
    dense_candidate = RetrievedCandidate(
        chunk_id="c1", doc_id="d-dense", text="dense text", score=0.9, rank=1
    )
    lexical_candidate = RetrievedCandidate(
        chunk_id="c1", doc_id="d-lexical", text="lexical text", score=5.0, rank=1
    )

    fused = RRFHybridizer(k=60).fuse([dense_candidate], [lexical_candidate])

    assert fused[0].doc_id == "d-dense"
    assert fused[0].text == "dense text"


def test_rrf_ordering_is_deterministic_across_repeated_calls():
    dense = [_candidate("a", rank=1), _candidate("b", rank=2)]
    lexical = [_candidate("c", rank=1), _candidate("d", rank=2)]

    hybridizer = RRFHybridizer(k=60)
    first = [c.chunk_id for c in hybridizer.fuse(dense, lexical)]
    second = [c.chunk_id for c in hybridizer.fuse(dense, lexical)]

    assert first == second


def test_rrf_breaks_score_ties_deterministically_by_chunk_id():
    # Two chunks, each appearing only in one list at the same rank -> tied
    # RRF scores. Ordering must not crash and must be stable/deterministic.
    dense = [_candidate("zeta", rank=1)]
    lexical = [_candidate("alpha", rank=1)]

    fused = RRFHybridizer(k=60).fuse(dense, lexical)

    assert fused[0].score == fused[1].score
    # Tie-break is by chunk_id ascending.
    assert [c.chunk_id for c in fused] == ["alpha", "zeta"]
    assert [c.rank for c in fused] == [1, 2]
