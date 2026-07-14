"""Unit tests for `WeightedSumHybridizer` (Ф3.3 alternative to RRF)."""

from __future__ import annotations

from app.core.types import RetrievedCandidate
from app.hybrid.weighted import WeightedSumHybridizer


def _candidate(chunk_id: str, score: float, rank: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text=f"text {chunk_id}", score=score, rank=rank
    )


def test_minmax_normalization_correctness():
    # dense scores 0.0, 0.5, 1.0 -> minmax normalized to 0.0, 0.5, 1.0 (already
    # spans the full range); lexical empty so lexical_weight contributes 0.
    dense = [
        _candidate("c1", score=0.0, rank=3),
        _candidate("c2", score=0.5, rank=2),
        _candidate("c3", score=1.0, rank=1),
    ]
    hybridizer = WeightedSumHybridizer(dense_weight=1.0, lexical_weight=0.0, normalization="minmax")

    fused = hybridizer.fuse(dense, [])
    scores_by_id = {c.chunk_id: c.score for c in fused}

    assert scores_by_id["c1"] == 0.0
    assert scores_by_id["c2"] == 0.5
    assert scores_by_id["c3"] == 1.0
    assert [c.chunk_id for c in fused] == ["c3", "c2", "c1"]


def test_weight_sensitivity_shifting_toward_dense_changes_top_rank():
    # Crafted fixture where dense and lexical disagree on the top candidate:
    # dense ranks c1 highest, lexical ranks c2 highest.
    dense = [
        _candidate("c1", score=1.0, rank=1),
        _candidate("c2", score=0.0, rank=2),
    ]
    lexical = [
        _candidate("c2", score=1.0, rank=1),
        _candidate("c1", score=0.0, rank=2),
    ]

    dense_leaning = WeightedSumHybridizer(dense_weight=0.9, lexical_weight=0.1)
    lexical_leaning = WeightedSumHybridizer(dense_weight=0.1, lexical_weight=0.9)

    fused_dense_leaning = dense_leaning.fuse(dense, lexical)
    fused_lexical_leaning = lexical_leaning.fuse(dense, lexical)

    assert fused_dense_leaning[0].chunk_id == "c1"
    assert fused_lexical_leaning[0].chunk_id == "c2"


def test_divide_by_zero_guard_minmax_all_equal_scores():
    dense = [_candidate("c1", score=0.5, rank=1), _candidate("c2", score=0.5, rank=2)]

    fused = WeightedSumHybridizer(dense_weight=1.0, lexical_weight=0.0, normalization="minmax").fuse(
        dense, []
    )

    assert all(c.score == 1.0 for c in fused)
    assert len(fused) == 2


def test_divide_by_zero_guard_zscore_all_equal_scores():
    dense = [_candidate("c1", score=0.5, rank=1), _candidate("c2", score=0.5, rank=2)]

    fused = WeightedSumHybridizer(dense_weight=1.0, lexical_weight=0.0, normalization="zscore").fuse(
        dense, []
    )

    assert all(c.score == 0.0 for c in fused)
    assert len(fused) == 2


def test_chunk_present_in_only_one_list_defaults_missing_side_to_zero():
    dense = [_candidate("only_dense", score=1.0, rank=1)]

    fused = WeightedSumHybridizer(dense_weight=0.5, lexical_weight=0.5).fuse(dense, [])

    assert len(fused) == 1
    # minmax on a single-element list hits the all-equal guard -> 1.0,
    # lexical side defaults to 0.0 -> fused = 0.5 * 1.0 + 0.5 * 0.0
    assert fused[0].score == 0.5
