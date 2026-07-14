"""Weighted-sum hybridization (Ф3.3, alternative to RRF).

Unlike RRF (rank-based), this fuses raw scores -- which requires normalizing
each list's scores onto a comparable scale first, since dense cosine
similarity and BM25 scores live on unrelated scales.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Literal

from app.core.types import RetrievedCandidate
from app.hybrid.base import Hybridizer


def _normalize_minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        # All-equal scores: dividing by (hi - lo) would be a divide-by-zero.
        # Map them all to a constant (1.0) instead of crashing/NaN-ing --
        # an arbitrary but stable choice, consistent with "no information to
        # discriminate between them".
        return {cid: 1.0 for cid in scores}
    return {cid: (value - lo) / (hi - lo) for cid, value in scores.items()}


def _normalize_zscore(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    mean = statistics.fmean(values)
    if len(values) < 2:
        stdev = 0.0
    else:
        stdev = statistics.pstdev(values)
    if stdev == 0.0:
        # All-equal scores (or a single candidate): guard divide-by-zero by
        # mapping everything to 0.0 (the "average", z-score-neutral value)
        # rather than dividing by zero.
        return {cid: 0.0 for cid in scores}
    return {cid: (value - mean) / stdev for cid, value in scores.items()}


class WeightedSumHybridizer(Hybridizer):
    def __init__(
        self,
        dense_weight: float,
        lexical_weight: float,
        normalization: Literal["minmax", "zscore"] = "minmax",
    ) -> None:
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        self._normalization = normalization

    def _normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if self._normalization == "minmax":
            return _normalize_minmax(scores)
        return _normalize_zscore(scores)

    def fuse(
        self, dense: list[RetrievedCandidate], lexical: list[RetrievedCandidate]
    ) -> list[RetrievedCandidate]:
        dense_raw = {c.chunk_id: c.score for c in dense}
        lexical_raw = {c.chunk_id: c.score for c in lexical}
        dense_norm = self._normalize(dense_raw)
        lexical_norm = self._normalize(lexical_raw)

        representative: dict[str, RetrievedCandidate] = {}
        for candidate in lexical:
            representative[candidate.chunk_id] = candidate
        for candidate in dense:
            representative[candidate.chunk_id] = candidate

        all_ids = set(dense_raw) | set(lexical_raw)
        fused_scores = {
            cid: self._dense_weight * dense_norm.get(cid, 0.0)
            + self._lexical_weight * lexical_norm.get(cid, 0.0)
            for cid in all_ids
        }

        ordered_ids = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))

        fused: list[RetrievedCandidate] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            base = representative[chunk_id]
            fused.append(replace(base, score=fused_scores[chunk_id], rank=rank))
        return fused
