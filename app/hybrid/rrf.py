"""Reciprocal Rank Fusion (Ф3.3, main hybridization method).

RRF fuses two ranked lists by rank alone (not raw score), which sidesteps
the need to normalize dense cosine-similarity scores against BM25 scores --
two scales that are not directly comparable. Each list already carries a
1-based `.rank` (see `app.search.retrievers`'s `_with_one_based_rank`), so
fusion here trusts that rank directly rather than re-deriving it.
"""

from __future__ import annotations

from dataclasses import replace

from app.core.types import RetrievedCandidate
from app.hybrid.base import Hybridizer


class RRFHybridizer(Hybridizer):
    """`fused_score(chunk) = sum(1 / (k + rank_in_list))` over every list the
    chunk appears in. A chunk present in only one list still gets a score
    from that list alone -- it is not required to appear in both."""

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(
        self, dense: list[RetrievedCandidate], lexical: list[RetrievedCandidate]
    ) -> list[RetrievedCandidate]:
        scores: dict[str, float] = {}
        # Prefer the dense list's copy of a chunk when it appears in both
        # (arbitrary but consistent choice, documented in the plan) -- so
        # populate from lexical first, then let dense overwrite.
        representative: dict[str, RetrievedCandidate] = {}

        for candidate in lexical:
            assert candidate.rank is not None, "lexical candidates must carry a 1-based rank"
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (
                self._k + candidate.rank
            )
            representative[candidate.chunk_id] = candidate

        for candidate in dense:
            assert candidate.rank is not None, "dense candidates must carry a 1-based rank"
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (
                self._k + candidate.rank
            )
            representative[candidate.chunk_id] = candidate

        # Sort by fused score descending; break ties by chunk_id so ordering
        # is deterministic (and stable across repeated calls with the same
        # input) rather than depending on dict-iteration/insertion order.
        ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))

        fused: list[RetrievedCandidate] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            base = representative[chunk_id]
            fused.append(replace(base, score=scores[chunk_id], rank=rank))
        return fused
