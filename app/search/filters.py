"""Ф3.4 strict post-filtering: must_contain / must_exclude.

Applies in ALL search modes (dense/bm25/hybrid/hybrid_rerank) — it is a
correctness constraint on which chunks may appear in the result at all,
independent of how they were scored/ranked. Matching goes through
`LexicalIndex.contains_all`/`contains_any`, which tokenize/lemmatize `terms`
with the same `RussianTokenizer` used for BM25, so must_contain/exclude
terms are matched on lemma-normalized tokens rather than literal substrings.
"""

from __future__ import annotations

from app.core.types import RetrievedCandidate
from app.lexical.base import LexicalIndex


def apply_must_contain_exclude(
    candidates: list[RetrievedCandidate],
    must_contain: list[str],
    must_exclude: list[str],
    lexical_index: LexicalIndex,
) -> list[RetrievedCandidate]:
    if not must_contain and not must_exclude:
        return candidates

    filtered = candidates
    if must_contain:
        filtered = [c for c in filtered if lexical_index.contains_all(c.chunk_id, must_contain)]
    if must_exclude:
        filtered = [c for c in filtered if not lexical_index.contains_any(c.chunk_id, must_exclude)]
    return filtered
