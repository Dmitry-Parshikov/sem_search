"""Unit tests for `CrossEncoderReranker` (Ф3.5).

Loads the real dev cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
see `app.config.DEV_RERANKER_MODEL`), so marked slow.
"""

from __future__ import annotations

import pytest

from app.config import DEV_RERANKER_MODEL
from app.core.types import RetrievedCandidate
from app.rerank.cross_encoder import CrossEncoderReranker

pytestmark = pytest.mark.slow

QUERY = "договор аренды жилого помещения"

# Deliberately ordered so the truly relevant candidate ("relevant") is NOT
# first -- simulating a naive/BM25 pre-rerank order that a cross-encoder,
# which scores query+doc jointly, should be able to fix.
_CANDIDATES = [
    RetrievedCandidate(
        chunk_id="art",
        doc_id="d-art",
        text="Сегодня в городе открылась новая выставка современного искусства.",
        score=0.9,
        rank=1,
    ),
    RetrievedCandidate(
        chunk_id="api",
        doc_id="d-api",
        text="API — программный интерфейс, позволяющий приложениям обмениваться данными.",
        score=0.8,
        rank=2,
    ),
    RetrievedCandidate(
        chunk_id="relevant",
        doc_id="d-relevant",
        text="Договор аренды жилого помещения — это соглашение, по которому наймодатель "
        "предоставляет нанимателю квартиру за плату во временное пользование.",
        score=0.7,
        rank=3,
    ),
    RetrievedCandidate(
        chunk_id="weather",
        doc_id="d-weather",
        text="Синоптики прогнозируют похолодание и осадки в выходные дни.",
        score=0.6,
        rank=4,
    ),
]


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=DEV_RERANKER_MODEL, device="cpu", batch_size=8)


def test_rerank_reorders_so_the_truly_relevant_candidate_ranks_first(reranker):
    reranked = reranker.rerank(QUERY, _CANDIDATES, top_n=len(_CANDIDATES))

    assert {c.chunk_id for c in reranked} == {c.chunk_id for c in _CANDIDATES}
    assert reranked[0].chunk_id == "relevant"


def test_rerank_assigns_fresh_1_based_ranks(reranker):
    reranked = reranker.rerank(QUERY, _CANDIDATES, top_n=len(_CANDIDATES))

    assert [c.rank for c in reranked] == list(range(1, len(_CANDIDATES) + 1))


def test_rerank_scores_are_descending(reranker):
    reranked = reranker.rerank(QUERY, _CANDIDATES, top_n=len(_CANDIDATES))

    scores = [c.score for c in reranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_bounds_to_top_n_before_scoring(reranker):
    """`top_n` bounds which candidates are even considered -- taken from the
    FRONT of the input list, before any scoring happens. Here the truly
    relevant candidate sits at input position 3 (0-indexed), so a top_n of 2
    must exclude it entirely, not just deprioritize it."""

    top_n = 2
    reranked = reranker.rerank(QUERY, _CANDIDATES, top_n=top_n)

    assert len(reranked) == top_n
    assert {c.chunk_id for c in reranked} == {"art", "api"}
    assert "relevant" not in {c.chunk_id for c in reranked}


def test_rerank_empty_candidates_returns_empty(reranker):
    assert reranker.rerank(QUERY, [], top_n=10) == []
