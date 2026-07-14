"""Unit tests for `app.rerank.factory`: `build_reranker`/`get_or_build_reranker`
return `None` when disabled (so `SearchService` skips the reranking stage
entirely, same pattern as `build_typo_corrector`/`build_term_expander` in
`app.query.factory`), a real `CrossEncoderReranker` when enabled, and
`get_or_build_reranker` caches by config so repeated calls with the same
config don't reload the model.

Building a real instance loads the dev cross-encoder model, so marked slow.
"""

from __future__ import annotations

import pytest

from app.config import DEV_RERANKER_MODEL, RerankingConfig
from app.rerank.cross_encoder import CrossEncoderReranker
from app.rerank.factory import build_reranker, get_or_build_reranker


def test_build_reranker_returns_none_when_disabled():
    """No model loading involved -- kept out of the `slow` bucket so it still
    runs in the fast dev loop (`pytest -m "not slow"`)."""

    cfg = RerankingConfig(enabled=False)

    assert build_reranker(cfg) is None


def test_get_or_build_reranker_returns_none_when_disabled():
    cfg = RerankingConfig(enabled=False)

    assert get_or_build_reranker(cfg) is None


@pytest.mark.slow
def test_build_reranker_returns_real_instance_when_enabled():
    cfg = RerankingConfig(enabled=True, model_name=DEV_RERANKER_MODEL, device="cpu", batch_size=8)

    reranker = build_reranker(cfg)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == DEV_RERANKER_MODEL


@pytest.mark.slow
def test_get_or_build_reranker_returns_same_instance_for_same_config():
    cfg = RerankingConfig(enabled=True, model_name=DEV_RERANKER_MODEL, device="cpu", batch_size=8)

    first = get_or_build_reranker(cfg)
    second = get_or_build_reranker(cfg)

    assert first is second
