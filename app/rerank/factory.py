from __future__ import annotations

from app.config import RerankingConfig
from app.rerank.base import Reranker
from app.rerank.cross_encoder import CrossEncoderReranker

# Process-wide cache of built rerankers, keyed by the config's JSON
# representation -- mirrors `app.embedding.factory.get_or_build_embedder`.
# Loading a `CrossEncoder` is expensive (real model weights from disk), and
# unlike the embedder there is no architectural requirement that indexing and
# querying share one instance -- but many short-lived test `TestClient`/app
# instances in the same process (each with the same default `reranking`
# config) would otherwise each pay that cost again, which is the practical
# reason this cache exists.
_RERANKER_CACHE: dict[str, Reranker] = {}


def build_reranker(cfg: RerankingConfig) -> Reranker | None:
    """`None` when disabled, so `SearchService` skips the reranking stage
    entirely (mirrors `build_typo_corrector`/`build_term_expander` in
    `app.query.factory`). Always constructs a fresh instance -- use
    `get_or_build_reranker` where reusing a process-wide singleton for a
    given config is desirable (e.g. `app.main`'s lifespan)."""

    if not cfg.enabled:
        return None
    return CrossEncoderReranker(
        model_name=cfg.model_name, device=cfg.device, batch_size=cfg.batch_size
    )


def get_or_build_reranker(cfg: RerankingConfig) -> Reranker | None:
    """`None` when disabled; otherwise returns the SAME `CrossEncoderReranker`
    instance for a repeated identical config (process-wide cache), so
    building many `SearchService`s/apps against the same reranking config
    only loads the underlying model once."""

    if not cfg.enabled:
        return None
    key = cfg.model_dump_json()
    reranker = _RERANKER_CACHE.get(key)
    if reranker is None:
        reranker = build_reranker(cfg)
        _RERANKER_CACHE[key] = reranker
    return reranker
