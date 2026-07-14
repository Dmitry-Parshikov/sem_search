from __future__ import annotations

from app.config import EmbeddingConfig
from app.embedding.base import Embedder
from app.embedding.st_embedder import SentenceTransformerEmbedder

# Process-wide cache of built embedders, keyed by a JSON-serialized
# representation of the config. `functools.lru_cache` can't be used directly
# on a pydantic model argument (not hashable unless frozen), so we key on
# `cfg.model_dump_json()` instead. This makes `get_or_build_embedder` return
# the SAME instance for the same config across the app (Ф2.5: indexing and
# querying must share one embedder instance).
_EMBEDDER_CACHE: dict[str, Embedder] = {}


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        query_prefix=cfg.query_prefix,
        passage_prefix=cfg.passage_prefix,
    )


def get_or_build_embedder(cfg: EmbeddingConfig) -> Embedder:
    key = cfg.model_dump_json()
    embedder = _EMBEDDER_CACHE.get(key)
    if embedder is None:
        embedder = build_embedder(cfg)
        _EMBEDDER_CACHE[key] = embedder
    return embedder
