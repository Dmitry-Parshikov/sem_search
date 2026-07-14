from __future__ import annotations

from app.config import VectorStoreConfig
from app.vector_store.base import VectorStore
from app.vector_store.qdrant_store import QdrantVectorStore


def build_vector_store(cfg: VectorStoreConfig) -> VectorStore:
    if cfg.backend != "qdrant":
        raise ValueError(f"Unknown vector store backend: {cfg.backend!r}")

    qdrant_cfg = cfg.qdrant
    if qdrant_cfg.mode == "embedded":
        return QdrantVectorStore(path=qdrant_cfg.path)
    return QdrantVectorStore(url=qdrant_cfg.url)
