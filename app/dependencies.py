"""FastAPI DI wiring point.

Singletons (embedder, vector store, lexical index, reranker, hybridizer, ...)
are built once in `app.main`'s lifespan and stashed on `request.app.state`.
These getters are the single place routes reach into that state, so routes
never import concrete implementations directly (keeps the DI story real,
not just an interface exercise).

Phase 1: only `get_settings` has something to read from `app.state` (the
lifespan already sets `app.state.settings`). Everything else is not built
yet -- those getters raise `NotImplementedError` naming the phase that wires
them, rather than pretending to work.
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.lexical.base import LexicalIndex
from app.rerank.base import Reranker
from app.vector_store.base import VectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_embedder(request: Request) -> Embedder:
    if not hasattr(request.app.state, "embedder"):
        raise NotImplementedError("get_embedder: wired in Phase 2")
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    if not hasattr(request.app.state, "vector_store"):
        raise NotImplementedError("get_vector_store: wired in Phase 2")
    return request.app.state.vector_store


def get_lexical_index(request: Request) -> LexicalIndex:
    if not hasattr(request.app.state, "lexical_index"):
        raise NotImplementedError("get_lexical_index: wired in Phase 2")
    return request.app.state.lexical_index


def get_reranker(request: Request) -> Reranker:
    if not hasattr(request.app.state, "reranker"):
        raise NotImplementedError("get_reranker: wired in Phase 7")
    return request.app.state.reranker


def get_hybridizer(request: Request) -> Hybridizer:
    if not hasattr(request.app.state, "hybridizer"):
        raise NotImplementedError("get_hybridizer: wired in Phase 5")
    return request.app.state.hybridizer
