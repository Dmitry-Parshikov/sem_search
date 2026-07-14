"""FastAPI DI wiring point.

Singletons (embedder, vector store, chunker, indexing service, reranker,
hybridizer, ...) are built once in `app.main`'s lifespan and stashed on
`request.app.state`. These getters are the single place routes reach into
that state, so routes never import concrete implementations directly
(keeps the DI story real, not just an interface exercise).

Phase 3: `get_settings`, `get_embedder`, `get_vector_store`, and
`get_indexing_service` read real singletons built by the lifespan.
`get_lexical_index` stays unwired here on purpose: unlike embedder/vector
store, there is no single fixed lexical-index singleton for the app's
lifetime -- the *query-time* lexical index depends on whichever
`index_version` is currently active (and changes on reindex/rollback), so
loading it is a per-request concern for Phase 4's search routes, not
something the lifespan can pin down once at startup. `reranker`/
`hybridizer` remain unbuilt until Phases 7/5.
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.indexing.service import IndexingService
from app.lexical.base import LexicalIndex
from app.rerank.base import Reranker
from app.vector_store.base import VectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_indexing_service(request: Request) -> IndexingService:
    return request.app.state.indexing_service


def get_lexical_index(request: Request) -> LexicalIndex:
    if not hasattr(request.app.state, "lexical_index"):
        raise NotImplementedError("get_lexical_index: wired in Phase 4 (per active index_version)")
    return request.app.state.lexical_index


def get_reranker(request: Request) -> Reranker:
    if not hasattr(request.app.state, "reranker"):
        raise NotImplementedError("get_reranker: wired in Phase 7")
    return request.app.state.reranker


def get_hybridizer(request: Request) -> Hybridizer:
    if not hasattr(request.app.state, "hybridizer"):
        raise NotImplementedError("get_hybridizer: wired in Phase 5")
    return request.app.state.hybridizer
