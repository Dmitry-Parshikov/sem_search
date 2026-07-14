"""FastAPI DI wiring point.

Singletons (embedder, vector store, chunker, indexing service, reranker,
hybridizer, ...) are built once in `app.main`'s lifespan and stashed on
`request.app.state`. These getters are the single place routes reach into
that state, so routes never import concrete implementations directly.

The lifespan builds:
- `ActiveIndexResolver` once at startup — it re-checks the manifest and
  rebuilds its cached `LexicalIndex` only when the active version changes.
- `hybridizer` (stateless, per config) as a process-lifetime singleton.
- `typo_corrector` / `term_expander` — real instances, or `None` when
  disabled via config.
- `reranker` — `None` when `reranking.enabled` is False, otherwise a real
  `CrossEncoderReranker`.
- `admin_service` and `query_logger` — backing `/admin/*` routes and
  structured search-audit logging (Ф4.2).
"""

from __future__ import annotations

from fastapi import Request

from app.admin.query_log import QueryLogger
from app.admin.service import AdminService
from app.config import Settings
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.indexing.service import IndexingService
from app.query.base import TermExpander, TypoCorrector
from app.rerank.base import Reranker
from app.search.active_index import ActiveIndexResolver
from app.search.service import SearchService
from app.vector_store.base import VectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_indexing_service(request: Request) -> IndexingService:
    return request.app.state.indexing_service


def get_active_index_resolver(request: Request) -> ActiveIndexResolver:
    return request.app.state.active_index_resolver


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_reranker(request: Request) -> Reranker | None:
    return request.app.state.reranker


def get_hybridizer(request: Request) -> Hybridizer:
    return request.app.state.hybridizer


def get_typo_corrector(request: Request) -> TypoCorrector | None:
    return request.app.state.typo_corrector


def get_term_expander(request: Request) -> TermExpander | None:
    return request.app.state.term_expander


def get_admin_service(request: Request) -> AdminService:
    return request.app.state.admin_service


def get_query_logger(request: Request) -> QueryLogger:
    return request.app.state.query_logger
