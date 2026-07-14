"""FastAPI DI wiring point.

Singletons (embedder, vector store, chunker, indexing service, reranker,
hybridizer, ...) are built once in `app.main`'s lifespan and stashed on
`request.app.state`. These getters are the single place routes reach into
that state, so routes never import concrete implementations directly
(keeps the DI story real, not just an interface exercise).

Phase 3: `get_settings`, `get_embedder`, `get_vector_store`, and
`get_indexing_service` read real singletons built by the lifespan.

Phase 4: there is still no single fixed lexical-index singleton for the
app's lifetime -- the *query-time* lexical index depends on whichever
`index_version` is currently active (and changes on reindex/rollback) --
but rather than leaving that as a per-request `NotImplementedError`, the
lifespan now builds a single `ActiveIndexResolver` (`app.search.active_index`)
once at startup and keeps it on `app.state.active_index_resolver`. The
resolver itself re-checks the manifest and rebuilds its cached
`LexicalIndex` only when the active version actually changes, so search
always reflects the current active version without a process restart.

Phase 5: `hybridizer` is now a real singleton too (`app.hybrid.factory
.build_hybridizer`, built from `settings.hybridization`), stashed on
`app.state.hybridizer` in the lifespan. `reranker` remains unbuilt until
Phase 7.

Phase 6: `typo_corrector`/`term_expander` (`app.query.factory`) are built in
the lifespan and injected into `SearchService` directly -- these getters
exist mainly so tests can reach in and swap `app.state.typo_corrector` /
`app.state.term_expander` for a fake (e.g. to exercise the NFR
"Надёжность" degrade-on-failure path) the same way other singletons here
are reachable.
"""

from __future__ import annotations

from fastapi import Request

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


def get_reranker(request: Request) -> Reranker:
    if not hasattr(request.app.state, "reranker"):
        raise NotImplementedError("get_reranker: wired in Phase 7")
    return request.app.state.reranker


def get_hybridizer(request: Request) -> Hybridizer:
    return request.app.state.hybridizer


def get_typo_corrector(request: Request) -> TypoCorrector | None:
    return request.app.state.typo_corrector


def get_term_expander(request: Request) -> TermExpander | None:
    return request.app.state.term_expander
