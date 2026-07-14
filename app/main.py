from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.admin.query_log import QueryLogger
from app.admin.service import AdminService
from app.api import routes_admin, routes_health, routes_index, routes_reindex, routes_search
from app.chunking.factory import build_chunker
from app.config import Settings
from app.embedding.factory import get_or_build_embedder
from app.hybrid.factory import build_hybridizer
from app.indexing.service import IndexingService
from app.lexical.base import LexicalIndex
from app.lexical.factory import build_lexical_index
from app.preprocessing.loaders import TextPreprocessor
from app.query.factory import build_term_expander, build_typo_corrector
from app.rerank.factory import get_or_build_reranker
from app.search.active_index import ActiveIndexResolver
from app.search.service import SearchService
from app.vector_store.factory import build_vector_store


def _make_lifespan(settings_override: Settings | None) -> Callable[[FastAPI], AsyncIterator[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_override or Settings.load()
        app.state.settings = settings

        # Built once per process: embedder (Ф2.5 -- same instance for
        # indexing and querying), vector store, chunker. The lexical index is
        # NOT a singleton here -- `IndexingService` needs a *fresh* one per
        # run (see `app.lexical.factory.build_lexical_index`'s docstring),
        # so only the factory is stashed/passed, not a built instance.
        app.state.embedder = get_or_build_embedder(settings.embedding)
        app.state.vector_store = build_vector_store(settings.vector_store)
        app.state.chunker = build_chunker(settings.chunking)

        def lexical_index_factory() -> LexicalIndex:
            return build_lexical_index(settings.lexical)

        app.state.indexing_service = IndexingService(
            embedder=app.state.embedder,
            chunker=app.state.chunker,
            vector_store=app.state.vector_store,
            preprocessor=TextPreprocessor(),
            lexical_index_factory=lexical_index_factory,
            settings=settings,
        )

        # Phase 4: the active-index resolver is a long-lived singleton (it
        # caches the currently-active version's LexicalIndex across
        # requests) built once here, not per-request -- see
        # `app.search.active_index.ActiveIndexResolver`'s docstring.
        app.state.active_index_resolver = ActiveIndexResolver(settings)

        # Phase 5: the hybridizer (RRF or weighted, per config) is stateless
        # and cheap to build, so a single process-lifetime singleton is fine.
        app.state.hybridizer = build_hybridizer(settings.hybridization)

        # Phase 6: typo corrector / term expander are each either a real,
        # stateless (cheap-to-build) singleton, or None when disabled via
        # config -- `SearchService` skips a stage entirely when its
        # collaborator is None instead of calling a forced no-op.
        app.state.typo_corrector = build_typo_corrector(settings.query_processing.typo_correction)
        app.state.term_expander = build_term_expander(settings.query_processing.term_expansion)

        # Phase 7: the reranker (cross-encoder, Ф3.5) is `None` when
        # `reranking.enabled` is False in config -- `SearchService` then
        # treats `hybrid_rerank` identically to `hybrid` (configuration, not
        # a failure). When enabled, a `.rerank()` exception at request time
        # is still caught inside `SearchService` (NFR "Надёжность").
        # `get_or_build_reranker` (not the plain `build_reranker`) is used
        # here so repeated app instances built against the same reranking
        # config (e.g. many short-lived test apps) share one loaded model
        # instead of reloading it from disk each time -- see
        # `app.rerank.factory`'s docstring.
        app.state.reranker = get_or_build_reranker(settings.reranking)

        # Phase 8: query audit logging (Ф4.2, always-on) and the admin
        # service backing `/admin/versions` + `/admin/rollback/{version}`
        # (Ф4.1). `AdminService` reads the manifest fresh from disk on every
        # call (see its docstring), so it needs no cached state of its own
        # beyond the vector store singleton + settings.
        app.state.query_logger = QueryLogger(Path(settings.admin.query_log_path))
        app.state.admin_service = AdminService(vector_store=app.state.vector_store, settings=settings)

        app.state.search_service = SearchService(
            embedder=app.state.embedder,
            vector_store=app.state.vector_store,
            active_index_resolver=app.state.active_index_resolver,
            hybridizer=app.state.hybridizer,
            settings=settings,
            query_logger=app.state.query_logger,
            typo_corrector=app.state.typo_corrector,
            term_expander=app.state.term_expander,
            reranker=app.state.reranker,
        )

        yield

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Builds the FastAPI app. `settings` is an optional injection point for
    tests (so they can point `admin.manifest_path` / `vector_store.qdrant.path`
    / `admin.lexical_index_dir` at a temp dir instead of monkeypatching
    `Settings.load`) -- defaults to `Settings.load()` when omitted, so
    existing callers (`app = create_app()` below, `uvicorn app.main:app`)
    are unaffected.
    """

    app = FastAPI(title="sem_search", lifespan=_make_lifespan(settings))

    app.include_router(routes_index.router)
    app.include_router(routes_reindex.router)
    app.include_router(routes_health.router)
    app.include_router(routes_search.router)
    app.include_router(routes_admin.router)

    return app


app = create_app()
