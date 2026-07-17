from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.admin.query_log import QueryLogger
from app.admin.service import AdminService
from app.api import routes_admin, routes_folder_index, routes_health, routes_index, routes_reindex, routes_search
from app.chunking.factory import build_chunker
from app.config import Settings
from app.embedding.factory import get_or_build_embedder
from app.hybrid.factory import build_hybridizer
from app.indexing.service import IndexingService
from app.lexical.base import LexicalIndex
from app.lexical.factory import build_lexical_index
from app.preprocessing.loaders import TextPreprocessor
from app.query.factory import build_query_expander, build_typo_corrector
from app.rerank.factory import get_or_build_reranker
from app.search.active_index import ActiveIndexResolver
from app.search.service import SearchService
from app.vector_store.factory import build_vector_store


def _make_lifespan(settings_override: Settings | None) -> Callable[[FastAPI], AsyncIterator[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # `SEM_SEARCH_CONFIG` selects the config profile (e.g. the
        # resource-saving profile `config/config.resource_saving.yaml`)
        # without code changes; defaults to the main `config/config.yaml`.
        config_path = os.environ.get("SEM_SEARCH_CONFIG", "config/config.yaml")
        settings = settings_override or Settings.load(config_path)
        app.state.settings = settings

        # Built once per process: embedder (Ф2.5 — same instance for
        # indexing and querying), vector store, chunker. The lexical index is
        # NOT a singleton here — `IndexingService` needs a *fresh* one per
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

        # Active-index resolver: caches the current version's LexicalIndex
        # across requests, rebuilt only when the active version changes.
        app.state.active_index_resolver = ActiveIndexResolver(settings)

        # Hybridizer (RRF or weighted, per config) — stateless singleton.
        app.state.hybridizer = build_hybridizer(settings.hybridization)

        # Typo corrector / term expander: either a real singleton or None when
        # disabled via config. `SearchService` skips a stage entirely when its
        # collaborator is None.
        app.state.typo_corrector = build_typo_corrector(settings.query_processing.typo_correction)
        app.state.term_expander = build_query_expander(settings.query_processing)

        # Reranker (cross-encoder, Ф3.5): `None` when `reranking.enabled` is
        # False. When enabled, a `.rerank()` exception is caught inside
        # `SearchService` (NFR "Надёжность"). `get_or_build_reranker` shares
        # one loaded model across repeated app instances (e.g. test apps).
        app.state.reranker = get_or_build_reranker(settings.reranking)

        # Query audit logging (Ф4.2) and admin service (Ф4.1).
        # `AdminService` reads the manifest fresh from disk on every call.
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
    app.include_router(routes_folder_index.router)

    _TEMPLATE = Path(__file__).resolve().parent / "templates" / "search.html"
    _HTML = _TEMPLATE.read_text(encoding="utf-8") if _TEMPLATE.exists() else "<h1>sem_search</h1>"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return _HTML

    return app


app = create_app()
