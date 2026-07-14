from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes_health, routes_index, routes_reindex, routes_search
from app.chunking.factory import build_chunker
from app.config import Settings
from app.embedding.factory import get_or_build_embedder
from app.indexing.service import IndexingService
from app.lexical.base import LexicalIndex
from app.lexical.factory import build_lexical_index
from app.preprocessing.loaders import TextPreprocessor
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
        app.state.search_service = SearchService(
            embedder=app.state.embedder,
            vector_store=app.state.vector_store,
            active_index_resolver=app.state.active_index_resolver,
            settings=settings,
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

    return app


app = create_app()
