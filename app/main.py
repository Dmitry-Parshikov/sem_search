from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Later phases build embedder/vector_store/lexical_index/reranker/hybridizer
    # singletons here and stash them on app.state (see app/dependencies.py).
    app.state.settings = Settings.load()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="sem_search", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        # Filled in properly (subsystem statuses, real index_version) in Phase 3/8.
        return {"status": "ok", "index_version": None}

    return app


app = create_app()
