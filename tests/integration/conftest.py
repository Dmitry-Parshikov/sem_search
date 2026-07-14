from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import AdminConfig, AppMeta, QdrantConfig, Settings, VectorStoreConfig
from app.main import create_app


def _build_settings(root: Path, collection_name: str) -> Settings:
    """Settings rooted entirely under a temp dir -- manifest, lexical
    pickles, embedded Qdrant storage, and (via `app.data_dir`) the
    persisted-corpus JSON used by `/reindex` -- so integration tests never
    touch the real `data/` dir."""

    return Settings(
        app=AppMeta(data_dir=str(root)),
        admin=AdminConfig(
            manifest_path=str(root / "index_manifest.json"),
            query_log_path=str(root / "logs" / "queries.jsonl"),
            lexical_index_dir=str(root / "lexical"),
        ),
        vector_store=VectorStoreConfig(
            qdrant=QdrantConfig(
                mode="embedded",
                path=str(root / "qdrant"),
                collection_name=collection_name,
            )
        ),
    )


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped `TestClient` running against a temp data dir.

    Session-scoped so the real dev ST embedder (`intfloat/multilingual-e5-small`)
    loads exactly once for the whole integration test file rather than once
    per test -- loading it is the slow part, not building an (embedded,
    file-backed) Qdrant store or a chunker.
    """

    root = tmp_path_factory.mktemp("sem_search_integration")
    settings = _build_settings(root, collection_name="integration_chunks")
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def fresh_client(tmp_path: Path):
    """A separate app/client per test, pointed at its own empty temp dir --
    for scenarios that need to observe "nothing indexed yet" state, which
    the shared session-scoped `client` fixture can no longer provide once
    earlier tests have indexed into it.

    Reuses the embedder singleton cache (keyed by config) from `client`, so
    this does NOT reload the ST model -- only a fresh (empty) embedded
    Qdrant path and a fresh manifest are actually new here.
    """

    root = tmp_path
    settings = _build_settings(root, collection_name="fresh_chunks")
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client
