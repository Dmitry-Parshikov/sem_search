"""Unit tests for `AdminService.rollback`/`list_versions`/`get_active`
(Ф4.1, plan decision #5): rollback is a manifest pointer-swap after
validating the target version's Qdrant collection + BM25 pickle still exist
-- no rebuild, no destructive Qdrant/file operations.

Uses a tiny fake `VectorStore` (following the same local-fake convention as
`tests/unit/test_search_retrievers.py` -- there is no shared `tests/fakes.py`
in this codebase yet) so this stays a pure unit test, independent of a real
Qdrant instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.admin.manifest import IndexManifest
from app.admin.service import AdminService
from app.config import AdminConfig, Settings
from app.core.errors import IndexVersionAssetsMissingError, IndexVersionNotFoundError
from app.core.types import RetrievedCandidate, Vector
from app.vector_store.base import VectorStore


class FakeVectorStore(VectorStore):
    """Reports collection existence from a settable set, nothing else used."""

    def __init__(self, existing_collections: set[str]) -> None:
        self._existing = set(existing_collections)

    def create_collection(self, name: str, dimension: int, distance: str = "cosine") -> None:
        raise NotImplementedError

    def upsert(self, collection: str, chunks, vectors) -> None:
        raise NotImplementedError

    def search(
        self,
        collection: str,
        query_vector: Vector,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        raise NotImplementedError

    def delete_collection(self, name: str) -> None:
        raise NotImplementedError

    def collection_exists(self, name: str) -> bool:
        return name in self._existing

    def health(self) -> bool:
        return True


def make_entry(index_version: str, lexical_path: Path, status: str) -> dict[str, Any]:
    return {
        "index_version": index_version,
        "created_at": "2024-01-01T00:00:00+00:00",
        "embedding_model": "fake-embedder",
        "embedding_dimension": 16,
        "chunking_strategy": "paragraph",
        "chunking_config_signature": "paragraph|min_chars=0",
        "lexical_lemmatization": True,
        "bm25_params": {"k1": 1.5, "b": 0.75},
        "vector_collection_name": f"chunks__{index_version}",
        "lexical_index_path": str(lexical_path),
        "document_count": 3,
        "chunk_count": 5,
        "source_corpus": "test",
        "status": status,
    }


def _settings(root: Path) -> Settings:
    return Settings(admin=AdminConfig(manifest_path=str(root / "manifest.json")))


def _build_two_version_manifest(tmp_path: Path) -> tuple[Settings, str, str, Path, Path]:
    """Builds a manifest with 2 versions the way `record_new` would leave it
    after two indexing runs: v1 superseded, v2 active. Both versions' lexical
    pickle files actually exist on disk (empty placeholder files are enough
    -- only existence is checked)."""

    settings = _settings(tmp_path)

    lexical_v1 = tmp_path / "lexical" / "v1.pkl"
    lexical_v2 = tmp_path / "lexical" / "v2.pkl"
    lexical_v1.parent.mkdir(parents=True, exist_ok=True)
    lexical_v1.write_bytes(b"fake-pickle-v1")
    lexical_v2.write_bytes(b"fake-pickle-v2")

    manifest = IndexManifest()
    manifest.record_new(make_entry("v1", lexical_v1, status="active"))
    manifest.record_new(make_entry("v2", lexical_v2, status="active"))
    manifest.save(Path(settings.admin.manifest_path))

    return settings, "v1", "v2", lexical_v1, lexical_v2


def test_list_versions_and_get_active(tmp_path: Path):
    settings, v1, v2, _lex1, _lex2 = _build_two_version_manifest(tmp_path)
    vector_store = FakeVectorStore(existing_collections={f"chunks__{v1}", f"chunks__{v2}"})
    service = AdminService(vector_store=vector_store, settings=settings)

    versions = service.list_versions()
    assert {v["index_version"] for v in versions} == {v1, v2}

    active = service.get_active()
    assert active is not None
    assert active["index_version"] == v2


def test_list_versions_empty_when_nothing_indexed(tmp_path: Path):
    settings = _settings(tmp_path)
    vector_store = FakeVectorStore(existing_collections=set())
    service = AdminService(vector_store=vector_store, settings=settings)

    assert service.list_versions() == []
    assert service.get_active() is None


def test_rollback_success_flips_status_and_active_version(tmp_path: Path):
    settings, v1, v2, _lex1, _lex2 = _build_two_version_manifest(tmp_path)
    vector_store = FakeVectorStore(existing_collections={f"chunks__{v1}", f"chunks__{v2}"})
    service = AdminService(vector_store=vector_store, settings=settings)

    result = service.rollback(v1)

    assert result["index_version"] == v1
    assert result["status"] == "active"

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    assert manifest.active_version == v1
    versions = {v["index_version"]: v for v in manifest.list_versions()}
    assert versions[v1]["status"] == "active"
    assert versions[v2]["status"] == "superseded"


def test_rollback_unknown_version_raises_not_found(tmp_path: Path):
    settings, _v1, _v2, _lex1, _lex2 = _build_two_version_manifest(tmp_path)
    vector_store = FakeVectorStore(existing_collections=set())
    service = AdminService(vector_store=vector_store, settings=settings)

    with pytest.raises(IndexVersionNotFoundError):
        service.rollback("does-not-exist")


def test_rollback_raises_when_collection_missing(tmp_path: Path):
    settings, v1, v2, _lex1, _lex2 = _build_two_version_manifest(tmp_path)
    # v1's collection does NOT exist in the fake store -- rollback must fail.
    vector_store = FakeVectorStore(existing_collections={f"chunks__{v2}"})
    service = AdminService(vector_store=vector_store, settings=settings)

    with pytest.raises(IndexVersionAssetsMissingError):
        service.rollback(v1)

    # Manifest must be left untouched (still v2 active) after a failed rollback.
    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    assert manifest.active_version == v2


def test_rollback_raises_when_lexical_path_missing(tmp_path: Path):
    settings, v1, v2, lexical_v1, _lex2 = _build_two_version_manifest(tmp_path)
    vector_store = FakeVectorStore(existing_collections={f"chunks__{v1}", f"chunks__{v2}"})
    service = AdminService(vector_store=vector_store, settings=settings)

    # Simulate the lexical pickle having been deleted from disk.
    lexical_v1.unlink()

    with pytest.raises(IndexVersionAssetsMissingError):
        service.rollback(v1)

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    assert manifest.active_version == v2
