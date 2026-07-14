from __future__ import annotations

from pathlib import Path

from app.admin.manifest import IndexManifest


def sample_entry(index_version: str) -> dict:
    return {
        "index_version": index_version,
        "created_at": "2024-01-01T00:00:00+00:00",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "chunking_strategy": "paragraph",
        "chunking_config_signature": "paragraph|min_chars=0",
        "lexical_lemmatization": True,
        "bm25_params": {"k1": 1.5, "b": 0.75},
        "vector_collection_name": f"sem_search_chunks__{index_version}",
        "lexical_index_path": f"./data/lexical/{index_version}.pkl",
        "document_count": 3,
        "chunk_count": 5,
        "source_corpus": "test",
    }


def test_load_missing_file_returns_empty_manifest(tmp_path: Path):
    manifest = IndexManifest.load(tmp_path / "does_not_exist.json")
    assert manifest.active_version is None
    assert manifest.list_versions() == []
    assert manifest.get_active() is None


def test_record_new_sets_active_and_appends(tmp_path: Path):
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1"))

    assert manifest.active_version == "v1"
    assert len(manifest.list_versions()) == 1
    assert manifest.get_active()["index_version"] == "v1"
    assert manifest.get_active()["status"] == "active"


def test_record_new_supersedes_previous_active(tmp_path: Path):
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1"))
    manifest.record_new(sample_entry("v2"))

    versions = {v["index_version"]: v for v in manifest.list_versions()}
    assert versions["v1"]["status"] == "superseded"
    assert versions["v2"]["status"] == "active"
    assert manifest.active_version == "v2"


def test_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "manifest.json"
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1"))
    manifest.record_new(sample_entry("v2"))
    manifest.save(path)

    loaded = IndexManifest.load(path)
    assert loaded.active_version == "v2"
    assert len(loaded.list_versions()) == 2
    assert loaded.get_active()["index_version"] == "v2"


def test_rollback_pointer_update(tmp_path: Path):
    """Simulates a rollback: manually flip active_version back to an older
    entry to verify the manifest data layer supports the pointer-swap
    semantics used by the admin rollback endpoint."""

    path = tmp_path / "manifest.json"
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1"))
    manifest.record_new(sample_entry("v2"))
    manifest.save(path)

    reloaded = IndexManifest.load(path)
    reloaded.active_version = "v1"
    reloaded.save(path)

    final = IndexManifest.load(path)
    assert final.active_version == "v1"
    # Statuses in the persisted version entries are untouched by a pointer
    # swap alone (rollback logic that also flips `status` fields belongs to
    # admin service, not this data layer).
    assert len(final.list_versions()) == 2


def test_atomic_write_leaves_old_file_intact_on_repeated_save(tmp_path: Path):
    """The temp-file-then-os.replace pattern must not corrupt the target
    file across repeated saves (simulating a second save after a first
    completed, verifying no partial/corrupt state is left behind)."""

    path = tmp_path / "manifest.json"
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1"))
    manifest.save(path)
    first_content = path.read_text(encoding="utf-8")
    assert "v1" in first_content

    manifest.record_new(sample_entry("v2"))
    manifest.save(path)
    second_content = path.read_text(encoding="utf-8")
    assert "v1" in second_content and "v2" in second_content

    # No leftover .tmp file after a successful save.
    assert not path.with_suffix(".tmp").exists()

    # File is valid JSON at every point (not truncated/partial).
    reloaded = IndexManifest.load(path)
    assert len(reloaded.list_versions()) == 2
