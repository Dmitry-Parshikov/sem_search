"""Unit tests for `ActiveIndexResolver`'s single-slot cache: it must reuse
its cached `LexicalIndex` while the manifest's active version is unchanged,
and rebuild/reload when the active version changes (the mechanism that
makes reindex/rollback "just work" for search without a process restart).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.admin.manifest import IndexManifest
from app.config import AdminConfig, LexicalConfig, Settings
from app.core.errors import NoActiveIndexError
from app.lexical.bm25_index import BM25LexicalIndex
from app.search.active_index import ActiveIndexResolver


def make_settings(root: Path) -> Settings:
    return Settings(
        admin=AdminConfig(
            manifest_path=str(root / "manifest.json"),
            lexical_index_dir=str(root / "lexical"),
        ),
        lexical=LexicalConfig(use_lemmatization=False),
    )


def sample_entry(index_version: str, lexical_path: Path) -> dict:
    return {
        "index_version": index_version,
        "created_at": "2024-01-01T00:00:00+00:00",
        "embedding_model": "fake-embedder",
        "embedding_dimension": 16,
        "chunking_strategy": "paragraph",
        "chunking_config_signature": "paragraph|min_chars=0",
        "lexical_lemmatization": False,
        "bm25_params": {"k1": 1.5, "b": 0.75},
        "vector_collection_name": f"chunks__{index_version}",
        "lexical_index_path": str(lexical_path),
        "document_count": 1,
        "chunk_count": 1,
        "source_corpus": "test",
    }


def write_lexical_pickle(settings: Settings, index_version: str, text: str) -> Path:
    from app.core.types import Chunk

    lexical_index = BM25LexicalIndex(k1=1.5, b=0.75, tokenizer=_tokenizer(settings))
    lexical_index.build([Chunk(chunk_id=f"{index_version}::0", doc_id="d0", text=text, position=0)])
    path = Path(settings.admin.lexical_index_dir) / f"{index_version}.pkl"
    lexical_index.save(path)
    return path


def _tokenizer(settings: Settings):
    from app.lexical.tokenizer import RussianTokenizer

    return RussianTokenizer(use_lemmatization=settings.lexical.use_lemmatization)


def test_resolve_raises_when_no_active_version(tmp_path: Path):
    settings = make_settings(tmp_path)
    resolver = ActiveIndexResolver(settings)

    with pytest.raises(NoActiveIndexError):
        resolver.resolve()


def test_resolve_caches_lexical_index_across_calls_for_same_version(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    lexical_path = write_lexical_pickle(settings, "v1", "договор аренды помещения")

    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1", lexical_path))
    manifest.save(Path(settings.admin.manifest_path))

    load_calls = []
    original_load = BM25LexicalIndex.load

    def spy_load(self, path):
        load_calls.append(path)
        return original_load(self, path)

    monkeypatch.setattr(BM25LexicalIndex, "load", spy_load)

    resolver = ActiveIndexResolver(settings)
    context1 = resolver.resolve()
    context2 = resolver.resolve()

    assert context1.index_version == "v1"
    assert context1.collection_name == f"chunks__v1"
    assert context1.lexical_index is context2.lexical_index
    assert len(load_calls) == 1  # loaded once, reused on the second resolve()


def test_resolve_reloads_when_active_version_changes(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    lexical_path_v1 = write_lexical_pickle(settings, "v1", "договор аренды помещения")
    lexical_path_v2 = write_lexical_pickle(settings, "v2", "API и REST интеграция")

    manifest_path = Path(settings.admin.manifest_path)
    manifest = IndexManifest()
    manifest.record_new(sample_entry("v1", lexical_path_v1))
    manifest.save(manifest_path)

    load_calls = []
    original_load = BM25LexicalIndex.load

    def spy_load(self, path):
        load_calls.append(path)
        return original_load(self, path)

    monkeypatch.setattr(BM25LexicalIndex, "load", spy_load)

    resolver = ActiveIndexResolver(settings)
    context1 = resolver.resolve()
    assert context1.index_version == "v1"
    assert len(load_calls) == 1

    # Simulate a reindex/rollback: rewrite the manifest with a new active version.
    manifest = IndexManifest.load(manifest_path)
    manifest.record_new(sample_entry("v2", lexical_path_v2))
    manifest.save(manifest_path)

    context2 = resolver.resolve()
    assert context2.index_version == "v2"
    assert len(load_calls) == 2  # reloaded because the active version changed
    assert context2.lexical_index is not context1.lexical_index

    results = context2.lexical_index.search("REST", top_k=1)
    assert results and results[0].doc_id == "d0"
