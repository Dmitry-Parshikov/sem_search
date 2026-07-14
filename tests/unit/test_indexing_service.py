from __future__ import annotations

import hashlib
from pathlib import Path

from app.admin.manifest import IndexManifest
from app.chunking.paragraph import ParagraphChunker
from app.config import AdminConfig, QdrantConfig, Settings, VectorStoreConfig
from app.core.types import Document, Vector
from app.embedding.base import Embedder
from app.indexing.service import IndexingService
from app.lexical.factory import build_lexical_index
from app.preprocessing.loaders import TextPreprocessor
from app.vector_store.qdrant_store import QdrantVectorStore


class FakeEmbedder(Embedder):
    """Deterministic hash-based fake vectors -- no real ML model loaded,
    keeps this integration-ish test fast."""

    _DIM = 16

    def encode_documents(self, texts: list[str]) -> list[Vector]:
        return [self._vector_for(t) for t in texts]

    def encode_query(self, text: str) -> Vector:
        return self._vector_for(text)

    @property
    def dimension(self) -> int:
        return self._DIM

    @property
    def model_name(self) -> str:
        return "fake-embedder-v1"

    @staticmethod
    def _vector_for(text: str) -> Vector:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: FakeEmbedder._DIM]]


def make_settings(root: Path) -> Settings:
    return Settings(
        admin=AdminConfig(
            manifest_path=str(root / "manifest.json"),
            query_log_path=str(root / "logs" / "queries.jsonl"),
            lexical_index_dir=str(root / "lexical"),
        ),
        vector_store=VectorStoreConfig(
            qdrant=QdrantConfig(
                mode="embedded",
                path=str(root / "qdrant"),
                collection_name="test_chunks",
            )
        ),
    )


def make_documents() -> list[Document]:
    return [
        Document(doc_id="d1", text="Первый документ.\n\nВторой абзац первого документа.", source="test"),
        Document(doc_id="d2", text="Документ про API и REST интеграцию.", source="test"),
        Document(doc_id="d3", text="Юридический документ про договор аренды.", source="test"),
        Document(doc_id="d4", text="Новость дня: открылась выставка.", source="test"),
    ]


def build_service(root: Path) -> tuple[IndexingService, Settings, QdrantVectorStore]:
    settings = make_settings(root)
    embedder = FakeEmbedder()
    chunker = ParagraphChunker(min_chars=0)
    vector_store = QdrantVectorStore(path=settings.vector_store.qdrant.path)
    preprocessor = TextPreprocessor()

    def lexical_factory():
        return build_lexical_index(settings.lexical)

    service = IndexingService(
        embedder=embedder,
        chunker=chunker,
        vector_store=vector_store,
        preprocessor=preprocessor,
        lexical_index_factory=lexical_factory,
        settings=settings,
    )
    # Embedded Qdrant exclusively locks its storage folder per process, so
    # callers must reuse THIS vector_store instance for any post-run
    # assertions against the same path rather than opening a second client.
    return service, settings, vector_store


def test_run_creates_manifest_active_version(tmp_path: Path):
    service, settings, _vector_store = build_service(tmp_path)
    documents = make_documents()

    index_version = service.run(documents, source_corpus="unit-test-corpus")

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    assert manifest.active_version == index_version
    active = manifest.get_active()
    assert active["document_count"] == len(documents)
    assert active["chunk_count"] > 0
    assert active["source_corpus"] == "unit-test-corpus"
    assert active["status"] == "active"
    assert active["embedding_model"] == "fake-embedder-v1"
    assert active["chunking_strategy"] == settings.chunking.strategy


def test_run_creates_qdrant_collection_with_correct_point_count(tmp_path: Path):
    service, settings, vector_store = build_service(tmp_path)
    documents = make_documents()
    service.run(documents)

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    active = manifest.get_active()
    collection = active["vector_collection_name"]

    assert vector_store.collection_exists(collection) is True

    fake_query_vector = FakeEmbedder().encode_query("что угодно")
    results = vector_store.search(collection, fake_query_vector, top_k=1000)
    assert len(results) == active["chunk_count"]


def test_run_builds_lexical_index(tmp_path: Path):
    service, settings, _vector_store = build_service(tmp_path)
    documents = make_documents()
    service.run(documents)

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    active = manifest.get_active()
    lexical_path = Path(active["lexical_index_path"])
    assert lexical_path.exists()

    loaded_index = build_lexical_index(settings.lexical)
    loaded_index.load(lexical_path)
    results = loaded_index.search("договор аренды", top_k=3)
    assert results
    assert any(r.doc_id == "d3" for r in results)


def test_reindex_determinism_same_chunk_ids_and_vectors(tmp_path: Path):
    """Two runs of the same corpus/config produce an identical chunk_id set
    and identical stored vectors (previews acceptance criterion 5; the full
    real-model version comes in Phase 10)."""

    documents = make_documents()

    service1, settings1, vs1 = build_service(tmp_path / "run1")
    version1 = service1.run(documents)
    active1 = IndexManifest.load(Path(settings1.admin.manifest_path)).get_active()

    service2, settings2, vs2 = build_service(tmp_path / "run2")
    version2 = service2.run(documents)
    active2 = IndexManifest.load(Path(settings2.admin.manifest_path)).get_active()

    assert active1["chunk_count"] == active2["chunk_count"]

    # Hash suffix of index_version depends only on model+chunking signature,
    # not on wall-clock timestamp, so it must match across the two runs.
    assert version1.split("_")[-1] == version2.split("_")[-1]

    fake_query_vector = FakeEmbedder().encode_query("любой запрос")
    results1 = sorted(
        vs1.search(active1["vector_collection_name"], fake_query_vector, top_k=1000),
        key=lambda r: r.chunk_id,
    )
    results2 = sorted(
        vs2.search(active2["vector_collection_name"], fake_query_vector, top_k=1000),
        key=lambda r: r.chunk_id,
    )

    assert [r.chunk_id for r in results1] == [r.chunk_id for r in results2]
    assert [r.doc_id for r in results1] == [r.doc_id for r in results2]
    assert [r.text for r in results1] == [r.text for r in results2]

    # Deeper check: retrieve the raw stored vectors (not just ranking) via
    # the underlying qdrant client, to confirm they are bit-for-bit
    # identical -- expected here since FakeEmbedder is a pure function of
    # chunk text and chunking is deterministic.
    ids1 = [r.chunk_id for r in results1]
    points1 = vs1._client.retrieve(  # noqa: SLF001 - test-only introspection
        collection_name=active1["vector_collection_name"],
        ids=[_point_id(cid) for cid in ids1],
        with_vectors=True,
    )
    points2 = vs2._client.retrieve(  # noqa: SLF001 - test-only introspection
        collection_name=active2["vector_collection_name"],
        ids=[_point_id(cid) for cid in ids1],
        with_vectors=True,
    )
    vectors1 = {p.payload["chunk_id"]: p.vector for p in points1}
    vectors2 = {p.payload["chunk_id"]: p.vector for p in points2}
    assert vectors1 == vectors2


def _point_id(chunk_id: str) -> str:
    from app.vector_store.qdrant_store import _point_id as impl

    return impl(chunk_id)
