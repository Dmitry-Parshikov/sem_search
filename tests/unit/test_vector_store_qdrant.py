from __future__ import annotations

from pathlib import Path

from app.core.types import Chunk
from app.vector_store.qdrant_store import QdrantVectorStore

DIM = 4


def make_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="d1::0000", doc_id="d1", text="про кошек", position=0, metadata={"topic": "animals", "source": "s1"}),
        Chunk(chunk_id="d2::0000", doc_id="d2", text="про собак", position=0, metadata={"topic": "animals", "source": "s2"}),
        Chunk(chunk_id="d3::0000", doc_id="d3", text="про право", position=0, metadata={"topic": "law", "source": "s1"}),
    ]


def make_vectors() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]


def test_create_upsert_search_round_trip(tmp_path: Path):
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    store.create_collection("test_coll", dimension=DIM)
    assert store.collection_exists("test_coll") is True

    store.upsert("test_coll", make_chunks(), make_vectors())

    results = store.search("test_coll", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "d1::0000"
    assert results[0].doc_id == "d1"
    assert results[0].text == "про кошек"
    assert results[0].metadata["topic"] == "animals"


def test_create_collection_idempotent(tmp_path: Path):
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    store.create_collection("coll", dimension=DIM)
    # Calling again must not raise.
    store.create_collection("coll", dimension=DIM)
    assert store.collection_exists("coll") is True


def test_metadata_filter_narrows_results(tmp_path: Path):
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    store.create_collection("test_coll", dimension=DIM)
    store.upsert("test_coll", make_chunks(), make_vectors())

    results = store.search(
        "test_coll",
        query_vector=[0.5, 0.0, 0.5, 0.0],
        top_k=10,
        metadata_filter={"topic": "animals"},
    )
    assert len(results) == 2
    assert {r.doc_id for r in results} == {"d1", "d2"}

    results_law = store.search(
        "test_coll",
        query_vector=[0.5, 0.0, 0.5, 0.0],
        top_k=10,
        metadata_filter={"topic": "law"},
    )
    assert len(results_law) == 1
    assert results_law[0].doc_id == "d3"


def test_delete_collection(tmp_path: Path):
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    store.create_collection("coll", dimension=DIM)
    assert store.collection_exists("coll") is True
    store.delete_collection("coll")
    assert store.collection_exists("coll") is False
    # Deleting again (non-existent) must not raise.
    store.delete_collection("coll")


def test_health_true_when_reachable(tmp_path: Path):
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    assert store.health() is True


def test_requires_exactly_one_of_path_or_url():
    import pytest

    with pytest.raises(ValueError):
        QdrantVectorStore()
    with pytest.raises(ValueError):
        QdrantVectorStore(path="a", url="b")
