"""Integration tests for `dense_rerank` mode: dense retrieval followed by
cross-encoder reranking WITHOUT any BM25/fusion step. Mirrors
`test_search_hybrid_rerank.py` (reranking must change the top order and must
degrade gracefully when the reranker fails).

Runs against a real embedded Qdrant + the real dev ST embedder + the real dev
cross-encoder, so it's marked slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import AdminConfig, AppMeta, QdrantConfig, Settings, VectorStoreConfig
from app.main import create_app
from app.rerank.base import Reranker

pytestmark = pytest.mark.slow

QUERY = "как расторгнуть договор аренды квартиры"

DOCS = [
    {
        "doc_id": "dr1",
        "text": (
            "Договор аренды квартиры заключается в письменной форме между "
            "арендодателем и арендатором; в договоре аренды указываются срок "
            "аренды, размер арендной платы и порядок передачи квартиры."
        ),
        "source": "test",
    },
    {
        "doc_id": "dr2",
        "text": (
            "Наниматель вправе прекратить пользование жилым помещением по "
            "соглашению сторон, направив письменное уведомление за один "
            "месяц до планируемого выезда."
        ),
        "source": "test",
    },
    {
        "doc_id": "dr3",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "dr4",
        "text": "API — программный интерфейс, позволяющий приложениям обмениваться данными.",
        "source": "test",
    },
]


def _build_settings(root: Path) -> Settings:
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
                collection_name="dr_chunks",
            )
        ),
    )


@pytest.fixture(scope="module")
def dr_client(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("sem_search_dense_rerank")
    settings = _build_settings(root)
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def _index_dr_docs(dr_client: TestClient):
    response = dr_client.post("/index", json={"documents": DOCS, "source_corpus": "dr-demo"})
    assert response.status_code == 200
    return response.json()


def test_dense_rerank_mode_returns_valid_nonempty_hits(dr_client, _index_dr_docs):
    response = dr_client.post("/search", json={"query": QUERY, "mode": "dense_rerank"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "dense_rerank"
    assert body["hits"]
    assert body["index_version"] == _index_dr_docs["index_version"]
    for hit in body["hits"]:
        assert set(hit.keys()) >= {"chunk_id", "doc_id", "text", "score", "metadata", "highlights"}


def test_dense_rerank_changes_top_order_relative_to_dense(dr_client, _index_dr_docs):
    """Reranking must actually change the ordering produced by plain `dense`
    retrieval, not just relabel the mode."""

    dense_response = dr_client.post("/search", json={"query": QUERY, "mode": "dense", "top_k": 4})
    rerank_response = dr_client.post(
        "/search", json={"query": QUERY, "mode": "dense_rerank", "top_k": 4}
    )

    assert dense_response.status_code == 200
    assert rerank_response.status_code == 200

    dense_order = [hit["chunk_id"] for hit in dense_response.json()["hits"]]
    rerank_order = [hit["chunk_id"] for hit in rerank_response.json()["hits"]]

    assert dense_order, "expected non-empty dense hits to compare against"
    assert rerank_order != dense_order


def test_dense_rerank_failure_degrades_gracefully_to_dense_ordering(dr_client, _index_dr_docs):
    """NFR "Надёжность": a broken reranker must not fail the request --
    `/search` with `mode=dense_rerank` still returns 200 with valid hits
    (falling back to the pre-rerank dense order), and the failure is surfaced
    as a non-empty `warnings` entry."""

    class RaisingReranker(Reranker):
        def rerank(self, query, candidates, top_n):
            raise RuntimeError("boom: reranker backend unavailable")

        @property
        def model_name(self) -> str:
            return "raising-fake"

    fake = RaisingReranker()
    dr_client.app.state.reranker = fake
    dr_client.app.state.search_service._reranker = fake

    response = dr_client.post("/search", json={"query": QUERY, "mode": "dense_rerank"})

    assert response.status_code == 200
    body = response.json()
    assert body["hits"]
    assert body["warnings"]
    assert any("rerank" in w.lower() for w in body["warnings"])
