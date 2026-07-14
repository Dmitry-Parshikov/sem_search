"""Integration tests for `hybrid_rerank` mode: acceptance criterion #2
("при hybrid_rerank итоговый порядок отличается от hybrid за счёт
реранжирования топ-N") plus the NFR "Надёжность" degrade-gracefully contract
for the reranker specifically (mirrors the typo/expansion degradation tests
in `test_search_query_processing.py`).

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

# hr1/hr2 create a deliberate hybrid-vs-cross-encoder disagreement for the
# query below:
#  - hr1 repeats the query's literal words ("договор", "аренды") heavily in a
#    context that is adjacent but NOT about ending a lease (about SIGNING
#    one) -- a strong BM25/RRF surface-overlap winner.
#  - hr2 is semantically exactly what the query asks about (ending/
#    terminating a lease) but phrased with synonyms ("наниматель",
#    "прекратить пользование", "уведомление за месяц") that share almost no
#    literal tokens with the query -- a weak BM25 candidate but the
#    genuinely on-topic one a cross-encoder (scoring query+doc jointly)
#    should be able to recognize.
# Filler docs (hr3/hr4) are unrelated topics, padding the pool.
QUERY = "как расторгнуть договор аренды квартиры"

DOCS = [
    {
        "doc_id": "hr1",
        "text": (
            "Договор аренды квартиры заключается в письменной форме между "
            "арендодателем и арендатором; в договоре аренды указываются срок "
            "аренды, размер арендной платы и порядок передачи квартиры."
        ),
        "source": "test",
    },
    {
        "doc_id": "hr2",
        "text": (
            "Наниматель вправе прекратить пользование жилым помещением по "
            "соглашению сторон, направив письменное уведомление за один "
            "месяц до планируемого выезда."
        ),
        "source": "test",
    },
    {
        "doc_id": "hr3",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "hr4",
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
                collection_name="hr_chunks",
            )
        ),
    )


@pytest.fixture(scope="module")
def hr_client(tmp_path_factory: pytest.TempPathFactory):
    """Module-scoped so the real dev embedder + dev cross-encoder each load
    exactly once for this file (mirrors `conftest.client`'s rationale)."""

    root = tmp_path_factory.mktemp("sem_search_hybrid_rerank")
    settings = _build_settings(root)
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def _index_hr_docs(hr_client: TestClient):
    response = hr_client.post("/index", json={"documents": DOCS, "source_corpus": "hr-demo"})
    assert response.status_code == 200
    return response.json()


def test_hybrid_rerank_mode_returns_valid_nonempty_hits(hr_client, _index_hr_docs):
    response = hr_client.post("/search", json={"query": QUERY, "mode": "hybrid_rerank"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid_rerank"
    assert body["hits"]
    assert body["index_version"] == _index_hr_docs["index_version"]
    for hit in body["hits"]:
        assert set(hit.keys()) >= {"chunk_id", "doc_id", "text", "score", "metadata", "highlights"}


def test_hybrid_rerank_changes_top_order_relative_to_hybrid(hr_client, _index_hr_docs):
    """Acceptance criterion #2: reranking must actually change the ordering
    produced by plain `hybrid` fusion, not just relabel the mode."""

    hybrid_response = hr_client.post("/search", json={"query": QUERY, "mode": "hybrid", "top_k": 4})
    rerank_response = hr_client.post(
        "/search", json={"query": QUERY, "mode": "hybrid_rerank", "top_k": 4}
    )

    assert hybrid_response.status_code == 200
    assert rerank_response.status_code == 200

    hybrid_order = [hit["chunk_id"] for hit in hybrid_response.json()["hits"]]
    rerank_order = [hit["chunk_id"] for hit in rerank_response.json()["hits"]]

    assert hybrid_order, "expected non-empty hybrid hits to compare against"
    assert rerank_order != hybrid_order


def test_reranker_failure_degrades_gracefully_to_hybrid_ordering(hr_client, _index_hr_docs):
    """NFR "Надёжность": a broken reranker must not fail the request --
    `/search` with `mode=hybrid_rerank` still returns 200 with valid hits
    (falling back to the pre-rerank, hybrid-fused-and-filtered order), and
    the failure is surfaced as a non-empty `warnings` entry."""

    class RaisingReranker(Reranker):
        def rerank(self, query, candidates, top_n):
            raise RuntimeError("boom: reranker backend unavailable")

        @property
        def model_name(self) -> str:
            return "raising-fake"

    fake = RaisingReranker()
    hr_client.app.state.reranker = fake
    hr_client.app.state.search_service._reranker = fake

    response = hr_client.post("/search", json={"query": QUERY, "mode": "hybrid_rerank"})

    assert response.status_code == 200
    body = response.json()
    assert body["hits"]
    assert body["warnings"]
    assert any("rerank" in w.lower() for w in body["warnings"])
