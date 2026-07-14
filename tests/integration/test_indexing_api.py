"""Integration tests for Phase 3's API layer (/index, /reindex, /health),
maps to plan step 3: "проверить на синтетическом корпусе (черновой)".

Runs against a real embedded Qdrant + the real dev ST embedder (see
`conftest.client`), so it's marked slow -- run explicitly with
`pytest tests/integration` (not filtered out by `-m "not slow"`).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

SAMPLE_DOCS = [
    {
        "doc_id": "d1",
        "text": "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ.",
        "source": "test",
    },
    {
        "doc_id": "d2",
        "text": "API — программный интерфейс, позволяющий приложениям обмениваться данными.",
        "source": "test",
    },
    {
        "doc_id": "d3",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "d4",
        "text": "REST и HTTP — основа большинства современных веб-сервисов и интеграций.",
        "source": "test",
    },
]


def test_index_returns_valid_response(client):
    response = client.post("/index", json={"documents": SAMPLE_DOCS, "source_corpus": "demo"})

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == len(SAMPLE_DOCS)
    assert body["chunk_count"] > 0
    assert body["index_version"]


def test_health_after_indexing_reports_ok_with_matching_version(client):
    index_response = client.post("/index", json={"documents": SAMPLE_DOCS, "source_corpus": "demo"})
    index_version = index_response.json()["index_version"]

    health_response = client.get("/health")

    assert health_response.status_code == 200
    body = health_response.json()
    assert body["status"] == "ok"
    assert body["index_version"] == index_version
    assert body["subsystems"]["vector_store"] is True
    assert body["subsystems"]["embedder"] is True
    assert body["subsystems"]["reranker"] == "not_configured"


def test_reindex_without_body_produces_new_version(client):
    first = client.post("/index", json={"documents": SAMPLE_DOCS, "source_corpus": "demo"})
    first_version = first.json()["index_version"]

    reindex_response = client.post("/reindex", json={})

    assert reindex_response.status_code == 200
    body = reindex_response.json()
    # `compute_index_version` includes a timestamp, so re-running the same
    # corpus/config must still yield a distinct version.
    assert body["index_version"] != first_version
    assert body["source_corpus"] == "demo"
    assert body["document_count"] == len(SAMPLE_DOCS)
    assert body["chunk_count"] > 0

    health_response = client.get("/health")
    assert health_response.json()["index_version"] == body["index_version"]


def test_reindex_unknown_corpus_returns_404(client):
    response = client.post("/reindex", json={"source_corpus": "does-not-exist-corpus"})

    assert response.status_code == 404


def test_health_before_any_indexing_never_crashes(fresh_client):
    response = fresh_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["index_version"] is None
    assert body["status"] in ("ok", "degraded")


def test_reindex_with_no_prior_index_returns_404(fresh_client):
    response = fresh_client.post("/reindex", json={})

    assert response.status_code == 404
