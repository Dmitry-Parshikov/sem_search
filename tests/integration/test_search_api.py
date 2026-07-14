"""Integration tests for Phase 4's `/search` endpoint (dense/bm25 modes),
maps to plan step 4: dense-retriever, BM25-retriever, `dense`/`bm25` modes
in `/search`.

Runs against a real embedded Qdrant + the real dev ST embedder (see
`conftest.client`), so it's marked slow.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

SAMPLE_DOCS = [
    {
        "doc_id": "s1",
        "text": "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ.",
        "source": "test",
    },
    {
        "doc_id": "s2",
        "text": "API — программный интерфейс, позволяющий приложениям обмениваться данными по протоколу HTTP.",
        "source": "test",
    },
    {
        "doc_id": "s3",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "s4",
        "text": "REST и HTTP — основа большинства современных веб-сервисов и интеграций.",
        "source": "test",
    },
    {
        "doc_id": "s5",
        "text": "Аренда квартиры оформляется письменным соглашением между сторонами.",
        "source": "test",
    },
]


@pytest.fixture(scope="module", autouse=True)
def _index_sample_corpus(client):
    """Indexes the sample corpus once for this module's `client`-based tests."""

    response = client.post("/index", json={
        "documents": SAMPLE_DOCS,
        "source_corpus": "search-demo",
    })
    assert response.status_code == 200
    return response.json()


def test_search_dense_mode_returns_hits(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "договор аренды", "mode": "dense"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "dense"
    assert body["hits"]
    assert body["index_version"] == _index_sample_corpus["index_version"]
    for hit in body["hits"]:
        assert set(hit.keys()) >= {"chunk_id", "doc_id", "text", "score", "metadata", "highlights"}


def test_search_bm25_mode_returns_hits(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "HTTP REST", "mode": "bm25"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "bm25"
    assert body["hits"]
    assert body["index_version"] == _index_sample_corpus["index_version"]


def test_search_hybrid_mode_returns_501(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "договор аренды", "mode": "hybrid"})

    assert response.status_code == 501


def test_search_hybrid_rerank_mode_returns_501(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "договор аренды", "mode": "hybrid_rerank"})

    assert response.status_code == 501


def test_search_top_k_is_respected(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "документ", "mode": "dense", "top_k": 2})

    assert response.status_code == 200
    assert len(response.json()["hits"]) <= 2


def test_search_before_any_indexing_returns_404(fresh_client):
    response = fresh_client.post("/search", json={"query": "что угодно", "mode": "dense"})

    assert response.status_code == 404


def test_search_uses_default_mode_and_top_k_when_omitted(client, _index_sample_corpus):
    response = client.post("/search", json={"query": "аренда"})

    # default_mode is "hybrid_rerank" per config.py -- not implemented yet, so
    # this should surface as 501, confirming the default is actually applied.
    assert response.status_code == 501
