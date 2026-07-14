"""Integration tests for query processing wired into `/search`:
Ф2.2 typo correction (suggestion only, never blocking) and Ф2.4 term
expansion (actually changes what gets retrieved), plus the NFR
"Надёжность" degrade-gracefully contract for both optional stages.

Runs against a real embedded Qdrant + the real dev ST embedder (same setup
as `test_search_api.py`), so it's marked slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import (
    AdminConfig,
    AppMeta,
    QdrantConfig,
    QueryProcessingConfig,
    Settings,
    TermExpansionConfig,
    VectorStoreConfig,
)
from app.main import create_app
from app.query.base import TermExpander, TypoCorrector

pytestmark = pytest.mark.slow

# qp2 (a decoy, unrelated doc) is listed FIRST on purpose: if term expansion
# had no real effect, a lexical (BM25) query for the bare abbreviation "бд"
# would match nothing in either doc, and any accidental tie-break would tend
# to favor whichever doc came first -- i.e. qp2, not qp1. Only a real
# expansion of "бд" -> "база данных" gives qp1 a genuine, non-accidental BM25
# score advantage.
DOCS = [
    {
        "doc_id": "qp2",
        "text": "Сегодня в городе открылась выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "qp1",
        "text": "База данных проекта хранит договоры аренды и связанные документы.",
        "source": "test",
    },
]


def _build_settings(root: Path, terms_path: Path) -> Settings:
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
                collection_name="qp_chunks",
            )
        ),
        # Test-specific term dictionary (not the real project
        # config/terms_dictionary.yaml, which may change independently) --
        # injected via a Settings override, same pattern conftest.py uses for
        # admin.manifest_path etc.
        query_processing=QueryProcessingConfig(
            term_expansion=TermExpansionConfig(enabled=True, dictionary_path=str(terms_path)),
        ),
    )


@pytest.fixture()
def qp_client(tmp_path: Path):
    """A dedicated app/client per test with its own temp `terms_dictionary
    .yaml` ("бд" -> "база данных") and its own empty index."""

    terms_path = tmp_path / "terms_dictionary.yaml"
    terms_path.write_text("бд:\n  - база данных\n", encoding="utf-8")

    settings = _build_settings(tmp_path, terms_path)
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def _index_qp_docs(qp_client: TestClient):
    response = qp_client.post("/index", json={"documents": DOCS, "source_corpus": "qp-demo"})
    assert response.status_code == 200
    return response.json()


def test_term_expansion_finds_doc_via_expanded_synonym_not_abbreviation(qp_client, _index_qp_docs):
    """"бд" (the abbreviation) appears in NO document text -- only its
    dictionary expansion "база данных" does, and only in doc qp1. A pure-BM25
    query for just "бд" can only find qp1 once expansion actually runs and
    changes the text handed to the retriever -- proving Ф2.4 changes
    retrieval, not just cosmetically returns a string."""

    response = qp_client.post("/search", json={"query": "бд", "mode": "bm25", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["expanded_query"] is not None
    assert "база данных" in body["expanded_query"]
    assert body["hits"], "expected the expanded query to retrieve at least one hit"
    assert body["hits"][0]["doc_id"] == "qp1"


def test_search_response_includes_typo_suggestion_and_warnings_fields(qp_client, _index_qp_docs):
    """Schema round-trip through the real API: `typo_suggestion` (nullable)
    and `warnings` (empty list in the happy path) are always present."""

    response = qp_client.post("/search", json={"query": "договор аренды", "mode": "dense"})

    assert response.status_code == 200
    body = response.json()
    assert "typo_suggestion" in body
    assert "warnings" in body
    assert body["warnings"] == []


def test_term_expansion_failure_degrades_gracefully(qp_client, _index_qp_docs):
    """NFR "Надёжность": a broken term expander must not fail the request --
    `/search` still returns 200 with valid hits, and the failure is surfaced
    as a non-empty `warnings` entry."""

    class RaisingExpander(TermExpander):
        def expand(self, query: str) -> str:
            raise RuntimeError("boom: expansion backend unavailable")

    fake = RaisingExpander()
    qp_client.app.state.term_expander = fake
    qp_client.app.state.search_service._term_expander = fake

    response = qp_client.post("/search", json={"query": "договор", "mode": "dense"})

    assert response.status_code == 200
    body = response.json()
    assert body["hits"]
    assert body["warnings"]
    assert any("expansion" in w.lower() for w in body["warnings"])


def test_typo_correction_failure_degrades_gracefully(qp_client, _index_qp_docs):
    """Same NFR, for the typo-correction stage."""

    class RaisingCorrector(TypoCorrector):
        def suggest(self, query: str, vocabulary: set[str]) -> str | None:
            raise RuntimeError("boom: typo backend unavailable")

    fake = RaisingCorrector()
    qp_client.app.state.typo_corrector = fake
    qp_client.app.state.search_service._typo_corrector = fake

    response = qp_client.post("/search", json={"query": "договор", "mode": "dense"})

    assert response.status_code == 200
    body = response.json()
    assert body["hits"]
    assert body["warnings"]
    assert any("typo" in w.lower() for w in body["warnings"])
