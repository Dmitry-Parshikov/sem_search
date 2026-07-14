"""Integration tests for `/admin/versions` and `/admin/rollback/{version}`
(Ф4.1, plan decision #5) plus the query-log side effect of `/search` (Ф4.2).

Runs against a real embedded Qdrant + the real dev ST embedder (see
`conftest.client`), so it's marked slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

FIRST_DOCS = [
    {
        "doc_id": "a1",
        "text": "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ.",
        "source": "test",
    },
    {
        "doc_id": "a2",
        "text": "API — программный интерфейс, позволяющий приложениям обмениваться данными.",
        "source": "test",
    },
]

SECOND_DOCS = [
    {
        "doc_id": "b1",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "b2",
        "text": "REST и HTTP — основа большинства современных веб-сервисов и интеграций.",
        "source": "test",
    },
]


def test_versions_after_single_index_shows_one_active_entry(fresh_client):
    index_response = fresh_client.post(
        "/index", json={"documents": FIRST_DOCS, "source_corpus": "admin-demo"}
    )
    assert index_response.status_code == 200
    first_version = index_response.json()["index_version"]

    versions_response = fresh_client.get("/admin/versions")
    assert versions_response.status_code == 200
    body = versions_response.json()

    assert body["active_version"] == first_version
    assert len(body["versions"]) == 1
    assert body["versions"][0]["index_version"] == first_version
    assert body["versions"][0]["status"] == "active"


def test_reindex_adds_second_version_and_supersedes_first(fresh_client):
    first = fresh_client.post("/index", json={"documents": FIRST_DOCS, "source_corpus": "admin-demo-2"})
    first_version = first.json()["index_version"]

    second = fresh_client.post("/reindex", json={})
    assert second.status_code == 200
    second_version = second.json()["index_version"]
    assert second_version != first_version

    versions_response = fresh_client.get("/admin/versions")
    body = versions_response.json()

    assert body["active_version"] == second_version
    assert len(body["versions"]) == 2

    by_version = {v["index_version"]: v for v in body["versions"]}
    assert by_version[first_version]["status"] == "superseded"
    assert by_version[second_version]["status"] == "active"


def test_rollback_switches_active_version_and_search_reflects_it(fresh_client):
    first = fresh_client.post("/index", json={"documents": FIRST_DOCS, "source_corpus": "admin-demo-3"})
    first_version = first.json()["index_version"]

    second = fresh_client.post("/reindex", json={})
    second_version = second.json()["index_version"]
    assert second_version != first_version

    rollback_response = fresh_client.post(f"/admin/rollback/{first_version}")
    assert rollback_response.status_code == 200
    rollback_body = rollback_response.json()
    assert rollback_body["active_version"] == first_version

    versions_response = fresh_client.get("/admin/versions")
    by_version = {v["index_version"]: v for v in versions_response.json()["versions"]}
    assert by_version[first_version]["status"] == "active"
    assert by_version[second_version]["status"] == "superseded"

    # `ActiveIndexResolver` re-reads the manifest on every `.resolve()` call
    # (see its docstring) -- a rollback must be picked up by `/search`
    # immediately, with no process restart.
    search_response = fresh_client.post("/search", json={"query": "договор аренды", "mode": "dense"})
    assert search_response.status_code == 200
    assert search_response.json()["index_version"] == first_version


def test_rollback_unknown_version_returns_404(fresh_client):
    fresh_client.post("/index", json={"documents": FIRST_DOCS, "source_corpus": "admin-demo-4"})

    response = fresh_client.post("/admin/rollback/does-not-exist-version")
    assert response.status_code == 404


def test_search_appends_query_log_entry(fresh_client):
    fresh_client.post("/index", json={"documents": FIRST_DOCS, "source_corpus": "admin-demo-5"})

    search_response = fresh_client.post("/search", json={"query": "договор аренды", "mode": "dense"})
    assert search_response.status_code == 200
    index_version = search_response.json()["index_version"]

    settings = fresh_client.app.state.settings
    log_path = Path(settings.admin.query_log_path)
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines

    entries = [json.loads(line) for line in lines]
    matching = [e for e in entries if e.get("query") == "договор аренды"]
    assert matching

    entry = matching[-1]
    assert entry["mode"] == "dense"
    assert entry["index_version"] == index_version
    assert "response_time_ms" in entry
    assert isinstance(entry["response_time_ms"], (int, float))
