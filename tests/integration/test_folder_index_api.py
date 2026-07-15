"""Integration tests for the `POST /index-from-folder` endpoint.

Indexes real documents from a temp folder via the API, then verifies they
appear in search results."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_index_from_folder_txt_files(client, tmp_path: Path):
    """Index a folder of UTF-8 .txt files and verify they appear in search."""
    # Create test files
    (tmp_path / "contract_lease.txt").write_text(
        "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ. "
        "Расторжение договора возможно по соглашению сторон или через суд.",
        encoding="utf-8",
    )
    (tmp_path / "docker_guide.md").write_text(
        "# Docker\n\nDocker-контейнер — это изолированное окружение, содержащее "
        "приложение и все его зависимости. Образ собирается по Dockerfile.",
        encoding="utf-8",
    )
    (tmp_path / "art_exhibition.txt").write_text(
        "В центральном выставочном зале города открылась экспозиция современного "
        "искусства. Посетители могут приобрести картины напрямую у авторов.",
        encoding="utf-8",
    )

    # Index
    index_resp = client.post(
        "/index-from-folder",
        json={"folder_path": str(tmp_path), "source_label": "folder-test-1"},
    )

    assert index_resp.status_code == 200
    body = index_resp.json()
    assert body["documents_found"] == 3
    assert body["documents_indexed"] == 3
    assert body["chunk_count"] > 0
    assert body["index_version"]
    assert len(body["errors"]) == 0

    # Verify via search
    search_resp = client.post(
        "/search",
        json={"query": "аренда помещения", "mode": "dense", "top_k": 5},
    )
    assert search_resp.status_code == 200
    hits = search_resp.json()["hits"]
    assert len(hits) >= 1
    # The lease contract doc should be among results
    doc_ids = {h["doc_id"] for h in hits}
    assert "contract_lease" in doc_ids


def test_index_from_folder_reports_file_errors(client, tmp_path: Path):
    """Empty and binary files should be reported as errors, not crash."""
    (tmp_path / "good.txt").write_text("Осмысленный текст.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    # A file with unsupported extension — should be ignored, not errored.
    (tmp_path / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    index_resp = client.post(
        "/index-from-folder",
        json={"folder_path": str(tmp_path), "source_label": "folder-test-2"},
    )

    assert index_resp.status_code == 200
    body = index_resp.json()
    # documents_found includes errored files
    assert body["documents_found"] == 2  # good.txt + empty.txt (png is skipped)
    assert body["documents_indexed"] == 1  # only good.txt
    assert body["chunk_count"] > 0
    assert len(body["errors"]) == 1
    assert body["errors"][0]["path"].endswith("empty.txt")
    assert "empty" in body["errors"][0]["error"].lower()


def test_index_from_folder_empty_folder_returns_422(client, tmp_path: Path):
    """An empty folder (no supported files) should return 422."""
    index_resp = client.post(
        "/index-from-folder",
        json={"folder_path": str(tmp_path), "source_label": "folder-test-3"},
    )

    assert index_resp.status_code == 422
    detail = index_resp.json()["detail"]
    assert "No readable documents" in detail["message"]


def test_index_from_folder_nonexistent_path_returns_422(client, tmp_path: Path):
    """A non-existent folder path should return 422."""
    index_resp = client.post(
        "/index-from-folder",
        json={"folder_path": str(tmp_path / "does_not_exist"), "source_label": "bad"},
    )

    assert index_resp.status_code == 422
