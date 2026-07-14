"""Unit tests for `QueryLogger` (Ф4.2): appends one JSON line per `.log()`
call, creates missing parent directories, and never corrupts previously
written lines across repeated calls."""

from __future__ import annotations

import json
from pathlib import Path

from app.admin.query_log import QueryLogger


def test_log_creates_missing_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "queries.jsonl"
    assert not path.parent.exists()

    logger = QueryLogger(path)
    logger.log({"query": "договор аренды", "mode": "hybrid"})

    assert path.exists()


def test_log_appends_valid_json_lines(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    logger = QueryLogger(path)

    logger.log({"query": "первый запрос", "mode": "dense", "index_version": "v1"})
    logger.log({"query": "второй запрос", "mode": "bm25", "index_version": "v1"})
    logger.log({"query": "третий запрос", "mode": "hybrid_rerank", "index_version": "v2"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["query"] == "первый запрос"
    assert parsed[1]["mode"] == "bm25"
    assert parsed[2]["index_version"] == "v2"


def test_log_preserves_non_ascii_text_readable(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    logger = QueryLogger(path)

    logger.log({"query": "русский текст с кириллицей"})

    content = path.read_text(encoding="utf-8")
    # ensure_ascii=False: Cyrillic should appear literally, not \uXXXX-escaped.
    assert "русский текст с кириллицей" in content


def test_log_entry_contains_expected_fields(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    logger = QueryLogger(path)

    entry = {
        "query": "запрос",
        "mode": "hybrid",
        "top_k": 10,
        "must_contain": ["код"],
        "must_exclude": [],
        "index_version": "v_20240101_abcdef",
        "response_time_ms": 12.5,
        "warnings": [],
    }
    logger.log(entry)

    (line,) = path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(line)
    assert parsed == entry
