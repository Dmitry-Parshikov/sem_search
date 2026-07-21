"""Persisted-corpus store backing the `/reindex` design decision.

Design decision (not spelled out by requirements.md section 9's API
contract, so documented here explicitly -- same rationale-in-comment
approach as the admin-router and must_contain-scope decisions in the plan):

`IndexingService.run()` needs an explicit `documents: list[Document]`
argument, but `POST /reindex`'s job per the spec is "full reindex of the
collection into a new index version" (e.g. after an embedding-model or
chunking config change) *without* the client re-uploading every document.

Resolution: `POST /index` persists the raw `Document` list it receives to
disk as JSON (one file per `source_corpus`, under `{data_dir}/corpus/`), in
addition to running `IndexingService`. `POST /reindex` loads that JSON back
and re-runs `IndexingService.run()` on it to produce a new version. This
keeps `IndexingService` itself corpus-storage-agnostic (it only ever sees an
in-memory `list[Document]`) while giving `/reindex` something to replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.types import Document

_CORPUS_SUBDIR = "corpus"


def _corpus_path(data_dir: Path, source_corpus: str) -> Path:
    return Path(data_dir) / _CORPUS_SUBDIR / f"{source_corpus}.json"


def document_to_dict(document: Document) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "text": document.text,
        "source": document.source,
        "section": document.section,
        "date": document.date,
        "extra": document.extra,
    }


def document_from_dict(data: dict[str, Any]) -> Document:
    return Document(
        doc_id=data["doc_id"],
        text=data["text"],
        source=data.get("source", "unknown"),
        section=data.get("section"),
        date=data.get("date"),
        extra=data.get("extra", {}),
    )


def save_corpus(documents: list[Document], data_dir: Path, source_corpus: str) -> Path:
    """Persists `documents` as JSON, replacing any previous save under the
    same `source_corpus` name (a fresh `/index` call for a corpus name is a
    full replacement -- consistent with reindex always being a full
    rebuild, never an incremental patch)."""

    path = _corpus_path(data_dir, source_corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    payload = [document_to_dict(d) for d in documents]
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    return path


def delete_document(data_dir: Path, source_corpus: str, doc_id: str) -> tuple[bool, int]:
    """Remove a single document from the persisted corpus JSON.

    Returns ``(deleted: bool, remaining_count: int)``.
    Raises ``FileNotFoundError`` if the corpus does not exist.
    """
    path = _corpus_path(data_dir, source_corpus)
    if not path.exists():
        raise FileNotFoundError(f"No persisted corpus found for source_corpus={source_corpus!r} at {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    original_len = len(payload)
    filtered = [d for d in payload if d.get("doc_id") != doc_id]
    removed = original_len - len(filtered)

    if removed:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    return removed > 0, len(filtered)


def load_corpus(data_dir: Path, source_corpus: str) -> list[Document]:
    """Raises `FileNotFoundError` if no corpus was ever persisted under this
    `source_corpus` name (caller -- `routes_reindex` -- turns this into a
    404)."""

    path = _corpus_path(data_dir, source_corpus)
    if not path.exists():
        raise FileNotFoundError(f"No persisted corpus found for source_corpus={source_corpus!r} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [document_from_dict(d) for d in payload]
