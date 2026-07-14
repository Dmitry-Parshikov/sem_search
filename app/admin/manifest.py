"""Index version manifest (Ф1.6 / Ф4.1 data layer).

Full admin API (routes) comes in Phase 8; this module just owns the JSON
data structure and atomic read/write, since `IndexingService` (Phase 2)
needs to record a version at the end of every indexing run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class IndexManifest:
    """Wraps the manifest JSON structure:

        {
            "active_version": str | None,
            "versions": [
                {
                    "index_version": str,
                    "created_at": str (ISO8601),
                    "embedding_model": str,
                    "embedding_dimension": int,
                    "chunking_strategy": str,
                    "chunking_config_signature": str,
                    "lexical_lemmatization": bool,
                    "bm25_params": {"k1": float, "b": float},
                    "vector_collection_name": str,
                    "lexical_index_path": str,
                    "document_count": int,
                    "chunk_count": int,
                    "source_corpus": str,
                    "status": "active" | "superseded",
                },
                ...
            ],
        }
    """

    def __init__(self, active_version: str | None = None, versions: list[dict[str, Any]] | None = None) -> None:
        self.active_version = active_version
        self.versions: list[dict[str, Any]] = versions if versions is not None else []

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(active_version=data.get("active_version"), versions=data.get("versions", []))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        data = {"active_version": self.active_version, "versions": self.versions}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def record_new(self, entry: dict[str, Any]) -> None:
        """Appends a new version entry, marks the previous active version
        (if any) as superseded, and sets the new entry as active."""

        for version in self.versions:
            if version.get("status") == "active":
                version["status"] = "superseded"

        entry = dict(entry)
        entry["status"] = "active"
        self.versions.append(entry)
        self.active_version = entry["index_version"]

    def get_active(self) -> dict[str, Any] | None:
        if self.active_version is None:
            return None
        for version in self.versions:
            if version.get("index_version") == self.active_version:
                return version
        return None

    def list_versions(self) -> list[dict[str, Any]]:
        return list(self.versions)
