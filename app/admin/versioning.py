"""Helpers tying `index_version` to per-version storage locations.

Each index version gets its own Qdrant collection and its own BM25 pickle
file, so rollback is a matter of switching the manifest's `active_version`
pointer rather than a destructive operation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.ids import compute_index_version

__all__ = ["compute_index_version", "collection_name_for_version", "lexical_path_for_version"]


def collection_name_for_version(base_name: str, index_version: str) -> str:
    return f"{base_name}__{index_version}"


def lexical_path_for_version(directory: Path, index_version: str) -> Path:
    return Path(directory) / f"{index_version}.pkl"
