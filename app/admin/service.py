"""Admin service (Ф4.1): index version listing and rollback.

Wraps `IndexManifest` + `VectorStore` + `Settings` into the operations
`app.api.routes_admin` needs. Rollback is non-destructive: each index version
already owns its own Qdrant collection and BM25 pickle, so "rolling back"
never rebuilds anything — it only validates that the target version's assets
still exist, then flips the manifest's `status` fields and `active_version`
pointer.

The manifest is loaded fresh from disk on every call (same pattern as
`app.search.active_index.ActiveIndexResolver`) rather than cached on
`AdminService`, so concurrent admin calls and search requests always see the
latest on-disk state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.admin.manifest import IndexManifest
from app.config import Settings
from app.core.errors import IndexVersionAssetsMissingError, IndexVersionNotFoundError
from app.vector_store.base import VectorStore


class AdminService:
    def __init__(self, vector_store: VectorStore, settings: Settings) -> None:
        self._vector_store = vector_store
        self._settings = settings

    def _manifest_path(self) -> Path:
        return Path(self._settings.admin.manifest_path)

    def list_versions(self) -> list[dict[str, Any]]:
        return IndexManifest.load(self._manifest_path()).list_versions()

    def get_active(self) -> dict[str, Any] | None:
        return IndexManifest.load(self._manifest_path()).get_active()

    def rollback(self, target_version: str) -> dict[str, Any]:
        """Reactivates a previously-recorded (`superseded`) index version.

        Raises `IndexVersionNotFoundError` if `target_version` isn't in the
        manifest at all, or `IndexVersionAssetsMissingError` if it is recorded
        but its Qdrant collection / lexical pickle are gone (e.g. manually
        deleted) -- rollback cannot resurrect data, only repoint to it.
        """

        manifest_path = self._manifest_path()
        manifest = IndexManifest.load(manifest_path)

        target_entry: dict[str, Any] | None = None
        for version in manifest.versions:
            if version.get("index_version") == target_version:
                target_entry = version
                break
        if target_entry is None:
            raise IndexVersionNotFoundError(f"Index version {target_version!r} not found in manifest.")

        collection_ok = self._vector_store.collection_exists(target_entry["vector_collection_name"])
        lexical_ok = Path(target_entry["lexical_index_path"]).exists()
        if not (collection_ok and lexical_ok):
            raise IndexVersionAssetsMissingError(
                f"Assets for index version {target_version!r} are missing "
                f"(collection_exists={collection_ok}, lexical_path_exists={lexical_ok}); "
                "cannot roll back to it."
            )

        for version in manifest.versions:
            if version.get("status") == "active":
                version["status"] = "superseded"
        target_entry["status"] = "active"
        manifest.active_version = target_version
        manifest.save(manifest_path)

        return target_entry
