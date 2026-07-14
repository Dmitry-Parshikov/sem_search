"""Active-index-version resolution for online search.

Unlike the embedder/vector store, there is no single lexical index singleton
for the app's lifetime — the query-time lexical index depends on whichever
`index_version` the manifest currently marks `active`, and that can change
underneath a running process via `/reindex` or `/admin/rollback/{version}`.
`ActiveIndexResolver` makes that work without a restart: it re-checks the
manifest on every call and only rebuilds the (comparatively expensive)
`LexicalIndex` when the active version actually changed, via a single-slot
cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.admin.manifest import IndexManifest
from app.config import Settings
from app.core.errors import NoActiveIndexError
from app.lexical.base import LexicalIndex
from app.lexical.factory import build_lexical_index


@dataclass(frozen=True)
class ActiveIndexContext:
    index_version: str
    collection_name: str
    lexical_index: LexicalIndex


class ActiveIndexResolver:
    """Single-slot cache keyed by `index_version`.

    Not thread-safe against concurrent writers of the *same* slot, but the
    dev/test deployment target here is a single worker process (per the
    plan's documented Qdrant-embedded-mode risk), so this mirrors that
    assumption rather than adding locking that nothing here would exercise.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cached_version: str | None = None
        self._cached_lexical_index: LexicalIndex | None = None

    def resolve(self) -> ActiveIndexContext:
        manifest = IndexManifest.load(Path(self._settings.admin.manifest_path))
        active = manifest.get_active()
        if active is None:
            raise NoActiveIndexError(
                "No active index version found; call POST /index (or /reindex) first."
            )

        index_version = active["index_version"]
        if index_version != self._cached_version or self._cached_lexical_index is None:
            lexical_index = build_lexical_index(self._settings.lexical)
            lexical_index.load(Path(active["lexical_index_path"]))
            self._cached_version = index_version
            self._cached_lexical_index = lexical_index

        return ActiveIndexContext(
            index_version=index_version,
            collection_name=active["vector_collection_name"],
            lexical_index=self._cached_lexical_index,
        )
