"""`POST /reindex` (Ф1.6 / Ф4.1 / NFR "Воспроизводимость"): full rebuild of
a previously indexed corpus into a brand-new `index_version`, without the
client re-uploading documents.

See `app.indexing.corpus_store` for the design rationale on how the corpus
to reindex is resolved/persisted. If `source_corpus` is omitted in the
request body, the most-recently-indexed corpus (the manifest's last
recorded version's `source_corpus`) is reindexed.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.admin.manifest import IndexManifest
from app.api.schemas import ReindexRequest, ReindexResponse
from app.config import Settings
from app.core.errors import IndexingError
from app.dependencies import get_indexing_service, get_settings
from app.indexing.corpus_store import load_corpus
from app.indexing.service import IndexingService

router = APIRouter()


@router.post("/reindex", response_model=ReindexResponse)
def reindex(
    body: ReindexRequest = ReindexRequest(),
    indexing_service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> ReindexResponse:
    manifest_path = Path(settings.admin.manifest_path)
    manifest = IndexManifest.load(manifest_path)

    source_corpus = body.source_corpus
    if not source_corpus:
        versions = manifest.list_versions()
        if not versions:
            raise HTTPException(
                status_code=404,
                detail="No previous indexing run found; nothing to reindex. Call POST /index first.",
            )
        source_corpus = versions[-1]["source_corpus"]

    try:
        documents = load_corpus(Path(settings.app.data_dir), source_corpus)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No persisted corpus found for source_corpus={source_corpus!r}.",
        ) from exc

    try:
        index_version = indexing_service.run(documents, source_corpus=source_corpus)
    except IndexingError as exc:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {exc}") from exc

    manifest = IndexManifest.load(manifest_path)
    active = manifest.get_active()
    assert active is not None  # IndexingService.run() just recorded it

    return ReindexResponse(
        index_version=index_version,
        document_count=active["document_count"],
        chunk_count=active["chunk_count"],
        source_corpus=source_corpus,
    )
