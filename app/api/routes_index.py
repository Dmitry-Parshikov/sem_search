"""`POST /index` (Ф1.1-Ф1.6): preprocessing -> chunking -> embedding ->
vector store + lexical index, orchestrated by `IndexingService`.

Also persists the raw documents (see `app.indexing.corpus_store` for the
design rationale) so `POST /reindex` can later replay this corpus without
the client re-uploading it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.admin.manifest import IndexManifest
from app.api.schemas import IndexRequest, IndexResponse
from app.config import Settings
from app.core.errors import IndexingError
from app.core.types import Document
from app.dependencies import get_indexing_service, get_settings
from app.indexing.corpus_store import save_corpus
from app.indexing.service import IndexingService

router = APIRouter()


@router.post("/index", response_model=IndexResponse)
def index_documents(
    body: IndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    documents = [
        Document(
            doc_id=doc.doc_id,
            text=doc.text,
            source=doc.source,
            section=doc.section,
            date=doc.date,
            extra=doc.extra,
        )
        for doc in body.documents
    ]

    try:
        index_version = indexing_service.run(documents, source_corpus=body.source_corpus)
    except IndexingError as exc:
        # Indexing failure is NOT one of the NFR "Надёжность" graceful-degradation
        # cases -- that NFR covers optional query-time stages (reranker, query
        # expansion). A failed indexing run must be reported, not swallowed.
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    save_corpus(documents, Path(settings.app.data_dir), body.source_corpus)

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    active = manifest.get_active()
    assert active is not None  # IndexingService.run() just recorded it

    return IndexResponse(
        index_version=index_version,
        document_count=active["document_count"],
        chunk_count=active["chunk_count"],
    )
