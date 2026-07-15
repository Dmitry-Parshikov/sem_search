"""`POST /index-from-folder`: scan a local folder for supported documents
(.txt, .md, .docx, .rtf), read text with auto encoding detection, and
feed the results into the standard `IndexingService` pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.admin.manifest import IndexManifest
from app.api.schemas import FileError, FolderIndexRequest, FolderIndexResponse
from app.config import Settings
from app.core.errors import IndexingError
from app.core.types import Document
from app.dependencies import get_indexing_service, get_settings
from app.indexing.corpus_store import save_corpus
from app.indexing.service import IndexingService
from app.preprocessing.file_loader import load_folder

router = APIRouter()


@router.post("/index-from-folder", response_model=FolderIndexResponse)
def index_from_folder(
    body: FolderIndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> FolderIndexResponse:
    # 1. Load files from disk
    raw_docs, load_errors = load_folder(body.folder_path, source_label=body.source_label)

    file_errors = [
        FileError(path=e["path"], suffix=e.get("suffix"), error=e["error"])
        for e in load_errors
    ]

    if not raw_docs:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No readable documents found in the folder",
                "errors": [e.model_dump() for e in file_errors],
            },
        )

    documents = [
        Document(
            doc_id=doc["doc_id"],
            text=doc["text"],
            source=doc["source"],
        )
        for doc in raw_docs
    ]

    # 2. Run standard indexing pipeline
    try:
        index_version = indexing_service.run(documents, source_corpus=body.source_label)
    except IndexingError as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    save_corpus(documents, Path(settings.app.data_dir), body.source_label)

    manifest = IndexManifest.load(Path(settings.admin.manifest_path))
    active = manifest.get_active()
    assert active is not None

    return FolderIndexResponse(
        index_version=index_version,
        documents_found=len(raw_docs) + len(load_errors),
        documents_indexed=len(raw_docs),
        chunk_count=active["chunk_count"],
        errors=file_errors,
    )
