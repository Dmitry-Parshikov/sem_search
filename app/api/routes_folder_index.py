"""`POST /index-from-folder`: scan a local folder for supported documents
(.txt, .md, .docx, .rtf), read text with auto encoding detection, and
feed the results into the standard `IndexingService` pipeline.

`POST /index-upload`: accept files uploaded directly from the browser
(via `<input type="file" webkitdirectory>` or `<input type="file" multiple>`),
so the user can pick a folder or individual files through a native OS
dialog instead of typing a server-side path.

Both endpoints support ``mode=replace|append``:
- ``replace`` (default) — full replacement of the corpus.
- ``append`` — load the existing persisted corpus, merge new documents
  (matching ``doc_id`` overwrites old), save the combined list, and
  reindex everything.  This allows building a knowledge base from
  multiple folders / file batches over time.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.admin.manifest import IndexManifest
from app.api.schemas import FileError, FolderIndexRequest, FolderIndexResponse
from app.config import Settings
from app.core.errors import IndexingError
from app.core.types import Document
from app.dependencies import get_indexing_service, get_settings
from app.indexing.corpus_store import load_corpus, save_corpus
from app.indexing.service import IndexingService
from app.preprocessing.file_loader import load_files, load_folder

router = APIRouter()


def _merge_documents(
    existing: list[Document],
    incoming: list[Document],
) -> list[Document]:
    """Merge incoming documents into an existing list.
    Documents with the same ``doc_id`` are overwritten by the incoming version
    (last-write-wins), so re-uploading a file updates it rather than
    duplicating."""
    merged: dict[str, Document] = {d.doc_id: d for d in existing}
    for d in incoming:
        merged[d.doc_id] = d
    return list(merged.values())


def _run_indexing(
    documents: list[Document],
    source_label: str,
    mode: str,
    data_dir: Path,
    indexing_service: IndexingService,
    manifest_path: Path,
) -> tuple[str, int, int]:
    """Shared indexing pipeline: optionally merge with existing corpus,
    run the indexer, persist the result."""

    if mode == "append":
        try:
            existing = load_corpus(data_dir, source_label)
        except FileNotFoundError:
            existing = []
        documents = _merge_documents(existing, documents)

    # Run standard indexing pipeline
    try:
        index_version = indexing_service.run(documents, source_corpus=source_label)
    except IndexingError as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    save_corpus(documents, data_dir, source_label)

    manifest = IndexManifest.load(manifest_path)
    active = manifest.get_active()
    assert active is not None

    return index_version, len(documents), active["chunk_count"]


@router.post("/index-from-folder", response_model=FolderIndexResponse)
def index_from_folder(
    body: FolderIndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> FolderIndexResponse:
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

    index_version, total_docs, chunk_count = _run_indexing(
        documents=documents,
        source_label=body.source_label,
        mode=body.mode,
        data_dir=Path(settings.app.data_dir),
        indexing_service=indexing_service,
        manifest_path=Path(settings.admin.manifest_path),
    )

    return FolderIndexResponse(
        index_version=index_version,
        documents_found=len(raw_docs) + len(load_errors),
        documents_indexed=len(raw_docs),
        chunk_count=chunk_count,
        errors=file_errors,
    )


@router.post("/index-upload", response_model=FolderIndexResponse)
async def index_upload(
    files: list[UploadFile],
    source_label: str = Form("upload"),
    mode: str = Form("replace"),
    indexing_service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> FolderIndexResponse:
    """Accept files uploaded from the browser (folder picker or multi-file
    picker), read them with the same auto-encoding pipeline used by
    `load_folder`, and run the standard indexing pipeline.

    Supports ``mode=append`` to add files to an existing corpus without
    losing previously indexed documents."""
    file_tuples: list[tuple[str, bytes, str]] = []
    for f in files:
        content = await f.read()
        suffix = Path(f.filename or "unknown").suffix
        file_tuples.append((f.filename or "unknown", content, suffix))

    raw_docs, load_errors = load_files(file_tuples, source_label=source_label)

    file_errors = [
        FileError(path=e["path"], suffix=e.get("suffix"), error=e["error"])
        for e in load_errors
    ]

    if not raw_docs:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No readable documents in the uploaded files",
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

    index_version, total_docs, chunk_count = _run_indexing(
        documents=documents,
        source_label=source_label,
        mode=mode,
        data_dir=Path(settings.app.data_dir),
        indexing_service=indexing_service,
        manifest_path=Path(settings.admin.manifest_path),
    )

    return FolderIndexResponse(
        index_version=index_version,
        documents_found=len(raw_docs) + len(load_errors),
        documents_indexed=len(raw_docs),
        chunk_count=chunk_count,
        errors=file_errors,
    )
