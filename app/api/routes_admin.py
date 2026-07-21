"""`GET /admin/versions`, `POST /admin/rollback/{version}` (Ф4.1): index
version listing and non-destructive rollback. Covers the "просмотр статуса
индекса ... откат к предыдущей версии" requirement.

`GET /admin/config` (development helper): exposes the running configuration
(read-only) so the web UI can show current embedder / chunking / reranker /
hybridization settings without needing server-side rendering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.admin.service import AdminService
from app.api.schemas import (
    AdminConfigResponse,
    ChunkingConfigOut,
    CorporaResponse,
    CorpusDetailResponse,
    CorpusDocumentInfo,
    CorpusInfo,
    DeleteDocumentResponse,
    DictionariesResponse,
    DictionaryInfo,
    DictionaryToggleRequest,
    EmbeddingConfigOut,
    HybridizationConfigOut,
    QueryProcessingConfigOut,
    RerankingConfigOut,
    RollbackResponse,
    SearchConfigOut,
    TypoCorrectionConfigOut,
    VersionInfo,
    VersionsResponse,
)
from app.config import Settings
from app.core.errors import IndexVersionAssetsMissingError, IndexVersionNotFoundError
from app.dependencies import get_admin_service, get_settings
from app.indexing.corpus_store import delete_document

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/versions", response_model=VersionsResponse)
def list_versions(admin_service: AdminService = Depends(get_admin_service)) -> VersionsResponse:
    versions = admin_service.list_versions()
    active = admin_service.get_active()
    return VersionsResponse(
        versions=[VersionInfo(**v) for v in versions],
        active_version=active["index_version"] if active else None,
    )


@router.post("/rollback/{version}", response_model=RollbackResponse)
def rollback(version: str, admin_service: AdminService = Depends(get_admin_service)) -> RollbackResponse:
    try:
        entry = admin_service.rollback(version)
    except IndexVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexVersionAssetsMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return RollbackResponse(
        active_version=entry["index_version"],
        created_at=entry["created_at"],
        document_count=entry["document_count"],
        chunk_count=entry["chunk_count"],
        source_corpus=entry["source_corpus"],
    )


@router.get("/config", response_model=AdminConfigResponse)
def get_config(settings: Settings = Depends(get_settings)) -> AdminConfigResponse:
    """Return the current running configuration (read-only) for the web UI."""
    chunking_params = settings.chunking.active_params()
    return AdminConfigResponse(
        config_profile=os.environ.get("SEM_SEARCH_CONFIG", "config/config.yaml"),
        embedding=EmbeddingConfigOut(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
            query_prefix=settings.embedding.query_prefix,
            passage_prefix=settings.embedding.passage_prefix,
        ),
        chunking=ChunkingConfigOut(
            strategy=settings.chunking.strategy,
            params=chunking_params.model_dump() if hasattr(chunking_params, "model_dump") else dict(chunking_params),
        ),
        reranking=RerankingConfigOut(
            enabled=settings.reranking.enabled,
            model_name=settings.reranking.model_name,
            device=settings.reranking.device,
            top_n=settings.reranking.top_n,
            batch_size=settings.reranking.batch_size,
        ),
        hybridization=HybridizationConfigOut(
            method=settings.hybridization.method,
            rrf_k=settings.hybridization.rrf_k,
        ),
        query_processing=QueryProcessingConfigOut(
            typo_correction=TypoCorrectionConfigOut(
                enabled=settings.query_processing.typo_correction.enabled,
                max_distance=settings.query_processing.typo_correction.max_distance,
                score_cutoff=settings.query_processing.typo_correction.score_cutoff,
            ),
            dictionaries_enabled=settings.query_processing.dictionaries_enabled,
            dictionaries_dir=settings.query_processing.dictionaries_dir,
        ),
        search=SearchConfigOut(
            default_mode=settings.search.default_mode,
            default_top_k=settings.search.default_top_k,
        ),
    )


@router.get("/corpora", response_model=CorporaResponse)
def list_corpora(settings: Settings = Depends(get_settings)) -> CorporaResponse:
    """Return every persisted corpus from ``{data_dir}/corpus/`` with
    document count, size and modification time so the web UI can show a
    pickable table rather than asking users to type folder paths."""
    corpus_dir = Path(settings.app.data_dir) / "corpus"
    corpora: list[CorpusInfo] = []

    if corpus_dir.is_dir():
        for json_file in sorted(corpus_dir.glob("*.json")):
            try:
                stat = json_file.stat()
                with open(json_file, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                count = len(docs) if isinstance(docs, list) else 0
            except (OSError, json.JSONDecodeError):
                count = 0

            from datetime import timezone

            mtime = stat.st_mtime
            from datetime import datetime

            last_mod = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            corpora.append(
                CorpusInfo(
                    name=json_file.stem,
                    document_count=count,
                    size_bytes=stat.st_size,
                    last_modified=last_mod,
                )
            )

    return CorporaResponse(corpora=corpora)


@router.get("/corpora/{name}", response_model=CorpusDetailResponse)
def get_corpus_detail(name: str, settings: Settings = Depends(get_settings)) -> CorpusDetailResponse:
    """Return the full document list for a single persisted corpus."""
    corpus_dir = Path(settings.app.data_dir) / "corpus"
    json_file = corpus_dir / f"{name}.json"

    if not json_file.is_file():
        raise HTTPException(status_code=404, detail=f"Corpus {name!r} not found")

    with open(json_file, "r", encoding="utf-8") as f:
        docs: list[dict] = json.load(f)

    documents = [
        CorpusDocumentInfo(
            doc_id=d.get("doc_id", "?"),
            text_preview=(d.get("text", "") or "")[:120],
            text_length=len(d.get("text", "") or ""),
        )
        for d in docs
    ]

    return CorpusDetailResponse(name=name, document_count=len(documents), documents=documents)


@router.delete("/corpora/{name}/documents/{doc_id}", response_model=DeleteDocumentResponse)
def remove_corpus_document(
    name: str,
    doc_id: str,
    settings: Settings = Depends(get_settings),
) -> DeleteDocumentResponse:
    """Remove a single document from a persisted corpus JSON.

    The document's vectors will remain in the active Qdrant index until the
    next reindex — removing from the JSON means the document won't reappear
    after reindex.
    """
    try:
        deleted, remaining = delete_document(Path(settings.app.data_dir), name, doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Corpus {name!r} not found")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id!r} not found in corpus {name!r}")

    return DeleteDocumentResponse(
        deleted=True,
        doc_id=doc_id,
        corpus=name,
        message=f"Удалён из {name}.json. Осталось {remaining} док. Требуется переиндексация для полного удаления из поиска.",
    )


# ── Dictionaries ─────────────────────────────────────────────────────

def _rebuild_expander(request: Request, settings: Settings) -> None:
    """Rebuild the term expander on ``app.state`` after a dictionary change."""
    from app.query.factory import build_query_expander

    request.app.state.term_expander = build_query_expander(settings.query_processing)


@router.get("/dictionaries", response_model=DictionariesResponse)
def list_dictionaries(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> DictionariesResponse:
    """Return every dictionary file in ``dictionaries_dir`` with entry counts."""
    dir_path = Path(settings.query_processing.dictionaries_dir)
    items: list[DictionaryInfo] = []

    if dir_path.is_dir():
        from app.query.term_expansion import load_dictionary

        for fpath in sorted(dir_path.glob("*")):
            if fpath.suffix.lower() not in (".json", ".yaml", ".yml"):
                continue
            d = load_dictionary(fpath)
            items.append(
                DictionaryInfo(
                    filename=fpath.name,
                    entry_count=len(d),
                    size_bytes=fpath.stat().st_size,
                )
            )

    return DictionariesResponse(
        enabled=settings.query_processing.dictionaries_enabled,
        dictionaries=items,
    )


@router.post("/dictionaries/upload")
async def upload_dictionary(
    request: Request,
    file: UploadFile,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Accept an uploaded ``.json`` or ``.yaml`` dictionary file and save it
    into ``dictionaries_dir``, then rebuild the in-memory term expander so it
    takes effect immediately (no restart needed)."""
    filename = file.filename or "uploaded_dict.json"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise HTTPException(status_code=400, detail="Only .json / .yaml files are supported")

    dir_path = Path(settings.query_processing.dictionaries_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    dest = dir_path / filename
    content = await file.read()
    dest.write_bytes(content)

    # Rebuild the in-memory expander so the new dictionary takes effect
    _rebuild_expander(request, settings)

    return {"uploaded": filename, "size_bytes": len(content)}


@router.delete("/dictionaries/{filename}")
def remove_dictionary(
    request: Request,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Delete a dictionary file from ``dictionaries_dir`` and rebuild the
    in-memory term expander."""
    dir_path = Path(settings.query_processing.dictionaries_dir)
    target = dir_path / filename

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Dictionary {filename!r} not found")

    target.unlink()
    _rebuild_expander(request, settings)

    return {"deleted": filename}


@router.post("/dictionaries/toggle")
def toggle_dictionaries(
    request: Request,
    body: DictionaryToggleRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Enable or disable dictionary-based query expansion at runtime."""
    settings.query_processing.dictionaries_enabled = body.enabled
    _rebuild_expander(request, settings)
    return {"dictionaries_enabled": body.enabled}
