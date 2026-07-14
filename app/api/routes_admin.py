"""`GET /admin/versions`, `POST /admin/rollback/{version}` (Ф4.1): index
version listing and non-destructive rollback. Covers the "просмотр статуса
индекса ... откат к предыдущей версии" requirement.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.admin.service import AdminService
from app.api.schemas import RollbackResponse, VersionInfo, VersionsResponse
from app.core.errors import IndexVersionAssetsMissingError, IndexVersionNotFoundError
from app.dependencies import get_admin_service

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
