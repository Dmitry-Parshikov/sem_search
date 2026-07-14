"""`GET /health` (Ф4.1 / NFR "Надёжность"): reports subsystem status and the
active index version. Must never itself fail -- every subsystem check is
individually guarded, so a down subsystem shows up as `false`/degraded
rather than crashing the endpoint.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from app.admin.manifest import IndexManifest
from app.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = getattr(request.app.state, "settings", None)
    subsystems: dict[str, bool | str] = {}

    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        subsystems["vector_store"] = False
    else:
        try:
            subsystems["vector_store"] = bool(vector_store.health())
        except Exception:
            subsystems["vector_store"] = False

    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        subsystems["embedder"] = False
    else:
        try:
            _ = embedder.dimension
            subsystems["embedder"] = True
        except Exception:
            subsystems["embedder"] = False

    # Reranker not wired until Phase 7 -- reported distinctly from a real
    # failure so `/health` consumers can tell "not built yet" apart from
    # "built but broken".
    subsystems["reranker"] = "not_configured"

    index_version: str | None = None
    if settings is not None:
        try:
            manifest = IndexManifest.load(Path(settings.admin.manifest_path))
            index_version = manifest.active_version
        except Exception:
            index_version = None

    checked_booleans = [v for v in subsystems.values() if isinstance(v, bool)]
    status = "ok" if checked_booleans and all(checked_booleans) else "degraded"

    return HealthResponse(status=status, index_version=index_version, subsystems=subsystems)
