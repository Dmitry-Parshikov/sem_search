"""`POST /search` (Ф2.1, Ф3.1, Ф3.2, Ф3.6): dense/bm25 retrieval in Phase 4.

`hybrid`/`hybrid_rerank` are real, selectable modes per the API contract
(Ф4.3), but their implementation is Phase 5 (RRF fusion) / Phase 7
(cross-encoder rerank) work -- requesting them now returns 501, not a fake
result, so callers can tell "not built yet" apart from a real failure.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import SearchHitOut, SearchRequest, SearchResponse
from app.config import Settings
from app.core.errors import NoActiveIndexError
from app.dependencies import get_search_service, get_settings
from app.search.service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    mode = body.mode or settings.search.default_mode
    top_k = body.top_k if body.top_k is not None else settings.search.default_top_k

    try:
        result = search_service.search(
            query=body.query,
            mode=mode,
            top_k=top_k,
            must_contain=body.must_contain,
            must_exclude=body.must_exclude,
        )
    except NoActiveIndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return SearchResponse(
        hits=[SearchHitOut(**hit.__dict__) for hit in result.hits],
        index_version=result.index_version,
        mode=result.mode,
        query=body.query,
    )
