"""`POST /search` (Ф2.1, Ф3.1-Ф3.4, Ф3.5, Ф3.6): dense/bm25/hybrid/
hybrid_rerank retrieval, with must_contain/must_exclude filtering applied in
all four modes (plan decision #2) and cross-encoder reranking (Ф3.5) applied
in `hybrid_rerank` on top of the hybrid-fused, filtered candidates.
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
        typo_suggestion=result.typo_suggestion,
        expanded_query=result.expanded_query,
        warnings=result.warnings,
    )
