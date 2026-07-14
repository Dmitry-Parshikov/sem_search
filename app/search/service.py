"""Online search orchestration (Ф2.1-Ф2.5, Ф3.1-Ф3.4, Ф3.6).

Phase 5 wires the real `hybrid` mode (RRF/weighted fusion via the injected
`Hybridizer`) and applies must_contain/must_exclude filtering
(`app.search.filters`) in EVERY mode -- dense, bm25, and hybrid alike -- per
plan decision #2: it is a correctness constraint, not something specific to
hybrid retrieval. `hybrid_rerank` still raises `NotImplementedError` until
Phase 7 adds cross-encoder reranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.core.types import RetrievedCandidate, SearchHit
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.search.active_index import ActiveIndexResolver
from app.search.filters import apply_must_contain_exclude
from app.search.retrievers import BM25Retriever, DenseRetriever
from app.vector_store.base import VectorStore

_BASE_MODES = ("dense", "bm25", "hybrid")
_NOT_YET_IMPLEMENTED_MODES = ("hybrid_rerank",)
_VALID_MODES = _BASE_MODES + _NOT_YET_IMPLEMENTED_MODES


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    index_version: str = ""
    mode: str = ""


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        active_index_resolver: ActiveIndexResolver,
        hybridizer: Hybridizer,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._active_index_resolver = active_index_resolver
        self._hybridizer = hybridizer
        self._settings = settings

    def search(
        self,
        query: str,
        mode: str,
        top_k: int,
        must_contain: list[str] | None = None,
        must_exclude: list[str] | None = None,
    ) -> SearchResult:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown search mode {mode!r}; expected one of {_VALID_MODES}")

        context = self._active_index_resolver.resolve()

        # Pool size: how many candidates each retriever pulls BEFORE
        # must_contain/must_exclude filtering (and, for hybrid, fusion).
        # Filtering can only shrink the candidate set, never grow it -- if we
        # only ever retrieved `top_k` candidates and then filtered, a query
        # with must_contain/must_exclude could come back with fewer than
        # `top_k` hits even when enough matching chunks exist further down
        # the underlying ranking. Retrieving `max(candidate_k, top_k)` up
        # front (config's `retrieval.candidate_k`, separate from the
        # response's `top_k` per the plan) gives filtering room to work with
        # before the final truncation below. This is a deliberate
        # implementation choice, not spelled out verbatim in the spec.
        pool_size = max(self._settings.retrieval.candidate_k, top_k)

        if mode == "dense":
            retriever = DenseRetriever(self._embedder, self._vector_store)
            candidates = retriever.retrieve(query, context.collection_name, pool_size)
        elif mode == "bm25":
            retriever = BM25Retriever(context.lexical_index)
            candidates = retriever.retrieve(query, pool_size)
        elif mode == "hybrid":
            dense_retriever = DenseRetriever(self._embedder, self._vector_store)
            bm25_retriever = BM25Retriever(context.lexical_index)
            dense_candidates = dense_retriever.retrieve(query, context.collection_name, pool_size)
            lexical_candidates = bm25_retriever.retrieve(query, pool_size)
            candidates = self._hybridizer.fuse(dense_candidates, lexical_candidates)
        else:
            raise NotImplementedError(
                f"mode {mode!r} is implemented in Phase 7 (cross-encoder reranking)"
            )

        # Ф3.4/plan decision #2: applied identically regardless of which mode
        # produced `candidates` -- this is what makes must_contain/exclude
        # work even in dense-only mode, not just hybrid.
        filtered = apply_must_contain_exclude(
            candidates,
            must_contain or [],
            must_exclude or [],
            context.lexical_index,
        )

        # Truncate to the caller's requested top_k. `SearchHit` carries no
        # `rank` field (informational rank isn't consumed downstream yet), so
        # there's nothing to renumber here -- the pre-truncation order from
        # the retriever/fusion step is preserved as-is.
        truncated = filtered[:top_k]
        hits = [_to_hit(candidate) for candidate in truncated]
        return SearchResult(hits=hits, index_version=context.index_version, mode=mode)


def _to_hit(candidate: RetrievedCandidate) -> SearchHit:
    # Highlighting (Ф3.6 "опционально подсвеченные совпадающие термины") is
    # not built yet -- left empty until a later phase.
    return SearchHit(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        text=candidate.text,
        score=candidate.score,
        metadata=candidate.metadata,
        highlights=[],
    )
