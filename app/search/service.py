"""Online search orchestration (Ф2.1-Ф2.5, Ф3.1-Ф3.2, Ф3.6).

Phase 4 wires only the two base retrieval modes (`dense`, `bm25`). `hybrid`
and `hybrid_rerank` are deliberately NOT stubbed with fake behavior here --
they raise `NotImplementedError` until Phase 5 (RRF fusion) and Phase 7
(cross-encoder reranking) build them for real.

`must_contain`/`must_exclude` are accepted end-to-end (schema -> route ->
service) so Phase 5 only has to add the actual filtering call (per plan
decision #2, must_contain/must_exclude apply in ALL modes, wired in
`app.search.filters`) instead of touching every layer again. They are not
applied yet in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.core.types import RetrievedCandidate, SearchHit
from app.embedding.base import Embedder
from app.search.active_index import ActiveIndexResolver
from app.search.retrievers import BM25Retriever, DenseRetriever
from app.vector_store.base import VectorStore

_BASE_MODES = ("dense", "bm25")
_NOT_YET_IMPLEMENTED_MODES = ("hybrid", "hybrid_rerank")
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
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._active_index_resolver = active_index_resolver
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

        # Ф3.4/plan decision #2: must_contain/must_exclude apply in ALL
        # modes, but the filtering step itself (app.search.filters) is
        # Phase 5 work -- accepted here only to keep the signature stable.
        _ = must_contain, must_exclude

        context = self._active_index_resolver.resolve()

        if mode == "dense":
            retriever = DenseRetriever(self._embedder, self._vector_store)
            candidates = retriever.retrieve(query, context.collection_name, top_k)
        elif mode == "bm25":
            retriever = BM25Retriever(context.lexical_index)
            candidates = retriever.retrieve(query, top_k)
        else:
            raise NotImplementedError(
                f"mode {mode!r} is implemented in Phase 5/7 (hybridization / reranking)"
            )

        hits = [_to_hit(candidate) for candidate in candidates]
        return SearchResult(hits=hits, index_version=context.index_version, mode=mode)


def _to_hit(candidate: RetrievedCandidate) -> SearchHit:
    # Highlighting (Ф3.6 "опционально подсвеченные совпадающие термины") is
    # Phase 5's app.search.highlighting -- left empty until then.
    return SearchHit(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        text=candidate.text,
        score=candidate.score,
        metadata=candidate.metadata,
        highlights=[],
    )
