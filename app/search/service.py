"""Online search orchestration (Ф2.1-Ф2.5, Ф3.1-Ф3.4, Ф3.6).

Phase 5 wires the real `hybrid` mode (RRF/weighted fusion via the injected
`Hybridizer`) and applies must_contain/must_exclude filtering
(`app.search.filters`) in EVERY mode -- dense, bm25, and hybrid alike -- per
plan decision #2: it is a correctness constraint, not something specific to
hybrid retrieval.

Phase 6 adds the two optional query-processing steps from the architecture
diagram (Ф2.2 typo correction, Ф2.4 term-dictionary expansion), run at the
very start of `.search()`, before any retrieval. Both are wired as `None`-able
collaborators (`TypoCorrector | None`, `TermExpander | None`) and wrapped in
`try/except Exception` -- per the NFR "Надёжность", failure of either must
degrade to the base (unmodified-query) behavior plus a logged + reported
warning, never fail the request.

Phase 7 adds Ф3.5 cross-encoder reranking and the real `hybrid_rerank` mode:
retrieve+fuse+filter exactly like `hybrid` (factored into
`_hybrid_candidates` below to avoid duplicating that block), then -- if a
`Reranker` is configured -- rerank the filtered candidates. Reranking is
itself an optional stage per the NFR "Надёжность", so a failure there is
caught the same way as typo correction/term expansion: log a warning, append
a warning to the response, and fall back to the pre-rerank (hybrid-fused,
filtered) ordering rather than failing the request.

Phase 8 adds Ф4.2 structured query logging: every successful `.search()` call
records one JSONL entry (query, mode, top_k, must_contain/exclude,
index_version, response time, warnings) via the injected `QueryLogger`.
Unlike the optional query-processing stages, logging isn't config-gated --
it always runs -- but a logging failure is still wrapped in try/except so it
can never break a search response (same reliability posture as those stages).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from app.admin.query_log import QueryLogger
from app.config import Settings
from app.core.types import RetrievedCandidate, SearchHit
from app.embedding.base import Embedder
from app.hybrid.base import Hybridizer
from app.query.base import TermExpander, TypoCorrector
from app.rerank.base import Reranker
from app.search.active_index import ActiveIndexContext, ActiveIndexResolver
from app.search.filters import apply_must_contain_exclude
from app.search.retrievers import BM25Retriever, DenseRetriever
from app.vector_store.base import VectorStore

logger = structlog.get_logger(__name__)

_VALID_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    index_version: str = ""
    mode: str = ""
    # Ф2.2: a suggested correction for the ORIGINAL `query`, or None if typo
    # correction is disabled/found nothing to fix. Never used as the query
    # actually searched.
    typo_suggestion: str | None = None
    # Ф2.4: the expanded query actually used for retrieval, only set when it
    # differs from the original `query` (transparency for the API response).
    expanded_query: str | None = None
    # NFR "Надёжность": populated when an optional stage (typo correction,
    # term expansion) failed and was skipped -- the request still succeeds.
    warnings: list[str] = field(default_factory=list)


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        active_index_resolver: ActiveIndexResolver,
        hybridizer: Hybridizer,
        settings: Settings,
        query_logger: QueryLogger,
        typo_corrector: TypoCorrector | None = None,
        term_expander: TermExpander | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._active_index_resolver = active_index_resolver
        self._hybridizer = hybridizer
        self._settings = settings
        self._query_logger = query_logger
        self._typo_corrector = typo_corrector
        self._term_expander = term_expander
        self._reranker = reranker

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

        start_time = time.perf_counter()

        context = self._active_index_resolver.resolve()

        warnings: list[str] = []

        # Ф2.2: typo suggestion is computed against the active index's
        # vocabulary and returned to the caller as a hint only -- it must
        # NEVER replace `query` for actual retrieval (see
        # `app.query.typo_correction` module docstring: "не блокируя
        # выполнение исходного запроса"). Any failure degrades to "no
        # suggestion" plus a warning, never a failed request.
        typo_suggestion: str | None = None
        if self._typo_corrector is not None:
            try:
                typo_suggestion = self._typo_corrector.suggest(
                    query, context.lexical_index.vocabulary()
                )
            except Exception as exc:  # noqa: BLE001 -- optional stage must never fail the request
                logger.warning("typo_correction_failed", query=query, error=str(exc))
                typo_suggestion = None
                warnings.append(f"Typo correction failed and was skipped: {exc}")

        # Ф2.4: expansion happens before the embedder/BM25 retrieval, per the
        # architecture diagram -- `effective_query` (not the raw `query`) is
        # what actually goes to the retrievers below. Same
        # graceful-degradation contract: on failure, fall back to the
        # original, unmodified query.
        effective_query = query
        expanded_query: str | None = None
        if self._term_expander is not None:
            try:
                expanded = self._term_expander.expand(query)
                if expanded != query:
                    effective_query = expanded
                    expanded_query = expanded
            except Exception as exc:  # noqa: BLE001 -- same rationale as above
                logger.warning("term_expansion_failed", query=query, error=str(exc))
                effective_query = query
                warnings.append(f"Term expansion failed and was skipped: {exc}")

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
            candidates = retriever.retrieve(effective_query, context.collection_name, pool_size)
            filtered = apply_must_contain_exclude(
                candidates, must_contain or [], must_exclude or [], context.lexical_index
            )
        elif mode == "bm25":
            retriever = BM25Retriever(context.lexical_index)
            candidates = retriever.retrieve(effective_query, pool_size)
            filtered = apply_must_contain_exclude(
                candidates, must_contain or [], must_exclude or [], context.lexical_index
            )
        elif mode == "hybrid":
            filtered = self._hybrid_candidates(
                effective_query, context, pool_size, must_contain, must_exclude
            )
        else:  # mode == "hybrid_rerank"
            filtered = self._hybrid_candidates(
                effective_query, context, pool_size, must_contain, must_exclude
            )
            if self._reranker is not None:
                try:
                    filtered = self._reranker.rerank(
                        effective_query, filtered, top_n=self._settings.reranking.top_n
                    )
                except Exception as exc:  # noqa: BLE001 -- optional stage must never fail the request
                    logger.warning("reranking_failed", query=effective_query, error=str(exc))
                    warnings.append(f"Reranker failed and was skipped: {exc}")
                    # `filtered` is left as-is: the pre-rerank, hybrid-fused
                    # and filtered candidate list -- i.e. degrade to plain
                    # `hybrid` behavior for this request.

        # Truncate to the caller's requested top_k. `SearchHit` carries no
        # `rank` field (informational rank isn't consumed downstream yet), so
        # there's nothing to renumber here -- the pre-truncation order from
        # the retriever/fusion/rerank step is preserved as-is.
        truncated = filtered[:top_k]
        hits = [_to_hit(candidate) for candidate in truncated]
        response_time_ms = (time.perf_counter() - start_time) * 1000

        result = SearchResult(
            hits=hits,
            index_version=context.index_version,
            mode=mode,
            typo_suggestion=typo_suggestion,
            expanded_query=expanded_query,
            warnings=warnings,
        )

        try:
            self._query_logger.log(
                {
                    "query": query,
                    "mode": mode,
                    "top_k": top_k,
                    "must_contain": must_contain or [],
                    "must_exclude": must_exclude or [],
                    "index_version": context.index_version,
                    "response_time_ms": response_time_ms,
                    "warnings": warnings,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- logging must never break a search response
            logger.warning("query_log_write_failed", query=query, error=str(exc))

        return result

    def _hybrid_candidates(
        self,
        effective_query: str,
        context: ActiveIndexContext,
        pool_size: int,
        must_contain: list[str] | None,
        must_exclude: list[str] | None,
    ) -> list[RetrievedCandidate]:
        """Shared by `hybrid` and `hybrid_rerank`: retrieve from both
        retrievers, fuse (Ф3.3), then apply the strict must_contain/exclude
        filter (Ф3.4/plan decision #2). `hybrid_rerank` adds cross-encoder
        reranking on top of this same, already-filtered list."""

        dense_retriever = DenseRetriever(self._embedder, self._vector_store)
        bm25_retriever = BM25Retriever(context.lexical_index)
        dense_candidates = dense_retriever.retrieve(effective_query, context.collection_name, pool_size)
        lexical_candidates = bm25_retriever.retrieve(effective_query, pool_size)
        candidates = self._hybridizer.fuse(dense_candidates, lexical_candidates)
        return apply_must_contain_exclude(
            candidates, must_contain or [], must_exclude or [], context.lexical_index
        )


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
