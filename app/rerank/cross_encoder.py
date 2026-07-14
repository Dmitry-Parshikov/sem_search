"""Ф3.5 cross-encoder reranking: `sentence_transformers.CrossEncoder` scores
`(query, candidate.text)` pairs jointly (query and document attend to each
other, unlike the independently-embedded dense retrieval step), which is
usually more accurate than either dense cosine similarity or BM25 alone --
but expensive, hence bounded to a small top_n rather than the whole
collection (NFR "Производительность": "реранжирование -- только по
ограниченному топ-N, никогда по всей коллекции").

Per the plan's risk #4, dev (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
and final (`BAAI/bge-reranker-v2-m3`) reranker models are not guaranteed to
return scores on the same scale (e.g. [0, 1] vs. unbounded logits) -- so this
class treats `.predict()`'s output as "higher is better" and nothing more:
no normalization/clamping is applied, only a descending sort. That
interpretation is encapsulated entirely here, so swapping the configured
model name needs no changes anywhere else.
"""

from __future__ import annotations

from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.core.types import RetrievedCandidate
from app.rerank.base import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[RetrievedCandidate], top_n: int
    ) -> list[RetrievedCandidate]:
        # NFR "Производительность": never score more than `top_n` candidates,
        # regardless of how many were passed in. The (lower-priority) tail
        # beyond top_n is intentionally dropped from the reranked view -- the
        # caller already retrieved a pool >= top_k upstream, and top_n is
        # configured >= top_k, so this never starves the final response.
        bounded = candidates[:top_n]
        if not bounded:
            return []

        pairs = [(query, candidate.text) for candidate in bounded]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        scored = list(zip(bounded, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            replace(candidate, score=float(score), rank=i + 1)
            for i, (candidate, score) in enumerate(scored)
        ]

    @property
    def model_name(self) -> str:
        return self._model_name
