from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.types import RetrievedCandidate


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedCandidate], top_n: int) -> list[RetrievedCandidate]:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
