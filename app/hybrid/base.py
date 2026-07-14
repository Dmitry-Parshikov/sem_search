from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.types import RetrievedCandidate


class Hybridizer(ABC):
    @abstractmethod
    def fuse(self, dense: list[RetrievedCandidate], lexical: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
        ...
