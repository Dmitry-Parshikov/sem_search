from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.types import Chunk, RetrievedCandidate


class LexicalIndex(ABC):
    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """Always a full rebuild, never an incremental patch (reproducibility)."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        ...

    @abstractmethod
    def contains_all(self, chunk_id: str, terms: list[str]) -> bool:
        ...

    @abstractmethod
    def contains_any(self, chunk_id: str, terms: list[str]) -> bool:
        ...

    @abstractmethod
    def vocabulary(self) -> set[str]:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...
