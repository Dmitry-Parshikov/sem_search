from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from app.core.types import Chunk, RetrievedCandidate, Vector


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, name: str, dimension: int, distance: Literal["cosine"] = "cosine") -> None:
        ...

    @abstractmethod
    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: Vector,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...
