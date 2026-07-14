from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.types import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...

    @property
    @abstractmethod
    def config_signature(self) -> str:
        """Stable string encoding strategy+params, folded into the index_version hash."""
        ...
