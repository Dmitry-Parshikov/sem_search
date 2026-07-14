from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.types import Vector


class Embedder(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[Vector]:
        ...

    @abstractmethod
    def encode_query(self, text: str) -> Vector:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
