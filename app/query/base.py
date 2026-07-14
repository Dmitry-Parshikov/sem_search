from __future__ import annotations

from abc import ABC, abstractmethod


class TypoCorrector(ABC):
    @abstractmethod
    def suggest(self, query: str, vocabulary: set[str]) -> str | None:
        """Returns a suggested correction, or None if the query looks fine."""
        ...


class TermExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> str:
        ...
