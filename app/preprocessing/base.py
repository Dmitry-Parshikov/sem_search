from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class Preprocessor(ABC):
    @abstractmethod
    def clean(self, raw_text: str, content_type: Literal["txt", "html", "plain"]) -> str:
        ...
