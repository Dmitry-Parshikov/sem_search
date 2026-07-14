from __future__ import annotations

from typing import Literal

from app.chunking.base import Chunker
from app.core.ids import make_chunk_id
from app.core.types import Chunk, Document


def _doc_metadata(document: Document) -> dict:
    return {
        "source": document.source,
        "section": document.section,
        "date": document.date,
        **document.extra,
    }


class FixedWindowChunker(Chunker):
    """Sliding window over the document text, by token count or char count,
    with configurable overlap (Ф1.2)."""

    def __init__(self, chunk_size: int, overlap: int, unit: Literal["tokens", "chars"] = "tokens") -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._unit = unit

    @property
    def config_signature(self) -> str:
        return f"fixed_window|chunk_size={self._chunk_size}|overlap={self._overlap}|unit={self._unit}"

    def chunk(self, document: Document) -> list[Chunk]:
        if self._unit == "tokens":
            units = document.text.split()
        else:
            units = list(document.text)

        if not units:
            return []

        stride = self._chunk_size - self._overlap
        metadata = _doc_metadata(document)

        chunks: list[Chunk] = []
        position = 0
        start = 0
        n = len(units)
        while start < n:
            end = min(start + self._chunk_size, n)
            window = units[start:end]
            text = " ".join(window) if self._unit == "tokens" else "".join(window)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, position),
                    doc_id=document.doc_id,
                    text=text,
                    position=position,
                    metadata=dict(metadata),
                )
            )
            position += 1
            if end == n:
                break
            start += stride

        return chunks
