from __future__ import annotations

import re

from app.chunking.base import Chunker
from app.core.ids import make_chunk_id
from app.core.types import Chunk, Document

# Blank-line separated paragraphs (one or more empty lines, allowing
# trailing whitespace on the blank line).
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def _doc_metadata(document: Document) -> dict:
    return {
        "source": document.source,
        "section": document.section,
        "date": document.date,
        **document.extra,
    }


class ParagraphChunker(Chunker):
    """Splits on blank lines; paragraphs shorter than `min_chars` are
    dropped (Ф1.2)."""

    def __init__(self, min_chars: int = 0) -> None:
        if min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        self._min_chars = min_chars

    @property
    def config_signature(self) -> str:
        return f"paragraph|min_chars={self._min_chars}"

    def chunk(self, document: Document) -> list[Chunk]:
        raw_paragraphs = _PARAGRAPH_SPLIT_RE.split(document.text.strip())
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        paragraphs = [p for p in paragraphs if len(p) >= self._min_chars]

        metadata = _doc_metadata(document)
        chunks: list[Chunk] = []
        for position, text in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, position),
                    doc_id=document.doc_id,
                    text=text,
                    position=position,
                    metadata=dict(metadata),
                )
            )
        return chunks
