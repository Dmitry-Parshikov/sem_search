from __future__ import annotations

from app.chunking.base import Chunker
from app.chunking.sentence_split import split_sentences
from app.core.ids import make_chunk_id
from app.core.types import Chunk, Document


def _doc_metadata(document: Document) -> dict:
    return {
        "source": document.source,
        "section": document.section,
        "date": document.date,
        **document.extra,
    }


class SentenceWindowChunker(Chunker):
    """Groups `window_sentences` sentences per chunk, sliding by
    `stride_sentences` (Ф1.2)."""

    def __init__(self, window_sentences: int, stride_sentences: int) -> None:
        if window_sentences <= 0:
            raise ValueError("window_sentences must be positive")
        if stride_sentences <= 0:
            raise ValueError("stride_sentences must be positive")
        self._window = window_sentences
        self._stride = stride_sentences

    @property
    def config_signature(self) -> str:
        return f"sentence_window|window_sentences={self._window}|stride_sentences={self._stride}"

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        metadata = _doc_metadata(document)
        chunks: list[Chunk] = []
        position = 0
        start = 0
        n = len(sentences)
        while start < n:
            end = min(start + self._window, n)
            text = " ".join(sentences[start:end])
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
            start += self._stride

        return chunks
