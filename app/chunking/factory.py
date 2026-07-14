from __future__ import annotations

from app.chunking.base import Chunker
from app.chunking.fixed_window import FixedWindowChunker
from app.chunking.paragraph import ParagraphChunker
from app.chunking.sentence_window import SentenceWindowChunker
from app.config import ChunkingConfig


def build_chunker(cfg: ChunkingConfig) -> Chunker:
    params = cfg.active_params()
    if cfg.strategy == "fixed_window":
        return FixedWindowChunker(
            chunk_size=params.chunk_size,
            overlap=params.overlap,
            unit=params.unit,
        )
    if cfg.strategy == "sentence_window":
        return SentenceWindowChunker(
            window_sentences=params.window_sentences,
            stride_sentences=params.stride_sentences,
        )
    if cfg.strategy == "paragraph":
        return ParagraphChunker(min_chars=params.min_chars)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")
