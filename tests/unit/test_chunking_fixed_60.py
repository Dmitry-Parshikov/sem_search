"""Unit tests for the `fixed_60` chunking strategy (task 1): a relative
fixed-window whose size is ~60% of the model's context limit with proportional
overlap, derived by `Fixed60Config` and built via the chunking factory."""

from __future__ import annotations

from app.chunking.factory import build_chunker
from app.chunking.fixed_window import FixedWindowChunker
from app.config import ChunkingConfig, Fixed60Config
from app.core.types import Document


def test_fixed_60_derives_chunk_size_and_overlap_from_context_limit():
    cfg = Fixed60Config(context_limit=512, window_ratio=0.6, overlap_ratio=0.2)

    assert cfg.chunk_size == 307  # int(512 * 0.6)
    assert cfg.overlap == 61  # int(307 * 0.2)


def test_fixed_60_overlap_never_reaches_chunk_size():
    cfg = Fixed60Config(context_limit=10, window_ratio=1.0, overlap_ratio=1.0)

    assert cfg.overlap == cfg.chunk_size - 1


def test_build_chunker_supports_fixed_60_strategy():
    cfg = ChunkingConfig(strategy="fixed_60", fixed_60=Fixed60Config(context_limit=512))

    chunker = build_chunker(cfg)

    assert isinstance(chunker, FixedWindowChunker)


def test_fixed_60_chunker_produces_overlapping_windows():
    cfg = ChunkingConfig(
        strategy="fixed_60",
        fixed_60=Fixed60Config(context_limit=10, window_ratio=0.6, overlap_ratio=0.2),
    )
    chunker = build_chunker(cfg)
    # chunk_size = int(10 * 0.6) = 6, overlap = int(6 * 0.2) = 1, stride = 5.
    text = " ".join(f"w{i}" for i in range(12))
    doc = Document(doc_id="d1", text=text, source="test")

    chunks = chunker.chunk(doc)

    assert chunks[0].text.split() == [f"w{i}" for i in range(6)]
    # stride 5 -> second window starts at w5.
    assert chunks[1].text.split()[0] == "w5"
