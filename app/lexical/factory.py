from __future__ import annotations

from app.config import LexicalConfig
from app.lexical.base import LexicalIndex
from app.lexical.bm25_index import BM25LexicalIndex
from app.lexical.tokenizer import RussianTokenizer


def build_lexical_index(cfg: LexicalConfig) -> LexicalIndex:
    """Builds a FRESH `LexicalIndex` instance.

    Each indexing run needs its own instance (not a shared singleton) since
    `.build()` fully replaces internal state -- callers (e.g.
    `IndexingService`) should call this factory anew per run rather than
    reusing one instance across reindexes.
    """

    tokenizer = RussianTokenizer(use_lemmatization=cfg.use_lemmatization)
    return BM25LexicalIndex(k1=cfg.bm25.k1, b=cfg.bm25.b, tokenizer=tokenizer)
