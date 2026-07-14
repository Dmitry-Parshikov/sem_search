from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.types import Chunk, RetrievedCandidate
from app.lexical.base import LexicalIndex
from app.lexical.tokenizer import RussianTokenizer


class BM25LexicalIndex(LexicalIndex):
    """`rank_bm25.BM25Okapi` wrapper over chunk texts, tokenized via
    `RussianTokenizer` (Ф1.4).

    `contains_all`/`contains_any` tokenize their `terms` argument with the
    SAME tokenizer used for BM25 scoring, so must_contain/must_exclude
    matching is consistent with (and lemma-normalized the same way as) the
    lexical retrieval branch.
    """

    def __init__(self, k1: float, b: float, tokenizer: RussianTokenizer) -> None:
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_tokens: dict[str, set[str]] = {}
        self._chunk_by_id: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunk_by_id = {c.chunk_id: c for c in chunks}

        tokenized_corpus: list[list[str]] = []
        self._chunk_tokens = {}
        for c in chunks:
            tokens = self._tokenizer.tokenize(c.text)
            tokenized_corpus.append(tokens)
            self._chunk_tokens[c.chunk_id] = set(tokens)

        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = self._tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self._chunk_ids, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        results: list[RetrievedCandidate] = []
        for rank, (chunk_id, score) in enumerate(ranked):
            chunk = self._chunk_by_id[chunk_id]
            results.append(
                RetrievedCandidate(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=dict(chunk.metadata),
                    rank=rank,
                )
            )
        return results

    def contains_all(self, chunk_id: str, terms: list[str]) -> bool:
        tokens = self._chunk_tokens.get(chunk_id)
        if tokens is None:
            return False
        normalized_terms = {t for term in terms for t in self._tokenizer.tokenize(term)}
        return normalized_terms.issubset(tokens)

    def contains_any(self, chunk_id: str, terms: list[str]) -> bool:
        tokens = self._chunk_tokens.get(chunk_id)
        if tokens is None:
            return False
        normalized_terms = {t for term in terms for t in self._tokenizer.tokenize(term)}
        return bool(normalized_terms & tokens)

    def vocabulary(self) -> set[str]:
        vocab: set[str] = set()
        for tokens in self._chunk_tokens.values():
            vocab |= tokens
        return vocab

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self._k1,
            "b": self._b,
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "chunk_tokens": self._chunk_tokens,
            "chunk_by_id": self._chunk_by_id,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        self._k1 = state["k1"]
        self._b = state["b"]
        self._bm25 = state["bm25"]
        self._chunk_ids = state["chunk_ids"]
        self._chunk_tokens = state["chunk_tokens"]
        self._chunk_by_id = state["chunk_by_id"]
