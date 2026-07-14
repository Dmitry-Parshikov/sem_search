"""Loads a real sentence-transformers model (the DEV embedder) -- slow.

Run explicitly with `pytest -m slow` or the full `pytest` invocation; the
fast dev loop uses `pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DEV_EMBEDDING_MODEL
from app.embedding.st_embedder import SentenceTransformerEmbedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=DEV_EMBEDDING_MODEL,
        device="cpu",
        batch_size=8,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )


def test_dimension_matches_actual_vectors(embedder: SentenceTransformerEmbedder):
    vectors = embedder.encode_documents(["тестовый документ про кошек"])
    assert len(vectors[0]) == embedder.dimension


def test_query_and_document_prefixing_differ(embedder: SentenceTransformerEmbedder):
    text = "семантический поиск"
    doc_vec = embedder.encode_documents([text])[0]
    query_vec = embedder.encode_query(text)
    # Different prefixes ("query: " vs "passage: ") should produce different
    # embeddings for the same underlying text.
    assert not np.allclose(doc_vec, query_vec)


def test_batch_vs_looped_single_calls_match(embedder: SentenceTransformerEmbedder):
    texts = [
        "Первый документ про право.",
        "Второй документ про IT-термины.",
        "Третий документ про новости.",
    ]
    batch_vectors = embedder.encode_documents(texts)
    looped_vectors = [embedder.encode_documents([t])[0] for t in texts]

    for batch_vec, looped_vec in zip(batch_vectors, looped_vectors):
        assert np.allclose(batch_vec, looped_vec, atol=1e-5)


def test_encode_order_preserved_regardless_of_input_order():
    """sentence-transformers sorts internally by length for efficiency but
    must restore original order -- verify this holds for the pinned version
    rather than assuming it."""

    embedder = SentenceTransformerEmbedder(
        model_name=DEV_EMBEDDING_MODEL,
        device="cpu",
        batch_size=8,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    texts = [
        "короткий",
        "средней длины предложение про документы",
        "очень длинное предложение с большим количеством слов про семантический поиск и индексацию",
        "а",
    ]
    reversed_texts = list(reversed(texts))

    forward_vectors = embedder.encode_documents(texts)
    reversed_vectors = embedder.encode_documents(reversed_texts)

    for i, text in enumerate(texts):
        j = reversed_texts.index(text)
        assert np.allclose(forward_vectors[i], reversed_vectors[j], atol=1e-5)
