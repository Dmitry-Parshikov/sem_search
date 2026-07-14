from __future__ import annotations

from app.chunking.sentence_window import SentenceWindowChunker
from app.core.types import Document


def make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, text=text, source="test")


def test_window_and_stride_correctness_multi_sentence():
    text = "Первое предложение. Второе предложение! Третье предложение? Четвёртое предложение."
    chunker = SentenceWindowChunker(window_sentences=2, stride_sentences=1)
    chunks = chunker.chunk(make_doc(text))

    assert len(chunks) == 3
    assert chunks[0].text == "Первое предложение. Второе предложение!"
    assert chunks[1].text == "Второе предложение! Третье предложение?"
    assert chunks[2].text == "Третье предложение? Четвёртое предложение."
    assert [c.position for c in chunks] == [0, 1, 2]


def test_stride_equal_to_window_no_overlap():
    text = "Раз. Два. Три. Четыре."
    chunker = SentenceWindowChunker(window_sentences=2, stride_sentences=2)
    chunks = chunker.chunk(make_doc(text))
    assert [c.text for c in chunks] == ["Раз. Два.", "Три. Четыре."]


def test_single_sentence_document():
    text = "Единственное предложение в документе."
    chunker = SentenceWindowChunker(window_sentences=3, stride_sentences=2)
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_document_produces_no_chunks():
    chunker = SentenceWindowChunker(window_sentences=3, stride_sentences=2)
    chunks = chunker.chunk(make_doc("   "))
    assert chunks == []


def test_config_signature_stability():
    a = SentenceWindowChunker(window_sentences=3, stride_sentences=2)
    b = SentenceWindowChunker(window_sentences=3, stride_sentences=2)
    c = SentenceWindowChunker(window_sentences=4, stride_sentences=2)
    assert a.config_signature == b.config_signature
    assert a.config_signature != c.config_signature


def test_chunk_ids_deterministic():
    text = "Раз. Два. Три."
    chunker = SentenceWindowChunker(window_sentences=1, stride_sentences=1)
    chunks = chunker.chunk(make_doc(text, doc_id="d42"))
    assert [c.chunk_id for c in chunks] == ["d42::0000", "d42::0001", "d42::0002"]
