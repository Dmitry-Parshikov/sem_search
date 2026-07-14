from __future__ import annotations

import pytest

from app.chunking.fixed_window import FixedWindowChunker
from app.core.types import Document


def make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, text=text, source="test", section="s1", date="2024-01-01", extra={"foo": "bar"})


def test_token_mode_basic_windowing():
    text = " ".join(f"w{i}" for i in range(10))  # w0 w1 ... w9
    chunker = FixedWindowChunker(chunk_size=4, overlap=1, unit="tokens")
    chunks = chunker.chunk(make_doc(text))

    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]
    assert [c.position for c in chunks] == [0, 1, 2]
    assert [c.chunk_id for c in chunks] == ["doc1::0000", "doc1::0001", "doc1::0002"]


def test_char_mode_basic_windowing():
    text = "abcdefghij"
    chunker = FixedWindowChunker(chunk_size=4, overlap=2, unit="chars")
    chunks = chunker.chunk(make_doc(text))

    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij"]


def test_overlap_correctness_tokens():
    text = " ".join(str(i) for i in range(6))
    chunker = FixedWindowChunker(chunk_size=3, overlap=1, unit="tokens")
    chunks = chunker.chunk(make_doc(text))
    # stride = chunk_size - overlap = 2
    assert [c.text for c in chunks] == ["0 1 2", "2 3 4", "4 5"]


def test_doc_shorter_than_chunk_size_produces_single_chunk():
    text = "short doc only"
    chunker = FixedWindowChunker(chunk_size=100, overlap=10, unit="tokens")
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_document_produces_no_chunks():
    chunker = FixedWindowChunker(chunk_size=10, overlap=2, unit="tokens")
    chunks = chunker.chunk(make_doc(""))
    assert chunks == []


def test_metadata_copied_from_document():
    chunker = FixedWindowChunker(chunk_size=10, overlap=2, unit="tokens")
    chunks = chunker.chunk(make_doc("some text here"))
    assert chunks[0].metadata == {
        "source": "test",
        "section": "s1",
        "date": "2024-01-01",
        "foo": "bar",
    }
    assert chunks[0].doc_id == "doc1"


def test_config_signature_stable_for_same_params():
    a = FixedWindowChunker(chunk_size=10, overlap=2, unit="tokens")
    b = FixedWindowChunker(chunk_size=10, overlap=2, unit="tokens")
    assert a.config_signature == b.config_signature


def test_config_signature_differs_for_different_params():
    a = FixedWindowChunker(chunk_size=10, overlap=2, unit="tokens")
    b = FixedWindowChunker(chunk_size=20, overlap=2, unit="tokens")
    c = FixedWindowChunker(chunk_size=10, overlap=4, unit="tokens")
    d = FixedWindowChunker(chunk_size=10, overlap=2, unit="chars")
    signatures = {a.config_signature, b.config_signature, c.config_signature, d.config_signature}
    assert len(signatures) == 4


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        FixedWindowChunker(chunk_size=10, overlap=10, unit="tokens")
    with pytest.raises(ValueError):
        FixedWindowChunker(chunk_size=10, overlap=-1, unit="tokens")
