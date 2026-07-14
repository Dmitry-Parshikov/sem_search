from __future__ import annotations

from app.chunking.paragraph import ParagraphChunker
from app.core.types import Document


def make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, text=text, source="test")


def test_paragraph_splitting_basic():
    text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
    chunker = ParagraphChunker(min_chars=0)
    chunks = chunker.chunk(make_doc(text))
    assert [c.text for c in chunks] == ["Первый абзац.", "Второй абзац.", "Третий абзац."]
    assert [c.position for c in chunks] == [0, 1, 2]


def test_paragraph_splitting_multiple_blank_lines():
    text = "Абзац раз.\n\n\n\nАбзац два."
    chunker = ParagraphChunker(min_chars=0)
    chunks = chunker.chunk(make_doc(text))
    assert [c.text for c in chunks] == ["Абзац раз.", "Абзац два."]


def test_min_chars_filters_short_paragraphs():
    text = "Ок.\n\nЭто достаточно длинный абзац для прохождения фильтра.\n\nНет."
    chunker = ParagraphChunker(min_chars=20)
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == "Это достаточно длинный абзац для прохождения фильтра."


def test_empty_document_produces_no_chunks():
    chunker = ParagraphChunker(min_chars=0)
    chunks = chunker.chunk(make_doc("   \n\n  "))
    assert chunks == []


def test_config_signature_stability():
    a = ParagraphChunker(min_chars=5)
    b = ParagraphChunker(min_chars=5)
    c = ParagraphChunker(min_chars=10)
    assert a.config_signature == b.config_signature
    assert a.config_signature != c.config_signature
