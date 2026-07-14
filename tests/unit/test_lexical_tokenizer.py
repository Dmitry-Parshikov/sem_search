from __future__ import annotations

from app.lexical.tokenizer import RussianTokenizer


def test_lemmatization_on_produces_normal_forms():
    tokenizer = RussianTokenizer(use_lemmatization=True)
    tokens = tokenizer.tokenize("В домах жили кошки")
    assert "дом" in tokens
    assert "кошка" in tokens


def test_lemmatization_off_keeps_surface_forms():
    tokenizer = RussianTokenizer(use_lemmatization=False)
    tokens = tokenizer.tokenize("В домах жили кошки")
    assert "домах" in tokens
    assert "кошки" in tokens
    assert "дом" not in tokens


def test_lemmatization_on_vs_off_differ_on_inflected_text():
    lemma_tokenizer = RussianTokenizer(use_lemmatization=True)
    literal_tokenizer = RussianTokenizer(use_lemmatization=False)
    text = "документов много документами"
    assert set(lemma_tokenizer.tokenize(text)) != set(literal_tokenizer.tokenize(text))


def test_mixed_ru_en_tokenization():
    tokenizer = RussianTokenizer(use_lemmatization=False)
    tokens = tokenizer.tokenize("IT-термин API используется в REST")
    assert "it" in tokens
    assert "термин" in tokens
    assert "api" in tokens
    assert "rest" in tokens


def test_punctuation_stripped():
    tokenizer = RussianTokenizer(use_lemmatization=False)
    tokens = tokenizer.tokenize("Привет, мир! Как дела?..")
    assert tokens == ["привет", "мир", "как", "дела"]


def test_empty_text_produces_no_tokens():
    tokenizer = RussianTokenizer(use_lemmatization=True)
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ...,,, ") == []
