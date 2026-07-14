from __future__ import annotations

import unicodedata

from app.preprocessing.loaders import TextPreprocessor


def test_html_tags_stripped():
    pp = TextPreprocessor()
    raw = "<html><body><p>Привет, <b>мир</b>!</p><script>alert(1)</script></body></html>"
    cleaned = pp.clean(raw, "html")
    assert "<" not in cleaned and ">" not in cleaned
    assert "alert" not in cleaned
    assert "Привет" in cleaned and "мир" in cleaned


def test_whitespace_collapsed():
    pp = TextPreprocessor()
    raw = "Привет,\n\n   мир!\t\tКак   дела?"
    cleaned = pp.clean(raw, "plain")
    assert "  " not in cleaned
    assert "\n" not in cleaned and "\t" not in cleaned
    assert cleaned == cleaned.strip()


def test_whitespace_collapsed_txt_mode():
    pp = TextPreprocessor()
    raw = "  много   пробелов   \n\n"
    cleaned = pp.clean(raw, "txt")
    assert cleaned == "много пробелов"


def test_nfc_normalization_idempotent():
    pp = TextPreprocessor()
    # Build a decomposed-form (NFD) string and confirm it's genuinely
    # decomposed before checking that .clean() normalizes it to NFC.
    decomposed = unicodedata.normalize("NFD", "текст с буквой йо") + " ещё текст"
    assert not unicodedata.is_normalized("NFC", decomposed)
    cleaned_once = pp.clean(decomposed, "plain")
    cleaned_twice = pp.clean(cleaned_once, "plain")
    assert cleaned_once == cleaned_twice
    assert unicodedata.is_normalized("NFC", cleaned_once)


def test_mixed_ru_en_untouched_by_html_cleaning():
    pp = TextPreprocessor()
    raw = "IT-термин API и REST — обычные слова."
    cleaned = pp.clean(raw, "plain")
    assert "API" in cleaned
    assert "REST" in cleaned
    assert "IT-термин" in cleaned
