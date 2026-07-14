"""Unit tests for `RapidfuzzTypoCorrector` (Ф2.2).

Fixture vocabulary/misspellings verified directly against `rapidfuzz` scoring
in this environment: "договр" -> "договор" (ratio ~92, edit distance 1),
"оренда" -> "аренда" (ratio ~83, edit distance 1), and "космонавтика" matches
nothing in the vocabulary at all (score_cutoff=80 rejects it outright).
"""

from __future__ import annotations

from app.query.typo_correction import RapidfuzzTypoCorrector

VOCAB = {"договор", "аренда", "недвижимость"}


def _corrector(max_distance: int = 2, score_cutoff: float = 80.0) -> RapidfuzzTypoCorrector:
    return RapidfuzzTypoCorrector(max_distance=max_distance, score_cutoff=score_cutoff)


def test_suggest_corrects_a_misspelled_word_within_max_distance():
    result = _corrector().suggest("договр текст", VOCAB)

    assert result is not None
    assert "договор" in result
    # Untouched word/text stays as typed.
    assert "текст" in result


def test_suggest_corrects_multiple_misspelled_words():
    result = _corrector().suggest("договр оренда", VOCAB)

    assert result is not None
    assert "договор" in result
    assert "аренда" in result


def test_suggest_returns_none_when_all_words_already_in_vocabulary():
    result = _corrector().suggest("договор аренда", VOCAB)

    assert result is None


def test_suggest_returns_none_when_no_close_match_exists():
    # "космонавтика" is not close to anything in VOCAB -- no forced bad guess.
    result = _corrector().suggest("космонавтика", VOCAB)

    assert result is None


def test_suggest_returns_none_for_empty_vocabulary():
    result = _corrector().suggest("договр", set())

    assert result is None


def test_suggest_respects_max_distance_even_if_score_cutoff_would_accept():
    # score_cutoff=0 accepts virtually anything via extractOne; max_distance=0
    # then vetoes any correction that isn't an exact match.
    corrector = _corrector(max_distance=0, score_cutoff=0.0)

    result = corrector.suggest("договр", VOCAB)

    assert result is None


def test_suggest_preserves_surrounding_punctuation_and_untouched_words():
    result = _corrector().suggest("договр, аренда!", VOCAB)

    assert result == "договор, аренда!"
