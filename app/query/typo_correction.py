"""Ф2.2: typo suggestion for the free-text query, via `rapidfuzz` fuzzy
matching against the active index's vocabulary (`LexicalIndex.vocabulary()`).

Important, per Ф2.2 ("возвращать предложенное исправление в ответе API, не
блокируя выполнение исходного запроса"): `suggest()` only ever returns a
*suggestion* string for the API response to surface to the caller. Nothing
here (or in `SearchService`) may substitute this suggestion in place of the
original query for actual retrieval -- that would silently change what the
user searched for, which the spec explicitly rules out.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from app.query.base import TypoCorrector

# Simple word splitter -- this doesn't need to share the full
# RussianTokenizer/lemmatization pipeline, just word-splitting, since we're
# fuzzy-matching surface words against whatever's in `vocabulary`.
_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9]+")


class RapidfuzzTypoCorrector(TypoCorrector):
    """Suggests a corrected query by fuzzy-matching out-of-vocabulary query
    words against the active index's vocabulary.

    Two knobs, both honored (per `TypoCorrectionConfig`):
    - `score_cutoff`: fast prefilter via `rapidfuzz.process.extractOne`
      (0-100 scale, `rapidfuzz.fuzz.ratio` scoring convention).
    - `max_distance`: a second, stricter check via actual Levenshtein edit
      distance -- `score_cutoff` alone (a normalized similarity ratio) can
      accept pairs whose absolute edit distance still exceeds what the config
      allows, especially for longer words.
    """

    def __init__(self, max_distance: int, score_cutoff: float) -> None:
        self._max_distance = max_distance
        self._score_cutoff = score_cutoff

    def suggest(self, query: str, vocabulary: set[str]) -> str | None:
        if not vocabulary:
            return None

        words = _WORD_RE.findall(query.lower())
        corrections: dict[str, str] = {}

        for word in words:
            # Already known (or already corrected once this call) -- no
            # correction needed.
            if word in vocabulary or word in corrections:
                continue

            match = process.extractOne(
                word, vocabulary, scorer=fuzz.ratio, score_cutoff=self._score_cutoff
            )
            if match is None:
                continue

            candidate = match[0]
            if Levenshtein.distance(word, candidate) > self._max_distance:
                continue

            corrections[word] = candidate

        if not corrections:
            return None

        # NOTE (known simplification): `candidate` terms come straight from
        # the index vocabulary, which may be lemma-normalized (see
        # `LexicalConfig.use_lemmatization`) -- so a "corrected" word may come
        # back as a dictionary/lemma form rather than the exact inflected
        # form the user probably meant. Acceptable for a thesis prototype
        # since Ф2.2 only requires the suggest-and-don't-block mechanism, not
        # perfect surface-form correction.
        return _replace_words(query, corrections)


def _replace_words(query: str, corrections: dict[str, str]) -> str:
    """Rebuilds `query` with only the corrected words swapped in place --
    everything else (punctuation, spacing, casing of untouched words) is
    left exactly as the user typed it."""

    def _sub(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = corrections.get(word.lower())
        return replacement if replacement is not None else word

    return _WORD_RE.sub(_sub, query)
