"""Regex-based Russian/English sentence splitter.

Simplification note: a proper sentence boundary detector (e.g. NLTK punkt)
is outside the approved tech list for this project (see requirements.md
section 7), so we use a regex heuristic instead. It splits on `.`, `!`,
`?`, `…` followed by whitespace and an uppercase/quote-starting next
sentence, with a small hard-coded list of common Russian abbreviations to
avoid splitting on them. This is "good enough" for short synthetic
documents used in this project's tests/demo corpus -- not a
general-purpose NLP-grade sentence segmenter.
"""

from __future__ import annotations

import re

# Common Russian abbreviations that end in a period but should not trigger
# a sentence break. Matched case-insensitively at the end of a token
# immediately preceding whitespace + the terminal punctuation check.
_ABBREVIATIONS = {
    "т.д", "т.п", "т.е", "т.к", "т.н",
    "др", "пр", "см", "рис", "стр", "гл", "п",
    "г", "гг", "в", "вв", "н.э", "до н.э",
    "им", "тыс", "млн", "млрд", "руб",
    "мр", "гр", "проф", "акад", "им",
}

# Sentence-ending punctuation, possibly repeated (e.g. "?!", "...").
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[\"'«“„А-ЯЁA-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Split `text` into a list of non-empty, stripped sentences."""

    text = text.strip()
    if not text:
        return []

    # First split on the sentence-boundary heuristic.
    raw_parts = _SENTENCE_END_RE.split(text)

    # Merge parts back together where the split occurred right after a
    # known abbreviation (i.e. the preceding "sentence" ends with one of
    # the abbreviations above followed by a period).
    sentences: list[str] = []
    buffer = ""
    for part in raw_parts:
        buffer = f"{buffer} {part}".strip() if buffer else part
        if _ends_with_abbreviation(buffer):
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


def _ends_with_abbreviation(fragment: str) -> bool:
    match = re.search(r"([A-Za-zА-Яа-яЁё.]+)\.$", fragment)
    if not match:
        return False
    token = match.group(1).lower().rstrip(".")
    # Also check the raw trailing token including internal dots, e.g. "т.д".
    tail_with_dots = re.search(r"([A-Za-zА-Яа-яЁё]+(?:\.[A-Za-zА-Яа-яЁё]+)*)\.$", fragment)
    candidates = {token}
    if tail_with_dots:
        candidates.add(tail_with_dots.group(1).lower())
    return bool(candidates & _ABBREVIATIONS)
