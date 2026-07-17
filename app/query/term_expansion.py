"""Ф2.4: query expansion via a simple, code-independent term dictionary
(`config/terms_dictionary.yaml`, format `термин: [синонимы/раскрытия]`) --
editable without touching the code, per spec.

Also hosts the pluggable abbreviation dictionary (`data/abbrev_dict.json`,
format `"аббревиатура": "расшифровка"`): a separate, toggleable source of
expansions loaded via `load_abbrev_dictionary` and applied through the same
`DictTermExpander`, so abbreviation and synonym expansion share one code path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from app.query.base import TermExpander

_BOUNDARY_CHARS = r"а-яёa-z0-9"


def load_term_dictionary(path: Path) -> dict[str, list[str]]:
    """Loads the `термин: [синонимы]` YAML dictionary used by
    `DictTermExpander`. Missing/empty file yields an empty dictionary rather
    than raising, so a misconfigured path degrades to "no expansions" instead
    of crashing the whole app at startup."""

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return {str(key): [str(v) for v in (value or [])] for key, value in data.items()}


def load_abbrev_dictionary(path: Path) -> dict[str, list[str]]:
    """Loads the `"аббревиатура": "расшифровка"` JSON dictionary and normalizes
    it into the `{key: [expansion]}` shape `DictTermExpander` consumes. A
    missing or malformed file yields an empty dictionary rather than raising,
    so a disabled/misconfigured abbreviation source degrades to "no
    expansions" instead of crashing the app at startup (graceful, per spec)."""

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    return {str(key): [str(value)] for key, value in data.items()}


class DictTermExpander(TermExpander):
    """Case-insensitive whole-phrase lookup of each dictionary key inside the
    query; every key found has its synonym/expansion list appended to the
    query (space-joined, no duplicate appends)."""

    def __init__(self, term_dict: dict[str, list[str]]) -> None:
        self._term_dict = term_dict

    def expand(self, query: str) -> str:
        appended: list[str] = []

        for key, synonyms in self._term_dict.items():
            if not _contains_whole_phrase(query, key):
                continue
            for synonym in synonyms:
                if synonym not in appended:
                    appended.append(synonym)

        if not appended:
            return query

        return query + " " + " ".join(appended)


class CompositeTermExpander(TermExpander):
    """Applies several expanders in sequence (e.g. synonym dictionary +
    abbreviation dictionary), each appending to the query produced by the
    previous one. Used to keep both expansion sources pluggable independently
    while still exposing a single `TermExpander` to `SearchService`."""

    def __init__(self, expanders: list[TermExpander]) -> None:
        self._expanders = expanders

    def expand(self, query: str) -> str:
        result = query
        for expander in self._expanders:
            result = expander.expand(result)
        return result


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    pattern = rf"(?<![{_BOUNDARY_CHARS}]){re.escape(phrase.lower())}(?![{_BOUNDARY_CHARS}])"
    return re.search(pattern, text.lower()) is not None
