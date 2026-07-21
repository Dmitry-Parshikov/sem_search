"""Ф2.4: query expansion via pluggable dictionaries.

Any ``.json`` or ``.yaml`` file placed in the configured ``dictionaries_dir``
is loaded as a dictionary mapping ``word → expansion(s)``.  Both formats are
supported:

JSON:
    {"аббревиатура": "расшифровка", "термин": "синоним"}

YAML:
    термин:
      - синоним1
      - синоним2
    другой_термин: раскрытие

When a dictionary key is found in the query (whole-phrase, case-insensitive),
its expansion(s) are appended to the query.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from app.query.base import TermExpander

_BOUNDARY_CHARS = r"а-яёa-z0-9"


def load_dictionary(path: Path) -> dict[str, list[str]]:
    """Load a single dictionary file (JSON or YAML) and normalise every value
    to ``list[str]``.  Unknown / empty / malformed files yield ``{}`` (graceful
    degradation — a broken dictionary is skipped, not fatal)."""
    suffix = path.suffix.lower()

    try:
        with open(path, encoding="utf-8") as fh:
            if suffix == ".json":
                raw = json.load(fh) or {}
            else:  # .yaml / .yml
                raw = yaml.safe_load(fh) or {}
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError):
        return {}

    if not isinstance(raw, dict):
        return {}

    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        k = str(key)
        if isinstance(value, list):
            result[k] = [str(v) for v in value]
        elif value is not None:
            result[k] = [str(value)]
    return result


class DictTermExpander(TermExpander):
    """Case-insensitive whole-phrase lookup of each dictionary key inside the
    query; every key found has its expansions appended to the query
    (space-joined, no duplicate appends)."""

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


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    pattern = rf"(?<![{_BOUNDARY_CHARS}]){re.escape(phrase.lower())}(?![{_BOUNDARY_CHARS}])"
    return re.search(pattern, text.lower()) is not None
