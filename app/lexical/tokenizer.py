"""Russian-aware tokenizer with optional lemmatization.

This is the ONLY file that imports a pymorphy package, per the isolation
requirement in the plan (risk #2): `pymorphy2` installs cleanly on
Windows/Python 3.11 but is broken at *runtime* here (it calls
`inspect.getargspec`, removed in Python 3.10+), so we prefer the actively
maintained fork `pymorphy3` and fall back to `pymorphy2` only if pymorphy3
is unavailable for some reason. Nothing outside this module references
either package -- callers only see `RussianTokenizer`.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


class _MorphAnalyzerProtocol(Protocol):
    def parse(self, word: str) -> list:  # pragma: no cover - structural typing only
        ...


def _build_morph_analyzer():
    """Constructs a pymorphy3 (preferred) or pymorphy2 (fallback)
    MorphAnalyzer.

    Dictionary lookup is done via each dict package's own `get_path()`
    helper (`pymorphy3_dicts_ru.get_path()` / `pymorphy2_dicts_ru.get_path()`)
    and passed explicitly to `MorphAnalyzer(path=...)`, rather than relying
    on pymorphy's default entry-point-based discovery -- that discovery path
    imports `pkg_resources`, which recent `setuptools` releases (>=81) no
    longer ship by default, and importing it raises `ModuleNotFoundError`
    on such environments (confirmed in this project's dev environment).
    Passing the path explicitly sidesteps `pkg_resources` entirely.
    """

    try:
        from pymorphy3 import MorphAnalyzer  # type: ignore[import-untyped]

        try:
            import pymorphy3_dicts_ru  # type: ignore[import-untyped]

            return MorphAnalyzer(path=pymorphy3_dicts_ru.get_path())
        except ImportError:
            return MorphAnalyzer()
    except ImportError:
        pass

    try:
        from pymorphy2 import MorphAnalyzer  # type: ignore[import-untyped]

        try:
            import pymorphy2_dicts_ru  # type: ignore[import-untyped]

            return MorphAnalyzer(path=pymorphy2_dicts_ru.get_path())
        except ImportError:
            return MorphAnalyzer()
    except ImportError as exc:
        raise ImportError(
            "Neither pymorphy3 nor pymorphy2 is installed; install "
            "pymorphy3 (recommended) or pymorphy2 to enable lemmatization."
        ) from exc


class RussianTokenizer:
    """Lowercases, strips punctuation, splits into word tokens; optionally
    lemmatizes each token via pymorphy3/pymorphy2 (Ф1.4)."""

    def __init__(self, use_lemmatization: bool) -> None:
        self._use_lemmatization = use_lemmatization
        self._morph: _MorphAnalyzerProtocol | None = None
        if use_lemmatization:
            # Expensive to construct -- cache once at the tokenizer instance level.
            self._morph = _build_morph_analyzer()

    def tokenize(self, text: str) -> list[str]:
        raw_tokens = _TOKEN_RE.findall(text.lower())
        if not self._use_lemmatization or self._morph is None:
            return raw_tokens
        return [self._lemmatize(tok) for tok in raw_tokens]

    def _lemmatize(self, token: str) -> str:
        parses = self._morph.parse(token)
        if not parses:
            return token
        return parses[0].normal_form
