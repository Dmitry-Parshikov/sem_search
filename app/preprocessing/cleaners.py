"""Low-level text cleaning helpers used by `TextPreprocessor` (Ф1.1).

Kept dependency-free on purpose (stdlib only): HTML stripping via
`html.parser.HTMLParser`, whitespace collapsing via regex, Unicode
normalization via `unicodedata`. No `beautifulsoup4` -- outside the
approved tech list (see requirements.md section 7).
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser

# Tags whose content is not real document text and should be dropped
# entirely, not just have their tags stripped.
_SKIP_CONTENT_TAGS = {"script", "style", "head", "noscript"}

_WHITESPACE_RE = re.compile(r"\s+")


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML -> plain text extractor.

    Not a full HTML5-compliant parser (that would need an extra dependency),
    just enough to strip tags/comments/script/style content for the kind of
    simple HTML documents this project indexes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr"}:
            # Block-ish tags: insert a space so words on either side of a
            # dropped tag don't get glued together, e.g. "<p>a</p><p>b</p>".
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def strip_html(raw_html: str) -> str:
    """Strip HTML tags/comments/script/style, returning plain text content."""

    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()
    return parser.get_text()


def collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace (spaces, tabs, newlines) to a single space and strip ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """Unicode NFC normalization (Ф1.1)."""

    return unicodedata.normalize("NFC", text)
