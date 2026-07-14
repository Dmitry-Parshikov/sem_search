"""`TextPreprocessor` (Ф1.1 implementation) and a directory-loading helper.

PDF/DOCX are assumed pre-extracted to plain text upstream (per
requirements.md 4.1: "уже-извлечённого текста из PDF/DOCX") -- no PDF/DOCX
parsing libraries here, only txt/html.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.core.types import Document
from app.preprocessing.base import Preprocessor
from app.preprocessing.cleaners import collapse_whitespace, normalize_unicode, strip_html

_HTML_EXTENSIONS = {".html", ".htm"}
_TEXT_EXTENSIONS = {".txt"}


class TextPreprocessor(Preprocessor):
    """Cleans raw document text: HTML stripping (if applicable), whitespace
    collapsing, Unicode NFC normalization."""

    def clean(self, raw_text: str, content_type: Literal["txt", "html", "plain"]) -> str:
        text = raw_text
        if content_type == "html":
            text = strip_html(text)
        text = normalize_unicode(text)
        text = collapse_whitespace(text)
        return text


def load_documents_from_dir(path: Path, source: str) -> list[Document]:
    """Load a directory of .txt/.html files into raw `Document` objects.

    doc_id = filename stem. Text is NOT cleaned here -- callers (typically
    `IndexingService`) run it through a `Preprocessor` before chunking. This
    keeps the loader a pure I/O helper and the cleaning step swappable/testable
    independently.
    """

    documents: list[Document] = []
    for file_path in sorted(Path(path).iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in _HTML_EXTENSIONS | _TEXT_EXTENSIONS:
            continue
        raw_text = file_path.read_text(encoding="utf-8")
        content_type = "html" if suffix in _HTML_EXTENSIONS else "txt"
        documents.append(
            Document(
                doc_id=file_path.stem,
                text=raw_text,
                source=source,
                extra={"content_type": content_type},
            )
        )
    return documents
