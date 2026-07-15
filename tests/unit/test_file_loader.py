"""Unit tests for `app.preprocessing.file_loader` — bulk folder scanning
with encoding detection and per-file error reporting."""

from __future__ import annotations

from pathlib import Path

from app.preprocessing.file_loader import SUPPORTED_SUFFIXES, load_folder


# ── Basic success paths ──

def test_loads_txt_file_utf8(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("Привет, мир!", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 1
    assert successes[0]["doc_id"] == "doc"
    assert successes[0]["text"] == "Привет, мир!"
    assert successes[0]["source"] == "folder"


def test_loads_md_file(tmp_path: Path):
    (tmp_path / "readme.md").write_text("# Заголовок\n\nТекст.", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 1
    assert "Заголовок" in successes[0]["text"]


def test_respects_source_label(tmp_path: Path):
    (tmp_path / "a.txt").write_text("текст", encoding="utf-8")
    successes, errors = load_folder(tmp_path, source_label="my-custom-corpus")
    assert len(successes) == 1
    assert successes[0]["source"] == "my-custom-corpus"


def test_doc_id_is_stem_not_full_path(tmp_path: Path):
    (tmp_path / "my_document_v2.txt").write_text("текст", encoding="utf-8")
    successes, _ = load_folder(tmp_path)
    assert successes[0]["doc_id"] == "my_document_v2"


# ── Multiple files / mixed types ──

def test_loads_all_txt_in_flat_folder(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"file_{i}.txt").write_text(f"текст {i}", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 5
    doc_ids = {d["doc_id"] for d in successes}
    assert doc_ids == {f"file_{i}" for i in range(5)}


def test_mixed_txt_and_md(tmp_path: Path):
    (tmp_path / "a.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "b.md").write_text("# markdown", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 2


# ── Unsupported / empty / errors ──

def test_unsupported_files_are_skipped(tmp_path: Path):
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("pdf content", encoding="utf-8")
    (tmp_path / "c.jpg").write_bytes(b"\xff\xd8\xff")
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 1
    assert successes[0]["doc_id"] == "a"


def test_empty_file_reported_as_error(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(successes) == 0
    assert len(errors) == 1
    assert errors[0]["suffix"] == ".txt"
    assert "empty" in errors[0]["error"].lower() or "empty" in errors[0]["error"]


def test_whitespace_only_file_reported_as_error(tmp_path: Path):
    (tmp_path / "blank.md").write_text("   \n\t  \n  ", encoding="utf-8")
    successes, errors = load_folder(tmp_path)
    assert len(successes) == 0
    assert len(errors) == 1


def test_non_existent_directory():
    successes, errors = load_folder("/nonexistent/path/12345")
    assert len(successes) == 0
    assert len(errors) == 1
    assert "Not a directory" in errors[0]["error"]


# ── Nested directories ──

def test_traverses_nested_directories(tmp_path: Path):
    (tmp_path / "a.txt").write_text("top", encoding="utf-8")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "b.txt").write_text("nested", encoding="utf-8")
    deep = sub / "deep"
    deep.mkdir()
    (deep / "c.md").write_text("deep nested", encoding="utf-8")

    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 3
    texts = {d["text"] for d in successes}
    assert texts == {"top", "nested", "deep nested"}


# ── Encoding detection ──

def test_reads_windows1251_encoded_txt(tmp_path: Path):
    """charset-normalizer should detect and decode Windows-1251 correctly."""
    text = "Кириллический текст в Windows-1251"
    raw = text.encode("windows-1251")
    (tmp_path / "win1251.txt").write_bytes(raw)
    successes, errors = load_folder(tmp_path)
    assert len(errors) == 0
    assert len(successes) == 1
    assert successes[0]["text"] == text


# ── smoke: all supported suffixes ──

def test_all_supported_suffixes_are_in_readers():
    """Guard against adding a suffix to SUPPORTED_SUFFIXES without a reader."""
    from app.preprocessing.file_loader import _READERS

    for suffix in SUPPORTED_SUFFIXES:
        assert suffix in _READERS, f"{suffix!r} has no reader registered"
