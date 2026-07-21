"""Unit tests for unified dictionary loading + `DictTermExpander` over JSON
and YAML abbreviation dictionaries, and the `build_query_expander` factory
with the new `QueryProcessingConfig` shape (dictionaries_dir + toggle)."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import QueryProcessingConfig
from app.query.factory import build_query_expander
from app.query.term_expansion import DictTermExpander, load_dictionary

ABBREV = {
    "МГУ": "Московский государственный университет",
    "ЕГЭ": "единый государственный экзамен",
}


def _write_dict(path: Path, data: dict[str, str]) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_dictionary_normalizes_json_to_list_values(tmp_path: Path):
    path = _write_dict(tmp_path / "abbrev.json", ABBREV)
    result = load_dictionary(path)
    assert result == {
        "МГУ": ["Московский государственный университет"],
        "ЕГЭ": ["единый государственный экзамен"],
    }


def test_load_dictionary_missing_file_is_graceful():
    result = load_dictionary(Path("does/not/exist.json"))
    assert result == {}


def test_dict_expander_appends_expansion_when_key_present(tmp_path: Path):
    path = _write_dict(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_dictionary(path))
    result = expander.expand("поступление в МГУ")
    assert result.startswith("поступление в МГУ")
    assert "Московский государственный университет" in result


def test_dict_expander_is_case_insensitive(tmp_path: Path):
    path = _write_dict(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_dictionary(path))
    result = expander.expand("когда сдают егэ")
    assert "единый государственный экзамен" in result


def test_dict_expander_no_op_when_no_match(tmp_path: Path):
    path = _write_dict(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_dictionary(path))
    assert expander.expand("сегодня хорошая погода") == "сегодня хорошая погода"


def test_build_query_expander_returns_none_when_disabled(tmp_path: Path):
    cfg = QueryProcessingConfig(dictionaries_enabled=False, dictionaries_dir=str(tmp_path))
    assert build_query_expander(cfg) is None


def test_build_query_expander_loads_all_dictionaries_in_dir(tmp_path: Path):
    _write_dict(tmp_path / "abbrev.json", ABBREV)
    terms_path = tmp_path / "terms.yaml"
    terms_path.write_text("бд:\n  - база данных\n", encoding="utf-8")

    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir=str(tmp_path))
    expander = build_query_expander(cfg)

    assert isinstance(expander, DictTermExpander)
    result = expander.expand("бд для МГУ")
    assert "база данных" in result
    assert "Московский государственный университет" in result


def test_build_query_expander_returns_none_for_empty_dir(tmp_path: Path):
    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir=str(tmp_path))
    # Empty directory — no dictionaries to load
    assert build_query_expander(cfg) is None


def test_build_query_expander_returns_none_for_nonexistent_dir():
    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir="./nonexistent_dir_12345")
    assert build_query_expander(cfg) is None
