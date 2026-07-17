"""Unit tests for the pluggable abbreviation expansion (task 4):
`load_abbrev_dictionary` + `DictTermExpander` over a JSON abbreviation
dictionary, `CompositeTermExpander`, and the `build_abbrev_expander`/
`build_query_expander` factory toggles (enabled/disabled, missing file)."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import QueryProcessingConfig, TermExpansionConfig
from app.query.factory import build_abbrev_expander, build_query_expander
from app.query.term_expansion import (
    CompositeTermExpander,
    DictTermExpander,
    load_abbrev_dictionary,
)

ABBREV = {
    "МГУ": "Московский государственный университет",
    "ЕГЭ": "единый государственный экзамен",
}


def _write_abbrev(path: Path, data: dict[str, str]) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_abbrev_dictionary_normalizes_to_list_values(tmp_path: Path):
    path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)

    result = load_abbrev_dictionary(path)

    assert result == {
        "МГУ": ["Московский государственный университет"],
        "ЕГЭ": ["единый государственный экзамен"],
    }


def test_load_abbrev_dictionary_missing_file_is_graceful():
    result = load_abbrev_dictionary(Path("does/not/exist.json"))

    assert result == {}


def test_abbrev_expander_appends_expansion_when_abbreviation_present(tmp_path: Path):
    path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_abbrev_dictionary(path))

    result = expander.expand("поступление в МГУ")

    assert result.startswith("поступление в МГУ")
    assert "Московский государственный университет" in result


def test_abbrev_expander_is_case_insensitive(tmp_path: Path):
    path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_abbrev_dictionary(path))

    result = expander.expand("когда сдают егэ")

    assert "единый государственный экзамен" in result


def test_abbrev_expander_no_op_when_no_abbreviation(tmp_path: Path):
    path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)
    expander = DictTermExpander(load_abbrev_dictionary(path))

    assert expander.expand("сегодня хорошая погода") == "сегодня хорошая погода"


def test_build_abbrev_expander_returns_none_when_disabled():
    cfg = QueryProcessingConfig(expansion_enabled=False)

    assert build_abbrev_expander(cfg) is None


def test_build_abbrev_expander_returns_instance_when_enabled(tmp_path: Path):
    path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)
    cfg = QueryProcessingConfig(expansion_enabled=True, abbrev_dict_path=str(path))

    expander = build_abbrev_expander(cfg)

    assert isinstance(expander, DictTermExpander)
    assert "единый государственный экзамен" in expander.expand("егэ")


def test_build_abbrev_expander_graceful_on_missing_dictionary():
    cfg = QueryProcessingConfig(expansion_enabled=True, abbrev_dict_path="does/not/exist.json")

    expander = build_abbrev_expander(cfg)

    assert isinstance(expander, DictTermExpander)
    # Missing dictionary -> empty -> expansion is a no-op, not an error.
    assert expander.expand("егэ") == "егэ"


def test_build_query_expander_returns_none_when_both_disabled():
    cfg = QueryProcessingConfig(
        term_expansion=TermExpansionConfig(enabled=False),
        expansion_enabled=False,
    )

    assert build_query_expander(cfg) is None


def test_build_query_expander_composes_synonym_and_abbrev(tmp_path: Path):
    terms_path = tmp_path / "terms.yaml"
    terms_path.write_text("бд:\n  - база данных\n", encoding="utf-8")
    abbrev_path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)

    cfg = QueryProcessingConfig(
        term_expansion=TermExpansionConfig(enabled=True, dictionary_path=str(terms_path)),
        expansion_enabled=True,
        abbrev_dict_path=str(abbrev_path),
    )

    expander = build_query_expander(cfg)

    assert isinstance(expander, CompositeTermExpander)
    result = expander.expand("бд для МГУ")
    assert "база данных" in result
    assert "Московский государственный университет" in result


def test_build_query_expander_single_when_only_abbrev_enabled(tmp_path: Path):
    abbrev_path = _write_abbrev(tmp_path / "abbrev.json", ABBREV)
    cfg = QueryProcessingConfig(
        term_expansion=TermExpansionConfig(enabled=False),
        expansion_enabled=True,
        abbrev_dict_path=str(abbrev_path),
    )

    expander = build_query_expander(cfg)

    assert isinstance(expander, DictTermExpander)
