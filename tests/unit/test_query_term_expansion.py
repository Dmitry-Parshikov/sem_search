"""Unit tests for `DictTermExpander` and `load_dictionary` (Ф2.4).

Uses temp JSON/YAML fixtures rather than depending on the real dictionary
files' contents (which may change independently)."""

from __future__ import annotations

from pathlib import Path

from app.query.term_expansion import DictTermExpander, load_dictionary

TERM_DICT = {
    "апи": ["api", "интерфейс программирования приложений"],
    "нпа": ["нормативно-правовой акт"],
    "гк рф": ["гражданский кодекс российской федерации"],
}


def test_expand_appends_synonyms_when_key_term_present():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("что такое апи")
    assert result.startswith("что такое апи")
    assert "api" in result
    assert "интерфейс программирования приложений" in result


def test_expand_is_a_no_op_when_no_dictionary_term_present():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("сегодня хорошая погода")
    assert result == "сегодня хорошая погода"


def test_expand_is_case_insensitive():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("Что такое АПИ и зачем он нужен")
    assert "api" in result


def test_expand_matches_multi_word_dictionary_keys():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("положения ГК РФ о договорах")
    assert "гражданский кодекс российской федерации" in result


def test_expand_does_not_match_substrings_of_other_words():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("апитека закрыта")
    assert result == "апитека закрыта"


def test_expand_does_not_duplicate_appends_for_repeated_key():
    expander = DictTermExpander(TERM_DICT)
    result = expander.expand("апи и снова апи")
    assert result.count("api") == 1


def test_load_dictionary_from_yaml_file(tmp_path: Path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text(
        "бд:\n  - база данных\n  - database\nип:\n  - индивидуальный предприниматель\n",
        encoding="utf-8",
    )
    result = load_dictionary(yaml_path)
    assert result == {
        "бд": ["база данных", "database"],
        "ип": ["индивидуальный предприниматель"],
    }


def test_load_dictionary_from_json_file(tmp_path: Path):
    import json
    json_path = tmp_path / "terms.json"
    json_path.write_text(
        json.dumps({"МГУ": "Московский государственный университет", "ЕГЭ": "единый государственный экзамен"}),
        encoding="utf-8",
    )
    result = load_dictionary(json_path)
    assert result == {
        "МГУ": ["Московский государственный университет"],
        "ЕГЭ": ["единый государственный экзамен"],
    }


def test_load_dictionary_json_single_value_becomes_list(tmp_path: Path):
    import json
    json_path = tmp_path / "single.json"
    json_path.write_text(json.dumps({"ключ": "значение"}), encoding="utf-8")
    result = load_dictionary(json_path)
    assert result == {"ключ": ["значение"]}


def test_load_dictionary_missing_file_returns_empty():
    result = load_dictionary(Path("does/not/exist.json"))
    assert result == {}


def test_expander_built_from_loaded_dictionary_expands_correctly(tmp_path: Path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text("бд:\n  - база данных\n", encoding="utf-8")
    term_dict = load_dictionary(yaml_path)
    expander = DictTermExpander(term_dict)
    result = expander.expand("подключение к бд")
    assert "база данных" in result
