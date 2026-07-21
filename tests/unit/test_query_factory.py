"""Unit tests for `app.query.factory`: `build_typo_corrector` returns `None`
when disabled, `build_query_expander` returns `None` when disabled or dir is
empty, and a real `DictTermExpander` when dictionaries are present."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import QueryProcessingConfig, TypoCorrectionConfig
from app.query.factory import build_query_expander, build_typo_corrector
from app.query.term_expansion import DictTermExpander
from app.query.typo_correction import RapidfuzzTypoCorrector


def test_build_typo_corrector_returns_none_when_disabled():
    cfg = TypoCorrectionConfig(enabled=False)
    assert build_typo_corrector(cfg) is None


def test_build_typo_corrector_returns_real_instance_when_enabled():
    cfg = TypoCorrectionConfig(enabled=True, max_distance=3, score_cutoff=75.0)
    corrector = build_typo_corrector(cfg)
    assert isinstance(corrector, RapidfuzzTypoCorrector)
    assert corrector._max_distance == 3
    assert corrector._score_cutoff == 75.0


def test_build_query_expander_returns_none_when_disabled(tmp_path: Path):
    cfg = QueryProcessingConfig(dictionaries_enabled=False, dictionaries_dir=str(tmp_path))
    assert build_query_expander(cfg) is None


def test_build_query_expander_returns_none_for_empty_dir(tmp_path: Path):
    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir=str(tmp_path))
    assert build_query_expander(cfg) is None


def test_build_query_expander_loads_yaml_dictionary(tmp_path: Path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text("бд:\n  - база данных\n", encoding="utf-8")
    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir=str(tmp_path))
    expander = build_query_expander(cfg)
    assert isinstance(expander, DictTermExpander)
    assert expander.expand("подключение к бд") == "подключение к бд база данных"


def test_build_query_expander_loads_json_dictionary(tmp_path: Path):
    json_path = tmp_path / "abbrev.json"
    json_path.write_text(json.dumps({"МГУ": "Московский государственный университет"}), encoding="utf-8")
    cfg = QueryProcessingConfig(dictionaries_enabled=True, dictionaries_dir=str(tmp_path))
    expander = build_query_expander(cfg)
    assert isinstance(expander, DictTermExpander)
    result = expander.expand("поступление в МГУ")
    assert "Московский государственный университет" in result
