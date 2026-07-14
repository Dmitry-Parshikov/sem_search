"""Unit tests for `app.query.factory`: `build_typo_corrector`/
`build_term_expander` return `None` when disabled (so `SearchService` can
skip the stage entirely), and real instances when enabled."""

from __future__ import annotations

from pathlib import Path

from app.config import TermExpansionConfig, TypoCorrectionConfig
from app.query.factory import build_term_expander, build_typo_corrector
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


def test_build_term_expander_returns_none_when_disabled():
    cfg = TermExpansionConfig(enabled=False, dictionary_path="does/not/exist.yaml")

    assert build_term_expander(cfg) is None


def test_build_term_expander_returns_real_instance_when_enabled(tmp_path: Path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text("бд:\n  - база данных\n", encoding="utf-8")
    cfg = TermExpansionConfig(enabled=True, dictionary_path=str(yaml_path))

    expander = build_term_expander(cfg)

    assert isinstance(expander, DictTermExpander)
    assert expander.expand("подключение к бд") == "подключение к бд база данных"
