from __future__ import annotations

from pathlib import Path

from app.config import TermExpansionConfig, TypoCorrectionConfig
from app.query.base import TermExpander, TypoCorrector
from app.query.term_expansion import DictTermExpander, load_term_dictionary
from app.query.typo_correction import RapidfuzzTypoCorrector


def build_typo_corrector(cfg: TypoCorrectionConfig) -> TypoCorrector | None:
    """`None` when disabled, so callers (`SearchService`) skip the step
    entirely rather than calling a corrector that's forced to no-op."""

    if not cfg.enabled:
        return None
    return RapidfuzzTypoCorrector(max_distance=cfg.max_distance, score_cutoff=cfg.score_cutoff)


def build_term_expander(cfg: TermExpansionConfig) -> TermExpander | None:
    """`None` when disabled; otherwise loads the dictionary from
    `cfg.dictionary_path` up front (once, at build time -- not per-request)."""

    if not cfg.enabled:
        return None
    term_dict = load_term_dictionary(Path(cfg.dictionary_path))
    return DictTermExpander(term_dict=term_dict)
