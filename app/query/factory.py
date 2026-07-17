from __future__ import annotations

from pathlib import Path

from app.config import QueryProcessingConfig, TermExpansionConfig, TypoCorrectionConfig
from app.query.base import TermExpander, TypoCorrector
from app.query.term_expansion import (
    CompositeTermExpander,
    DictTermExpander,
    load_abbrev_dictionary,
    load_term_dictionary,
)
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


def build_abbrev_expander(cfg: QueryProcessingConfig) -> TermExpander | None:
    """Pluggable abbreviation expansion: `None` when
    `expansion_enabled` is False, otherwise a `DictTermExpander` over the
    JSON abbreviation dictionary at `abbrev_dict_path`. A missing dictionary
    yields an (empty) no-op expander rather than an error (graceful)."""

    if not cfg.expansion_enabled:
        return None
    abbrev_dict = load_abbrev_dictionary(Path(cfg.abbrev_dict_path))
    return DictTermExpander(term_dict=abbrev_dict)


def build_query_expander(cfg: QueryProcessingConfig) -> TermExpander | None:
    """Combines the (synonym) term expander and the (abbreviation) expander,
    each independently toggleable via config. Returns `None` when both are
    disabled, a single expander when only one is active, or a
    `CompositeTermExpander` applying both in sequence."""

    expanders = [
        expander
        for expander in (
            build_term_expander(cfg.term_expansion),
            build_abbrev_expander(cfg),
        )
        if expander is not None
    ]
    if not expanders:
        return None
    if len(expanders) == 1:
        return expanders[0]
    return CompositeTermExpander(expanders)
