from __future__ import annotations

from pathlib import Path

from app.config import QueryProcessingConfig, TypoCorrectionConfig
from app.query.base import TermExpander, TypoCorrector
from app.query.term_expansion import DictTermExpander, load_dictionary
from app.query.typo_correction import RapidfuzzTypoCorrector


def build_typo_corrector(cfg: TypoCorrectionConfig) -> TypoCorrector | None:
    """`None` when disabled, so callers (`SearchService`) skip the step
    entirely rather than calling a corrector that's forced to no-op."""

    if not cfg.enabled:
        return None
    return RapidfuzzTypoCorrector(max_distance=cfg.max_distance, score_cutoff=cfg.score_cutoff)


def build_query_expander(cfg: QueryProcessingConfig) -> TermExpander | None:
    """Scans ``dictionaries_dir`` for ``.json`` / ``.yaml`` files, loads each
    as a ``{word: [expansions]}`` dictionary, and returns a single
    ``DictTermExpander`` over the merged result.

    Returns ``None`` when ``dictionaries_enabled`` is False or the directory
    contains no parseable dictionaries, so ``SearchService`` can skip the
    expansion step entirely.
    """
    if not cfg.dictionaries_enabled:
        return None

    dir_path = Path(cfg.dictionaries_dir)
    if not dir_path.is_dir():
        return None

    merged: dict[str, list[str]] = {}
    for file_path in sorted(dir_path.glob("*")):
        if file_path.suffix.lower() not in (".json", ".yaml", ".yml"):
            continue
        d = load_dictionary(file_path)
        for key, expansions in d.items():
            merged.setdefault(key, []).extend(
                e for e in expansions if e not in merged.get(key, [])
            )

    if not merged:
        return None
    return DictTermExpander(term_dict=merged)
