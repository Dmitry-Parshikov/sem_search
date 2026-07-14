from __future__ import annotations

from app.config import HybridizationConfig
from app.hybrid.base import Hybridizer
from app.hybrid.rrf import RRFHybridizer
from app.hybrid.weighted import WeightedSumHybridizer


def build_hybridizer(cfg: HybridizationConfig) -> Hybridizer:
    if cfg.method == "rrf":
        return RRFHybridizer(k=cfg.rrf_k)
    if cfg.method == "weighted":
        return WeightedSumHybridizer(
            dense_weight=cfg.weighted.dense_weight,
            lexical_weight=cfg.weighted.lexical_weight,
            normalization=cfg.weighted.normalization,
        )
    raise ValueError(f"Unknown hybridization method {cfg.method!r}")
