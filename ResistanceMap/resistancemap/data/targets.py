"""Target registry. Every reported metric must cite a target key + units.

Resolves the README contradiction where ``test_mse=0.7191`` and ``AUROC=0.89``
were quoted side-by-side without specifying that they refer to different
targets on different scales.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    key: str
    kind: str                 # "binary" | "regression"
    units: str                # e.g. "log10(uM)", "probability", "dimensionless AUC"
    primary_metric: str       # "auroc" | "auprc" | "mae" | "rmse"
    range: tuple[float, float] | None = None
    naive_baseline_required: bool = True


TARGETS = {
    "resistance_binary": Target(
        key="resistance_binary",
        kind="binary",
        units="probability",
        primary_metric="auroc",
        range=(0.0, 1.0),
    ),
    "log_ic50": Target(
        key="log_ic50",
        kind="regression",
        units="log10(uM)",
        primary_metric="mae",
        range=(-4.0, 4.0),
    ),
    "normalised_auc": Target(
        key="normalised_auc",
        kind="regression",
        units="dimensionless AUC [0,1]",
        primary_metric="mae",
        range=(0.0, 1.0),
    ),
    "trajectory_score": Target(
        key="trajectory_score",
        kind="regression",
        units="z-scored resistance",
        primary_metric="mae",
        range=None,
    ),
}


def require_target(key: str) -> Target:
    if key not in TARGETS:
        raise KeyError(
            f"Unknown target {key!r}. Register in resistancemap/data/targets.py "
            f"before use. Known: {sorted(TARGETS)}"
        )
    return TARGETS[key]
