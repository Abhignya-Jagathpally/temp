"""Executable guardrails.

The README claimed "8 built-in guardrails" without defining them. This module
makes each one an executable predicate. ``adjudicate`` returns the list of
failing rule names; the inference API returns HTTP 422 if non-empty.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


Rule = Callable[[dict[str, Any]], bool]


def _rule_probability_range(out: dict) -> bool:
    p = np.asarray(out.get("resistance_score", []))
    return bool(p.size and np.all((p >= 0.0) & (p <= 1.0)))


def _rule_ic50_range_log(out: dict) -> bool:
    y = np.asarray(out.get("log_ic50", []))
    return bool(y.size == 0 or np.all((y >= -4.0) & (y <= 4.0)))


def _rule_monotone_dose(out: dict) -> bool:
    curves = out.get("dose_response_curves", [])
    if not curves:
        return True
    for c in curves:
        viability = np.asarray(c["viability"])
        if np.any(np.diff(viability) > 1e-3):   # allow small noise
            return False
    return True


def _rule_stochastic_ci(out: dict) -> bool:
    """Require non-degenerate MC intervals -- not a point estimate masquerading as UQ."""
    lo = np.asarray(out.get("ci_lo", []))
    hi = np.asarray(out.get("ci_hi", []))
    if lo.size == 0:
        return True
    return bool(np.all(hi - lo > 1e-6))


def _rule_mechanism_in_vocab(out: dict, vocab: set[str] | None = None) -> bool:
    v = vocab or out.get("_mechanism_vocab")
    if not v:
        return True
    mechs = out.get("top_mechanisms", [])
    return all(m in v for m in mechs)


def _rule_drug_in_training(out: dict) -> bool:
    """If drug was held out, the output must carry a `drug_holdout=True` tag."""
    if out.get("drug_in_train") is False and not out.get("drug_holdout"):
        return False
    return True


def _rule_latent_nan_free(out: dict) -> bool:
    z = out.get("latent_z")
    return z is None or bool(np.all(np.isfinite(np.asarray(z))))


def _rule_cohort_access_tag(out: dict) -> bool:
    """Refuse public output if a controlled-access cohort contributed to it."""
    if out.get("used_controlled_access") and not out.get("public_safe", False):
        return False
    return True


GUARDRAILS: dict[str, Rule] = {
    "probability_range":  _rule_probability_range,
    "ic50_range_log":     _rule_ic50_range_log,
    "monotone_dose":      _rule_monotone_dose,
    "stochastic_ci":      _rule_stochastic_ci,
    "mechanism_in_vocab": _rule_mechanism_in_vocab,
    "drug_in_training":   _rule_drug_in_training,
    "latent_nan_free":    _rule_latent_nan_free,
    "cohort_access_tag":  _rule_cohort_access_tag,
}


def adjudicate(outputs: dict[str, Any]) -> list[str]:
    """Return list of failing guardrail names. Empty list = pass."""
    return [name for name, rule in GUARDRAILS.items() if not rule(outputs)]
