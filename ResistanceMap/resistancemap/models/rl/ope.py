"""Off-policy evaluation for the RL treatment optimiser.

The RL module cannot emit treatment recommendations until:
  1. Positivity holds (behavior policy covers all actions the target would take).
  2. WIS and doubly-robust estimators agree within tolerance.
  3. A sensitivity analysis bounds unmeasured-confounding impact.

Gate is enforced in ``RLTreatmentOptimizer.recommend`` via ``self._ope_passed``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class PolicyViolation(RuntimeError):
    """Positivity violated: no counterfactual support for requested action."""


@dataclass
class OPEResult:
    wis: float
    doubly_robust: float
    positivity_violation_rate: float
    agreement_gap: float
    passed: bool
    notes: str = ""


def weighted_importance_sampling(
    behavior_logp: np.ndarray,
    target_logp: np.ndarray,
    rewards: np.ndarray,
    clip: float = 10.0,
) -> float:
    """Self-normalised WIS. Clipping bounds per-sample weight influence."""
    log_w = np.clip(np.asarray(target_logp) - np.asarray(behavior_logp), -clip, clip)
    w = np.exp(log_w)
    r = np.asarray(rewards, dtype=float)
    denom = float(w.sum())
    if denom == 0.0:
        return float("nan")
    return float((w * r).sum() / denom)


def doubly_robust(
    behavior_logp: np.ndarray,
    target_logp: np.ndarray,
    rewards: np.ndarray,
    q_hat: np.ndarray,            # E[R | s, a_behavior] estimate
    v_hat: np.ndarray,             # E_{a ~ target}[Q(s,a)] estimate
    clip: float = 10.0,
) -> float:
    log_w = np.clip(np.asarray(target_logp) - np.asarray(behavior_logp), -clip, clip)
    w = np.exp(log_w)
    rewards = np.asarray(rewards, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    return float(np.mean(v_hat + w * (rewards - q_hat)))


def positivity_check(behavior_logp: np.ndarray, min_prob: float = 1e-3) -> float:
    """Fraction of samples where behaviour probability falls below min_prob."""
    return float((np.exp(np.asarray(behavior_logp)) < min_prob).mean())


def evaluate(
    behavior_logp: np.ndarray,
    target_logp: np.ndarray,
    rewards: np.ndarray,
    q_hat: np.ndarray,
    v_hat: np.ndarray,
    positivity_threshold: float = 0.05,
    agreement_tolerance: float = 0.1,
) -> OPEResult:
    pvr = positivity_check(behavior_logp)
    wis = weighted_importance_sampling(behavior_logp, target_logp, rewards)
    dr = doubly_robust(behavior_logp, target_logp, rewards, q_hat, v_hat)
    gap = abs(wis - dr)

    positivity_ok = pvr <= positivity_threshold
    agreement_ok = gap <= agreement_tolerance
    passed = positivity_ok and agreement_ok and np.isfinite(wis) and np.isfinite(dr)

    notes = []
    if not positivity_ok:
        notes.append(f"positivity violated on {pvr:.1%} (>={positivity_threshold:.1%})")
    if not agreement_ok:
        notes.append(f"WIS/DR disagree by {gap:.3f} (>= {agreement_tolerance})")
    return OPEResult(
        wis=wis, doubly_robust=dr,
        positivity_violation_rate=pvr, agreement_gap=gap,
        passed=bool(passed), notes="; ".join(notes),
    )
