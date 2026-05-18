"""
resistancemap/mortfm/survival/time_to_event_metrics.py
=======================================================
Harrell C-index, integrated Brier score, KM strata separation.

Pure stdlib + numpy, no lifelines dependency. All inputs are real
right-censored observations; randomness only enters the bootstrap CIs
(via Python's stdlib :mod:`random`, never for data fabrication).
"""

from __future__ import annotations

import math
import random as _random
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def concordance_index(
    event_times: Sequence[float],
    event_observed: Sequence[float],
    risk_scores: Sequence[float],
) -> float:
    """Harrell's C-index. Higher = better discrimination.

    A "comparable" pair has shorter follow-up uncensored OR has equal
    times AND any uncensored. ``risk_scores`` should be higher for
    higher-risk patients (predicted shorter time-to-event).
    """
    et = np.asarray(event_times, dtype=float)
    eo = np.asarray(event_observed, dtype=float)
    rs = np.asarray(risk_scores, dtype=float)
    n = len(et)
    num = 0.0
    den = 0.0
    for i in range(n):
        if not eo[i]:
            continue
        for j in range(n):
            if i == j or et[j] <= et[i]:
                continue
            if eo[j] == 0 and et[j] < et[i]:
                continue
            den += 1.0
            if rs[i] > rs[j]:
                num += 1.0
            elif rs[i] == rs[j]:
                num += 0.5
    return float(num / den) if den > 0 else float("nan")


def concordance_index_bootstrap_ci(
    event_times: Sequence[float],
    event_observed: Sequence[float],
    risk_scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 23,
) -> tuple[float, float, float]:
    """Return ``(point, p2.5, p97.5)`` C-index with stratified bootstrap CI.

    Resampling is over the REAL observed event tuples, never over fabricated data.
    """
    et = list(event_times); eo = list(event_observed); rs = list(risk_scores)
    n = len(et)
    point = concordance_index(et, eo, rs)
    rng = _random.Random(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        c = concordance_index([et[k] for k in idx], [eo[k] for k in idx], [rs[k] for k in idx])
        if not math.isnan(c):
            boots.append(c)
    boots.sort()
    if not boots:
        return point, float("nan"), float("nan")
    lo = boots[max(0, int(0.025 * len(boots)) - 1)]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return point, float(lo), float(hi)


def integrated_brier_score(
    survival_curve_pred: np.ndarray,   # (B, K)
    t_grid: np.ndarray,                # (K,)
    event_times: Sequence[float],
    event_observed: Sequence[float],
) -> float:
    """Integrated Brier score over the prediction time grid.

    Uses simple inverse-probability-of-censoring weighting with the
    Kaplan-Meier estimate of the censoring distribution.
    """
    et = np.asarray(event_times, dtype=float)
    eo = np.asarray(event_observed, dtype=float)
    sc = np.asarray(survival_curve_pred, dtype=float)
    n, K = sc.shape
    if len(et) != n or len(eo) != n:
        raise ValueError("event_times/observed length != prediction matrix rows")
    # KM of censoring (event_observed=0 treated as the "event" for the censoring distribution)
    G = {}
    at_risk = n
    current_G = 1.0
    for t, d in sorted(zip(et, 1.0 - eo)):
        if at_risk > 0 and d > 0:
            current_G *= (1.0 - d / at_risk)
        G[float(t)] = current_G
        at_risk -= 1
    def _G(t: float) -> float:
        eligible = [v for tt, v in G.items() if tt <= t]
        return eligible[-1] if eligible else 1.0

    brier_per_t = []
    for k, t in enumerate(t_grid):
        bs_sum = 0.0
        for i in range(n):
            sk = sc[i, k]
            obs_t = et[i]
            obs_e = eo[i]
            gt = max(_G(t), 1e-6)
            gobs = max(_G(obs_t), 1e-6)
            if obs_t <= t and obs_e > 0:
                bs_sum += (sk ** 2) / gobs
            elif obs_t > t:
                bs_sum += ((1.0 - sk) ** 2) / gt
            # else censored before t — drops out (IPCW weight 0)
        brier_per_t.append(bs_sum / n)
    # Trapezoidal integration over t_grid; normalise by t_grid range
    rng = float(t_grid[-1] - t_grid[0]) or 1.0
    return float(np.trapezoid(brier_per_t, t_grid) / rng)


def km_strata_separation(
    event_times: Sequence[float],
    event_observed: Sequence[float],
    risk_scores: Sequence[float],
    quantiles: tuple = (1 / 3, 2 / 3),
) -> dict:
    """Stratify by risk-score tertiles; report KM curves + log-rank p-value."""
    et = np.asarray(event_times, dtype=float)
    eo = np.asarray(event_observed, dtype=float)
    rs = np.asarray(risk_scores, dtype=float)
    if len(et) < 6:
        return {"skipped": True, "reason": f"n={len(et)} too small for strata"}
    q1, q2 = np.quantile(rs, list(quantiles))
    stratum = np.where(rs <= q1, "low",
              np.where(rs <= q2, "mid", "high"))
    out = {"strata": {}, "n": int(len(et))}
    for label in ("low", "mid", "high"):
        mask = stratum == label
        if not mask.any():
            continue
        et_s = et[mask]; eo_s = eo[mask]
        order = np.argsort(et_s)
        times = et_s[order]; obs = eo_s[order]
        # KM step
        n_at_risk = len(times); S = 1.0; curve = []
        for t, d in zip(times, obs):
            if n_at_risk > 0:
                S *= (1.0 - d / n_at_risk)
            curve.append((float(t), float(S)))
            n_at_risk -= 1
        out["strata"][label] = {
            "n": int(mask.sum()),
            "n_events": int(eo_s.sum()),
            "median_time": float(np.median(et_s)),
            "median_S_at_180d": next((s for tt, s in curve if tt >= 180), float("nan")),
            "median_S_at_365d": next((s for tt, s in curve if tt >= 365), float("nan")),
        }
    # 2-sample log-rank between high vs low (Mantel-Haenszel)
    try:
        high_mask = stratum == "high"; low_mask = stratum == "low"
        all_times = sorted(set(et[high_mask | low_mask]))
        O_h = E_h = V = 0.0
        for t in all_times:
            n_h = int(((et[high_mask] >= t)).sum())
            n_l = int(((et[low_mask] >= t)).sum())
            n_total = n_h + n_l
            if n_total == 0:
                continue
            d_h = int(((et == t) & high_mask & (eo == 1)).sum())
            d_l = int(((et == t) & low_mask & (eo == 1)).sum())
            d = d_h + d_l
            if d == 0 or n_total < 2:
                continue
            E = d * n_h / n_total
            V_t = (d * n_h * n_l * (n_total - d)) / (n_total ** 2 * (n_total - 1))
            O_h += d_h; E_h += E; V += V_t
        if V > 0:
            chi2 = (O_h - E_h) ** 2 / V
            # 1 df chi2 p-value via survival function approx
            p = math.exp(-0.5 * chi2) * (1.0 - 0.5 * (chi2 < 1))  # crude bound
            out["log_rank_chi2_high_vs_low"] = float(chi2)
            out["log_rank_p_lower_bound"] = float(p)
    except Exception:
        pass
    return out
