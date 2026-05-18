"""
resistancemap/mortfm/survival/baselines.py
===========================================
Mandatory comparators for every patient-level MORT-FM claim.

The v17 steering rule: NO patient-level claim is allowed unless it beats
the clinical-only baseline on the SAME LOO splits. This module supplies
two minimal, dependency-light baselines:

  * :class:`ClinicalOnlyCox` — Breslow-tied Cox proportional-hazards
    regression with L2 regularisation, fit by gradient descent. Stdlib +
    torch only; no lifelines dependency.
  * :class:`PermutationControl` — wraps any risk-score producer and
    randomly permutes the event labels to produce a null-distribution
    C-index. Uses stdlib :mod:`random` (resamples REAL events, never
    fabricates).

Both expose ``fit_predict_loo(X, event_time, event_observed) -> risk_scores``
so a downstream concordance_index call evaluates them on the same axis as
the LENS variants.
"""

from __future__ import annotations

import logging
import math
import random as _random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClinicalOnlyCox(nn.Module):
    """Cox PH with L2 regularisation, fit by Adam.

    Parameters
    ----------
    n_features
        Number of clinical features (e.g. ISS, age, gender, bort_1L,
        n_treatments — the encode_for_lens output).
    l2
        L2 weight-decay strength. Conservative default for n~30.
    """

    def __init__(self, n_features: int, l2: float = 1e-2) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.l2 = l2

    def linear_predictor(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.beta

    def _neg_partial_log_likelihood(
        self,
        X: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Breslow partial likelihood for tied events."""
        lp = self.linear_predictor(X)
        order = torch.argsort(time, descending=True)
        lp_s = lp[order]
        event_s = event[order]
        # Cumulative sum of exp(lp) over the risk set.
        # In descending time order, risk set at row i = rows 0..i.
        max_lp = lp_s.max()
        exp_lp = torch.exp(lp_s - max_lp)
        cum_risk = torch.cumsum(exp_lp, dim=0)
        log_cum_risk = torch.log(cum_risk.clamp(min=1e-12)) + max_lp
        pll = (event_s * (lp_s - log_cum_risk)).sum()
        return -pll + 0.5 * self.l2 * (self.beta ** 2).sum()

    def fit(
        self,
        X: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        *,
        epochs: int = 300,
        lr: float = 0.05,
    ) -> "ClinicalOnlyCox":
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self._neg_partial_log_likelihood(X, time, event)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict_risk(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_predictor(X)


def cox_fit_predict_loo(
    X: np.ndarray,
    event_time: Sequence[float],
    event_observed: Sequence[float],
    *,
    epochs: int = 300,
    lr: float = 0.05,
    l2: float = 1e-2,
) -> List[float]:
    """LOO Cox: for each held-out patient i, fit on n-1 and predict their risk score."""
    X_t = torch.tensor(X, dtype=torch.float32)
    et = torch.tensor(list(event_time), dtype=torch.float32)
    eo = torch.tensor(list(event_observed), dtype=torch.float32)
    n = X_t.shape[0]
    out: List[float] = []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        cox = ClinicalOnlyCox(n_features=X_t.shape[1], l2=l2)
        cox.fit(X_t[train_idx], et[train_idx], eo[train_idx], epochs=epochs, lr=lr)
        out.append(float(cox.predict_risk(X_t[i:i+1]).item()))
    return out


def permutation_null_cindex(
    risk_scores: Sequence[float],
    event_time: Sequence[float],
    event_observed: Sequence[float],
    *,
    n_permutations: int = 200,
    seed: int = 31,
) -> dict:
    """Empirical null-distribution C-index by permuting event labels.

    Resamples (event_observed, event_time) jointly to keep the marginal
    follow-up structure but break any signal in risk_scores. Returns
    the null distribution + the observed empirical p-value.
    """
    from resistancemap.mortfm.survival.time_to_event_metrics import concordance_index

    rs = list(risk_scores); et = list(event_time); eo = list(event_observed)
    n = len(rs)
    if n < 4:
        return {"null_mean": float("nan"), "null_std": float("nan"), "p_value": float("nan")}
    observed = concordance_index(et, eo, rs)
    rng = _random.Random(seed)
    null_cis: List[float] = []
    pairs = list(zip(et, eo))
    for _ in range(n_permutations):
        perm = pairs[:]
        rng.shuffle(perm)
        et_p = [p[0] for p in perm]; eo_p = [p[1] for p in perm]
        c = concordance_index(et_p, eo_p, rs)
        if not math.isnan(c):
            null_cis.append(c)
    if not null_cis:
        return {"null_mean": float("nan"), "null_std": float("nan"), "p_value": float("nan")}
    null_mean = float(np.mean(null_cis))
    null_std = float(np.std(null_cis)) if len(null_cis) > 1 else 0.0
    # One-sided: P(null >= observed)
    n_ge = sum(1 for c in null_cis if c >= observed)
    p_value = (n_ge + 1) / (len(null_cis) + 1)
    return {
        "observed_cindex": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "n_permutations": len(null_cis),
        "p_value": float(p_value),
    }
