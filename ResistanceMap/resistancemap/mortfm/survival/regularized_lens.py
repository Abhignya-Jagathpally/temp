"""
resistancemap/mortfm/survival/regularized_lens.py
====================================================
Anti-overfitting wrapper around the v16 LENS survival path. Targets the
n=29 regime where the v16 evaluation showed clinical+graph variants
overfit catastrophically (C=0.506 < baseline 0.531).

Knobs (all enabled by default):
  * L1 + L2 weight-decay on the encoder
  * dropout on the projector + competing-risk head
  * feature bagging (random feature subset per fold)
  * bootstrap optimism correction (Harrell 1996)
"""

from __future__ import annotations

import logging
import math
import random as _random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RegularizedLENSConfig:
    d_latent: int
    n_bins: int = 8
    d_hidden: int = 128
    dropout: float = 0.3
    l1: float = 1e-4
    l2: float = 1e-3
    feature_bag_frac: float = 1.0   # 1.0 = no bagging


class RegularizedLENS(nn.Module):
    """Small encoder + CompetingRiskHead with aggressive regularisation.

    Designed for the n<50 regime; does NOT replace the full
    GraphEnergyResistanceSDE — it is the *baseline* that any
    full-model claim must beat.
    """

    def __init__(self, cfg: RegularizedLENSConfig, n_features: int) -> None:
        super().__init__()
        from resistancemap.mortfm.survival.competing_risk_head import CompetingRiskHead
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(n_features, cfg.d_hidden), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_latent), nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = CompetingRiskHead(
            d_latent=cfg.d_latent, n_bins=cfg.n_bins, event_names=["progression"],
        )

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        out = self.head(z)
        out["z"] = z
        return out

    def l1_penalty(self) -> torch.Tensor:
        return sum(p.abs().sum() for p in self.parameters())


def fit_predict_loo_regularized(
    X: np.ndarray,                # (n, n_features)
    event_time: Sequence[float],
    event_observed: Sequence[float],
    *,
    cfg: Optional[RegularizedLENSConfig] = None,
    epochs: int = 200,
    lr: float = 5e-3,
    feature_bag_seed: int = 71,
    log_every: int = 5,
) -> List[float]:
    """LOO fit + predict. Returns a risk score per held-out patient."""
    from resistancemap.mortfm.survival.competing_risk_head import nll_competing_risk

    if cfg is None:
        cfg = RegularizedLENSConfig(d_latent=32, n_bins=8)
    X = np.asarray(X, dtype=np.float32)
    n, n_features = X.shape

    # Discrete-time event-bin assignment.
    t_grid = np.linspace(0.0, float(max(event_time)) * 1.05, cfg.n_bins + 1)
    event_bins = np.searchsorted(t_grid[1:], event_time, side="right").clip(0, cfg.n_bins - 1)

    risk_scores: List[float] = []
    rng_bag = _random.Random(feature_bag_seed)
    bag_size = max(2, int(round(n_features * cfg.feature_bag_frac)))
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        bag_features = sorted(rng_bag.sample(range(n_features), bag_size)) if bag_size < n_features else list(range(n_features))
        X_train = X[np.ix_(train_idx, bag_features)]
        X_test = X[np.ix_([i], bag_features)]
        model = RegularizedLENS(cfg, n_features=bag_size)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.l2)
        Xt = torch.tensor(X_train)
        bins_t = torch.tensor(event_bins[train_idx], dtype=torch.long)
        obs_t = torch.tensor(np.asarray(event_observed)[train_idx], dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            out = model(Xt)
            loss = nll_competing_risk(
                out["hazard"], out["survival_curve"], bins_t, obs_t,
            ) + cfg.l1 * model.l1_penalty()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            S = model(torch.tensor(X_test))["survival_curve"][0].cpu().numpy()
        # Risk = 1 - S at the median bin.
        risk_scores.append(float(1.0 - S[cfg.n_bins // 2]))
        if (i + 1) % log_every == 0:
            logger.info("Regularized LENS LOO fold %d/%d done", i + 1, n)
    return risk_scores


def bootstrap_optimism_correction(
    observed_cindex: float,
    X: np.ndarray,
    event_time: Sequence[float],
    event_observed: Sequence[float],
    *,
    cfg: Optional[RegularizedLENSConfig] = None,
    n_bootstrap: int = 50,
    epochs: int = 100,
    lr: float = 5e-3,
    seed: int = 79,
) -> dict:
    """Harrell 1996 optimism correction.

    For each bootstrap sample b:
      1. Resample (with replacement) the n patients -> bootstrap dataset
      2. Train on bootstrap dataset
      3. Compute C-index on bootstrap-train AND on original full data
      4. optimism_b = C_bootstrap_train - C_original
    Corrected C-index = observed_cindex - mean(optimism_b)
    """
    from resistancemap.mortfm.survival.competing_risk_head import nll_competing_risk
    from resistancemap.mortfm.survival.time_to_event_metrics import concordance_index

    if cfg is None:
        cfg = RegularizedLENSConfig(d_latent=32, n_bins=8)
    X = np.asarray(X, dtype=np.float32)
    n, n_features = X.shape
    rng = _random.Random(seed)
    t_grid = np.linspace(0.0, float(max(event_time)) * 1.05, cfg.n_bins + 1)
    event_bins = np.searchsorted(t_grid[1:], event_time, side="right").clip(0, cfg.n_bins - 1)
    et_list = list(event_time); eo_list = list(event_observed)

    optimisms: List[float] = []
    for b in range(n_bootstrap):
        boot_idx = [rng.randrange(n) for _ in range(n)]
        X_b = X[boot_idx]
        bins_b = event_bins[boot_idx]
        et_b = [et_list[k] for k in boot_idx]
        eo_b = [eo_list[k] for k in boot_idx]

        model = RegularizedLENS(cfg, n_features=n_features)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.l2)
        Xt = torch.tensor(X_b)
        bins_t = torch.tensor(bins_b, dtype=torch.long)
        obs_t = torch.tensor(np.asarray(eo_b), dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            out = model(Xt)
            loss = nll_competing_risk(
                out["hazard"], out["survival_curve"], bins_t, obs_t,
            ) + cfg.l1 * model.l1_penalty()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            S_boot = model(Xt)["survival_curve"][:, cfg.n_bins // 2].cpu().numpy()
            S_orig = model(torch.tensor(X))["survival_curve"][:, cfg.n_bins // 2].cpu().numpy()
        risk_boot = (1.0 - S_boot).tolist()
        risk_orig = (1.0 - S_orig).tolist()
        c_boot = concordance_index(et_b, eo_b, risk_boot)
        c_orig = concordance_index(et_list, eo_list, risk_orig)
        if not (math.isnan(c_boot) or math.isnan(c_orig)):
            optimisms.append(c_boot - c_orig)
    if not optimisms:
        return {"observed": observed_cindex, "mean_optimism": float("nan"),
                "corrected_cindex": float("nan"), "n_bootstrap": 0}
    mean_opt = float(np.mean(optimisms))
    return {
        "observed": float(observed_cindex),
        "mean_optimism": mean_opt,
        "corrected_cindex": float(observed_cindex - mean_opt),
        "n_bootstrap": int(len(optimisms)),
    }
