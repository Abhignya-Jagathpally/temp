"""Multi-α attention propagation over STRING — graph-attention generalization
of the v10 RWR closed-form solve.

Mathematical relationship to v10 RWR
------------------------------------
The v10 propagation step (`resistancemap.landscape.rwr_propagation.rwr_power`)
solves the linear system

    r = α s + (1-α) P r              (Köhler 2008 / Vanunu 2010)

via fixed-point iteration on a column-stochastic P = A D^{-1}, with a *single*
restart probability α (default 0.7).

Two facts about RWR that motivate the v11 generalization:

(F1) The choice α = 0.7 is a hyperparameter — Köhler reports α ∈ [0.3, 0.9]
     all give qualitatively similar driver-recall on yeast PPI. Different
     drug-target seeds may favor different α: tight pharmacological clusters
     (β5 proteasome) prefer high α (less spread), broad pathway-level
     druggable hubs (HDAC) may prefer low α (more spread). Single-α RWR
     cannot adapt.

(F2) RWR is a graph-attention layer with attention scores hard-coded to row-
     stochastic Markov transition. Replacing the hard-coded transition with a
     learnable / parameterized attention kernel is the natural generalization
     (Veličković 2018 GAT, Brody 2022 GATv2).

The v11 module here implements (F1) — a multi-α ensemble — without (F2)'s
learnable attention. (F2) is deferred to v12 because it requires drug-target
training labels and a held-out evaluation protocol, neither of which is in
scope for this sprint. The training-free multi-α ensemble is sufficient to
test whether the architectural inductive bias of α-averaging changes the F6
mean-rank statistic relative to single-α RWR.

Design — `MultiAlphaAttentionPropagation`
-----------------------------------------
- H heads, each running RWR with its own α_h ∈ (0, 1).
- All heads share the SAME column-stochastic transition P (= same STRING graph).
- Output r_ensemble(g) = (1/H) Σ_h r_h(g) where r_h is α_h-RWR.
- For H=1, α=0.7, output is bit-identical to v10 RWR up to convergence
  tolerance (< 1e-7 in L1 by default).

The single-graph-multi-α design is an *ensemble* over restart probabilities,
not a learned attention kernel. The point is to establish whether F6 / F3
verdicts depend on the choice of α — and if not (which is what we expect
from Köhler's robustness result), to document this and move on. If they DO
depend on α, then a future v12 with learnable α per drug becomes a
methodological contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp


@dataclass
class MultiAlphaConfig:
    alphas: Sequence[float] = (0.3, 0.5, 0.7, 0.9)
    max_iter: int = 200
    tol: float = 1e-8
    aggregator: str = "mean"


class MultiAlphaAttentionPropagation:
    """Training-free graph-attention propagation = ensemble of RWR with H different α.

    No learnable parameters. Use as a drop-in replacement for `rwr_power`:

        prop = MultiAlphaAttentionPropagation(P, MultiAlphaConfig())
        r, info = prop.propagate(seed_vector)
    """

    def __init__(self, P: sp.csr_matrix, cfg: MultiAlphaConfig | None = None):
        self.P = P
        self.cfg = cfg or MultiAlphaConfig()

    def _rwr_single(self, s: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
        s = s.astype(np.float32, copy=False)
        r = s.copy()
        diff = float("inf")
        for it in range(self.cfg.max_iter):
            r_new = alpha * s + (1.0 - alpha) * (self.P @ r)
            diff = float(np.abs(r_new - r).sum())
            r = r_new
            if diff < self.cfg.tol:
                return r, {"alpha": alpha, "iters": it + 1, "residual_l1": diff,
                            "converged": True}
        return r, {"alpha": alpha, "iters": self.cfg.max_iter,
                    "residual_l1": diff, "converged": False}

    def propagate(self, s: np.ndarray) -> tuple[np.ndarray, dict]:
        per_head = []
        infos = []
        for alpha in self.cfg.alphas:
            r_h, info = self._rwr_single(s, alpha)
            per_head.append(r_h)
            infos.append(info)
        stacked = np.stack(per_head, axis=0)
        if self.cfg.aggregator == "mean":
            r_ens = stacked.mean(axis=0)
        elif self.cfg.aggregator == "max":
            r_ens = stacked.max(axis=0)
        elif self.cfg.aggregator == "softmax_temp1":
            w = np.exp(stacked.sum(axis=1, keepdims=True))
            w = w / w.sum()
            r_ens = (stacked * w).sum(axis=0)
        else:
            raise ValueError(f"unknown aggregator: {self.cfg.aggregator}")
        return r_ens, {
            "n_heads": len(self.cfg.alphas),
            "alphas": list(self.cfg.alphas),
            "aggregator": self.cfg.aggregator,
            "per_head": infos,
        }


def reproduce_rwr_alpha_07(P: sp.csr_matrix, s: np.ndarray, tol: float = 1e-7) -> dict:
    """Sanity check: H=1, α=0.7 ensemble must match single-α RWR within `tol`.

    Returns dict with max abs deviation; raises AssertionError if tol exceeded.
    """
    from .rwr_propagation import rwr_power
    cfg = MultiAlphaConfig(alphas=(0.7,), aggregator="mean", tol=1e-9, max_iter=400)
    prop = MultiAlphaAttentionPropagation(P, cfg)
    r_v11, _ = prop.propagate(s)
    r_v10, _ = rwr_power(P, s, alpha=0.7, tol=1e-9, max_iter=400)
    max_dev = float(np.max(np.abs(r_v11 - r_v10)))
    if max_dev > tol:
        raise AssertionError(
            f"MultiAlphaAttentionPropagation(H=1, α=0.7) failed to match v10 RWR: "
            f"max abs deviation = {max_dev:.3e} > tol = {tol:.3e}"
        )
    return {"max_abs_deviation": max_dev, "tol": tol, "ok": True}


__all__ = [
    "MultiAlphaAttentionPropagation",
    "MultiAlphaConfig",
    "reproduce_rwr_alpha_07",
]
