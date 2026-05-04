"""Random Walk with Restart (RWR) propagation + NPI z-score correction.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.4:

    r = α (I - (1-α) P)^(-1) s

where P = A D^(-1) is the column-stochastic transition matrix on the STRING
graph, s is the seed indicator (1 on seed genes, 0 elsewhere), and α ∈ [0, 1)
is the restart probability.

Implemented via power iteration since direct sparse-inverse on 12,651 × 12,651
is expensive. Convergence: ‖r_{t+1} - r_t‖_1 < tol or max_iter reached.

The NPI z-score (Network Permutation-based, DADA tradition; per spec §2.4 row
"NPI z-score correction") corrects for hub bias by computing a degree-stratified
null: 1,000 random seed sets matched in size and degree distribution. For each
gene g, NPI z(g) = (r_g - mean_null_g) / std_null_g.

The combined-score → z-score map removes the systematic preference RWR has for
high-degree nodes (which would always rank high regardless of biology).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch


def column_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """P = A D^{-1}: column-stochastic transition (each column sums to 1)."""
    deg = np.asarray(A.sum(axis=0)).ravel()
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)
    D_inv = sp.diags(inv_deg)
    return (A @ D_inv).tocsr()


def rwr_power(
    P: sp.csr_matrix,
    s: np.ndarray,
    alpha: float = 0.7,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """RWR via power iteration on column-stochastic P.

    r_{t+1} = α s + (1-α) P r_t   (Köhler 2008, Vanunu 2010 form).
    Returns (r, info) where info contains converged iter and final residual.
    """
    s = s.astype(np.float32)
    r = s.copy()
    for it in range(max_iter):
        r_new = alpha * s + (1.0 - alpha) * (P @ r)
        diff = float(np.abs(r_new - r).sum())
        r = r_new
        if diff < tol:
            return r, {"iters": it + 1, "residual_l1": diff, "converged": True}
    return r, {"iters": max_iter, "residual_l1": diff, "converged": False}


def build_seed_vector(
    seed_genes: Sequence[str],
    gene_to_idx: dict[str, int],
    n_nodes: int,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a (n_nodes,) seed indicator. Missing genes are silently dropped."""
    s = np.zeros(n_nodes, dtype=np.float32)
    found = []
    for g in seed_genes:
        idx = gene_to_idx.get(g)
        if idx is None:
            continue
        w = (weights or {}).get(g, 1.0)
        s[idx] = w
        found.append(g)
    if s.sum() > 0:
        s = s / s.sum()
    return s, found


def degree_stratified_null(
    deg: np.ndarray,
    seed_indices: np.ndarray,
    n_perm: int,
    n_bins: int = 20,
    rng_seed: int = 0,
) -> np.ndarray:
    """Sample n_perm size-and-degree-matched random seed sets.

    Returns an int32 array of shape (n_perm, len(seed_indices)) with sampled
    indices.
    """
    n = len(deg)
    n_seed = len(seed_indices)
    # Bin nodes by log-degree
    log_deg = np.log1p(deg)
    bin_edges = np.linspace(log_deg.min(), log_deg.max() + 1e-9, n_bins + 1)
    node_bins = np.digitize(log_deg, bin_edges) - 1
    seed_bin_counts = np.zeros(n_bins, dtype=np.int64)
    for s in seed_indices:
        b = int(node_bins[s])
        if 0 <= b < n_bins:
            seed_bin_counts[b] += 1

    # Pre-compute bin → node-list
    bin_to_nodes: list[np.ndarray] = []
    for b in range(n_bins):
        bin_to_nodes.append(np.where(node_bins == b)[0])

    g = torch.Generator().manual_seed(rng_seed)
    out = np.empty((n_perm, n_seed), dtype=np.int32)
    for p in range(n_perm):
        sampled = []
        for b, count in enumerate(seed_bin_counts):
            if count == 0:
                continue
            pool = bin_to_nodes[b]
            if len(pool) <= count:
                # Pool too small; sample with replacement to fill
                idxs = torch.randint(0, len(pool), (count,), generator=g).numpy()
            else:
                idxs = torch.randperm(len(pool), generator=g)[:count].numpy()
            sampled.append(pool[idxs])
        sampled_flat = np.concatenate(sampled)
        # Pad/trim to exactly n_seed (in case of rounding)
        if len(sampled_flat) < n_seed:
            extra = torch.randint(0, n, (n_seed - len(sampled_flat),), generator=g).numpy()
            sampled_flat = np.concatenate([sampled_flat, extra])
        out[p, :n_seed] = sampled_flat[:n_seed]
    return out


def npi_zscore(
    P: sp.csr_matrix,
    deg: np.ndarray,
    seed_indices: np.ndarray,
    n_perm: int = 1000,
    alpha: float = 0.7,
    rng_seed: int = 0,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NPI degree-stratified z-score for RWR output.

    Returns (r_obs, mean_null, std_null) all shape (n_nodes,). The z-score is
    (r_obs - mean_null) / std_null (caller computes).
    """
    n = P.shape[0]
    s_obs = np.zeros(n, dtype=np.float32)
    s_obs[seed_indices] = 1.0
    s_obs = s_obs / s_obs.sum()
    r_obs, _ = rwr_power(P, s_obs, alpha=alpha)

    perm_indices = degree_stratified_null(deg, seed_indices, n_perm, rng_seed=rng_seed)
    sum_r = np.zeros(n, dtype=np.float64)
    sum_r2 = np.zeros(n, dtype=np.float64)
    for p in range(n_perm):
        s_p = np.zeros(n, dtype=np.float32)
        s_p[perm_indices[p]] = 1.0
        s_p = s_p / s_p.sum()
        r_p, _ = rwr_power(P, s_p, alpha=alpha)
        sum_r += r_p
        sum_r2 += r_p * r_p
        if progress and (p + 1) % 100 == 0:
            print(f"  NPI null {p+1}/{n_perm}")
    mean_null = (sum_r / n_perm).astype(np.float32)
    var_null = (sum_r2 / n_perm) - mean_null.astype(np.float64) ** 2
    std_null = np.sqrt(np.maximum(var_null, 0.0)).astype(np.float32)
    return r_obs, mean_null, std_null


def top_k_by_zscore(
    r_obs: np.ndarray,
    mean_null: np.ndarray,
    std_null: np.ndarray,
    gene_names: Sequence[str],
    k: int = 100,
    exclude: set[str] | None = None,
    std_floor_quantile: float = 0.10,
) -> list[tuple[str, float, float]]:
    """Top-k genes by z = (r_obs - mean_null) / std_null. Returns
    list of (gene, z, r_obs) tuples sorted descending by z.

    To prevent z-score explosion at low-degree / never-sampled nodes (where
    std_null ≈ 0 by Monte-Carlo undersampling), we floor std_null at the
    `std_floor_quantile`-quantile of all positive std values (default: 10th
    percentile). This is a standard variance-stabilizing trick for RWR null
    distributions (DADA tradition).
    """
    pos_std = std_null[std_null > 0]
    if len(pos_std) > 0:
        floor = float(np.quantile(pos_std, std_floor_quantile))
    else:
        floor = 1e-9
    std_safe = np.maximum(std_null, floor)
    z = (r_obs - mean_null) / std_safe
    order = np.argsort(-z)
    out = []
    excl = exclude or set()
    for i in order:
        g = gene_names[i]
        if g in excl:
            continue
        out.append((g, float(z[i]), float(r_obs[i])))
        if len(out) >= k:
            break
    return out


__all__ = [
    "column_normalize",
    "rwr_power",
    "build_seed_vector",
    "degree_stratified_null",
    "npi_zscore",
    "top_k_by_zscore",
]
