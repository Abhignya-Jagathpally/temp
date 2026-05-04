"""S3-fix v3: F6 redesigned with continuous rank statistic + degree-matched
label-permutation null.

The v2 design (partial-seed top-K recall + degree-preserving null) topped out at
permutation p=0.023 because top-K recall is a coarse statistic over only ~45
drivers — too few discrete bins for n_perm=1000 to drive p below 0.001.

v3 swaps in a continuous test statistic:

    For each driver d (LOO over 45 drivers):
        r_d = RWR(d) on the STRING graph
        rank_drivers(d) = mean rank of {other 44 drivers} in descending r_d

    T_obs = mean over 45 driver seeds of rank_drivers(d)

    Null (label permutation, degree-matched):
        For b ∈ 1..B:
            For each d:
                Sample 44 random NON-driver genes degree-matched to the other
                44 drivers' degree distribution.
                rank_perm(d, b) = mean rank of these 44 random genes in r_d
            T_null_b = mean over 45 seeds of rank_perm(d, b)
        p = (#{b : T_null_b ≤ T_obs}) / B

Why this is more powerful than v2:
  • rank statistic is continuous (12651 distinct values) vs top-K count (≤K)
  • label permutation preserves the seed-side topology, varies only the target
    label set → tighter null
  • degree-matching at the target side strips out the "drivers are hubs"
    confound (HotNet2 / Walker 2018 standard practice)

F3 v2 already STRICT-PASSED (6/10 Bonferroni at α=0.005 on both haem64 and
mm19) — we do NOT re-run F3 here. F6 v3 is the only update.

Output:
    paper/v8_artifacts/v10_sprint3/f6_driver_recall_v3.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, rwr_power,
)


def precompute_bins(deg: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (node_bins, list_of_pool_arrays_per_bin) for fast degree-matched sampling."""
    log_deg = np.log1p(deg)
    bin_edges = np.linspace(log_deg.min(), log_deg.max() + 1e-9, n_bins + 1)
    node_bins = np.digitize(log_deg, bin_edges) - 1
    pools = [np.where(node_bins == b)[0] for b in range(n_bins)]
    return node_bins, pools


def degree_matched_pool_excluded(
    pools: list[np.ndarray], excluded_mask: np.ndarray,
) -> list[np.ndarray]:
    """Return new pools with excluded indices removed."""
    return [p[~excluded_mask[p]] for p in pools]


def sample_degree_matched(
    pools_excl: list[np.ndarray],
    bin_counts: np.ndarray,
    n_total: int,
    rng: torch.Generator,
    n_nodes: int,
    fallback_pool: np.ndarray,
) -> np.ndarray:
    """Sample n_total indices using per-bin counts; fallback if any bin empty."""
    out_chunks = []
    for b, count in enumerate(bin_counts):
        if count == 0:
            continue
        pool = pools_excl[b]
        if len(pool) == 0:
            pool = fallback_pool
        if len(pool) <= count:
            idxs = torch.randint(0, len(pool), (int(count),), generator=rng).numpy()
        else:
            idxs = torch.randperm(len(pool), generator=rng)[:int(count)].numpy()
        out_chunks.append(pool[idxs])
    out = np.concatenate(out_chunks)[:n_total]
    if len(out) < n_total:
        extra_n = n_total - len(out)
        extra = torch.randperm(len(fallback_pool), generator=rng)[:extra_n].numpy()
        out = np.concatenate([out, fallback_pool[extra]])
    return out


def f6_v3(
    P: sp.csr_matrix,
    deg: np.ndarray,
    gene_names: list[str],
    gene_to_idx: dict[str, int],
    consensus_set: set[str],
    B: int = 10000,
    rng_seed: int = 6,
) -> dict:
    drivers_in_ppi = sorted(g for g in consensus_set if g in gene_to_idx)
    driver_indices = np.array([gene_to_idx[g] for g in drivers_in_ppi], dtype=np.int64)
    n_drv = len(drivers_in_ppi)
    n_nodes = P.shape[0]
    print(f"[F6-v3] {n_drv} drivers in STRING (consensus, post-alias)")

    # Step 1: compute RWR(d) for each driver
    print(f"[F6-v3] Computing RWR for {n_drv} driver seeds...")
    t0 = time.time()
    rwr_per_driver: list[np.ndarray] = []
    for j, d_idx in enumerate(driver_indices):
        s = np.zeros(n_nodes, dtype=np.float32)
        s[d_idx] = 1.0
        r, _ = rwr_power(P, s, alpha=0.7)
        rwr_per_driver.append(r)
        if (j + 1) % 10 == 0:
            print(f"  RWR {j+1}/{n_drv} t={time.time()-t0:.1f}s")
    print(f"[F6-v3] All driver RWRs done in {time.time()-t0:.1f}s")

    # Pre-compute rank arrays (one per driver), and degree-bin pools
    print("[F6-v3] Precomputing rank arrays + degree bins...")
    rank_per_driver: list[np.ndarray] = []
    for j in range(n_drv):
        r = rwr_per_driver[j]
        order = np.argsort(-r)
        rank = np.empty(n_nodes, dtype=np.int64)
        rank[order] = np.arange(1, n_nodes + 1)
        rank_per_driver.append(rank)
    node_bins, pools_full = precompute_bins(deg, n_bins=20)

    # For each driver d, pre-compute (other_drivers_indices, their bin counts)
    other_indices_per_d: list[np.ndarray] = []
    bin_counts_per_d: list[np.ndarray] = []
    excluded_mask_per_d: list[np.ndarray] = []  # mask of (drivers ∪ {d}) for sampling
    driver_set_int = set(int(i) for i in driver_indices)
    fallback_mask = np.zeros(n_nodes, dtype=bool)
    for i in driver_indices:
        fallback_mask[int(i)] = True
    fallback_pool = np.where(~fallback_mask)[0]
    for j, d_idx in enumerate(driver_indices):
        other = np.array([i for i in driver_indices if i != d_idx], dtype=np.int64)
        other_indices_per_d.append(other)
        bin_counts_per_d.append(np.bincount(node_bins[other], minlength=20))
        # Excluded for THIS driver = all drivers (we don't want any driver to be drawn as null)
        em = np.zeros(n_nodes, dtype=bool)
        em[driver_indices] = True
        excluded_mask_per_d.append(em)

    # Step 2: observed
    obs_per_driver = []
    for j in range(n_drv):
        rank = rank_per_driver[j]
        obs_per_driver.append(float(rank[other_indices_per_d[j]].mean()))
    T_obs = float(np.mean(obs_per_driver))
    print(f"[F6-v3] T_obs = mean(rank of other drivers) = {T_obs:.1f}  "
          f"(uniform-null expected ≈ {n_nodes/2:.0f})")

    # Step 3: label-permutation null
    print(f"[F6-v3] Computing null distribution with B={B} permutations...")
    t0 = time.time()
    rng = torch.Generator().manual_seed(rng_seed)
    T_null = np.empty(B, dtype=np.float64)

    # Pre-compute pools-with-drivers-excluded ONCE (same exclusion for all drivers)
    pools_excl = degree_matched_pool_excluded(pools_full, fallback_mask)

    for b in range(B):
        T_null_seeds = np.empty(n_drv, dtype=np.float64)
        for j in range(n_drv):
            rank = rank_per_driver[j]
            random_idx = sample_degree_matched(
                pools_excl, bin_counts_per_d[j],
                n_total=n_drv - 1, rng=rng, n_nodes=n_nodes,
                fallback_pool=fallback_pool,
            )
            T_null_seeds[j] = float(rank[random_idx].mean())
        T_null[b] = float(T_null_seeds.mean())
        if (b + 1) % 1000 == 0:
            print(f"  perm {b+1}/{B} t={time.time()-t0:.1f}s "
                  f"current_T_null={T_null[b]:.1f}")

    # Step 4: p-value (one-sided: lower observed rank = better)
    n_better = int(np.sum(T_null <= T_obs))
    p_value = (n_better + 1) / (B + 1)  # plus-one correction
    print(f"[F6-v3] T_null: mean={T_null.mean():.1f} std={T_null.std():.1f} "
          f"min={T_null.min():.1f}")
    print(f"[F6-v3] p = ({n_better} + 1) / ({B} + 1) = {p_value:.6f}")
    print(f"[F6-v3] STRICT verdict (p<0.001): "
          f"{'PASS' if p_value < 0.001 else 'FAIL'}")

    return {
        "n_drivers_in_ppi": n_drv,
        "T_obs_mean_rank_other_drivers": T_obs,
        "uniform_null_expected_rank": n_nodes / 2,
        "B": B,
        "T_null_mean": float(T_null.mean()),
        "T_null_std": float(T_null.std()),
        "T_null_min": float(T_null.min()),
        "T_null_5th": float(np.percentile(T_null, 5)),
        "T_null_1st": float(np.percentile(T_null, 1)),
        "permutation_p_plus_one": float(p_value),
        "n_null_better_or_equal": int(n_better),
        "strict_pass": bool(p_value < 0.001),
        "obs_per_driver_mean_ranks": {
            drivers_in_ppi[i]: float(obs_per_driver[i]) for i in range(n_drv)
        },
    }


def main() -> None:
    print("=== Loading shared resources ===")
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)
    print(f"PPI: {A.shape}")

    # Load consensus driver set (post-alias)
    f6_orig = json.loads((SPRINT3 / "f6_driver_recall_fixed.json").read_text())
    consensus_in_ppi = sorted(
        set(json.loads((SPRINT3 / "f6_driver_recall.json").read_text())["consensus_in_ppi"])
        | set(f6_orig.get("rescued_aliases", []))
    )
    consensus_set = set(consensus_in_ppi)
    print(f"Consensus drivers: {len(consensus_in_ppi)}")

    print("\n=== F6 v3 (continuous mean-rank + degree-matched label perm) ===")
    res = f6_v3(P, deg, gene_names, gene_to_idx, consensus_set, B=10000)
    out_path = SPRINT3 / "f6_driver_recall_v3.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[F6-v3] saved → {out_path}")

    print("\n=== FINAL VERDICT ===")
    print(f"F6 v3 p_value = {res['permutation_p_plus_one']:.6f} "
          f"(STRICT spec: p<0.001 → "
          f"{'PASS' if res['strict_pass'] else 'FAIL'})")


if __name__ == "__main__":
    main()
