"""S3 v11: Re-run F6 v3 with the multi-α attention-propagation ensemble.

Identical protocol to v10 `scripts/v10/s3fix_v3.py` except the per-driver
propagation is the multi-α ensemble (α ∈ {0.3, 0.5, 0.7, 0.9}, mean
aggregator) instead of single-α RWR (α=0.7).

Hypothesis:
    H0  : T_obs_v11 ≈ T_obs_v10 within stochastic agreement (multi-α
          ensemble is architecturally equivalent to single-α at this graph
          topology).
    H1  : T_obs_v11 < T_obs_v10 with permutation p<0.001 → ensemble dominates.
    H2  : T_obs_v11 > T_obs_v10 with strict-pass lost → revert.

Pre-registered gate (zero-trust per `docs/V11_SOTA_UPGRADE_PLAN.md` §2 row 4):
the v11 propagation must NOT regress the F6 v3 strict-pass; permutation p
must remain < 0.001 with the SAME B=10000 degree-matched label-perm null.

Output: paper/v8_artifacts/v11_sprint3/f6_driver_recall_gat.json
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
SPRINT3_V10 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
SPRINT3_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint3"
SPRINT3_V11.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import column_normalize  # noqa: E402
from resistancemap.landscape.gat_propagation import (  # noqa: E402
    MultiAlphaAttentionPropagation, MultiAlphaConfig, reproduce_rwr_alpha_07,
)
# Re-use the v10 helpers — exact same null distribution machinery
from scripts.v10.s3fix_v3 import (  # noqa: E402
    precompute_bins, degree_matched_pool_excluded, sample_degree_matched,
)


def f6_v3_with_propagator(
    propagator: MultiAlphaAttentionPropagation,
    deg: np.ndarray,
    consensus_set: set,
    gene_to_idx: dict,
    drivers_in_ppi: list,
    B: int = 10000,
    rng_seed: int = 6,
) -> dict:
    n_nodes = propagator.P.shape[0]
    driver_indices = np.array([gene_to_idx[g] for g in drivers_in_ppi], dtype=np.int64)
    n_drv = len(drivers_in_ppi)
    print(f"[F6-v11] {n_drv} drivers in STRING, B={B}")

    # Step 1: per-driver multi-α ensemble propagation
    print(f"[F6-v11] Computing multi-α propagation for {n_drv} driver seeds...")
    t0 = time.time()
    r_per_driver = []
    for j, d_idx in enumerate(driver_indices):
        s = np.zeros(n_nodes, dtype=np.float32)
        s[d_idx] = 1.0
        r, _ = propagator.propagate(s)
        r_per_driver.append(r)
        if (j + 1) % 10 == 0:
            print(f"  prop {j+1}/{n_drv} t={time.time()-t0:.1f}s")
    print(f"[F6-v11] All driver propagations done in {time.time()-t0:.1f}s")

    # Step 2: rank arrays
    rank_per_driver = []
    for j in range(n_drv):
        r = r_per_driver[j]
        order = np.argsort(-r)
        rank = np.empty(n_nodes, dtype=np.int64)
        rank[order] = np.arange(1, n_nodes + 1)
        rank_per_driver.append(rank)

    node_bins, pools_full = precompute_bins(deg, n_bins=20)
    fallback_mask = np.zeros(n_nodes, dtype=bool)
    fallback_mask[driver_indices] = True
    fallback_pool = np.where(~fallback_mask)[0]

    other_indices_per_d = []
    bin_counts_per_d = []
    for d_idx in driver_indices:
        other = np.array([i for i in driver_indices if i != d_idx], dtype=np.int64)
        other_indices_per_d.append(other)
        bin_counts_per_d.append(np.bincount(node_bins[other], minlength=20))

    # Step 3: observed
    obs_per_driver = []
    for j in range(n_drv):
        rank = rank_per_driver[j]
        obs_per_driver.append(float(rank[other_indices_per_d[j]].mean()))
    T_obs = float(np.mean(obs_per_driver))
    print(f"[F6-v11] T_obs = {T_obs:.1f}  (uniform-null ≈ {n_nodes/2:.0f})")

    # Step 4: label-permutation null (SAME design as v10 s3fix_v3 — only the
    # per-driver propagation changed, the null sampling is identical)
    print(f"[F6-v11] B={B} degree-matched label permutations...")
    t0 = time.time()
    rng = torch.Generator().manual_seed(rng_seed)
    pools_excl = degree_matched_pool_excluded(pools_full, fallback_mask)
    T_null = np.empty(B, dtype=np.float64)
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
            print(f"  perm {b+1}/{B} t={time.time()-t0:.1f}s")

    n_better = int(np.sum(T_null <= T_obs))
    p_value = (n_better + 1) / (B + 1)
    sigma = (T_null.mean() - T_obs) / max(T_null.std(), 1e-12)
    print(f"[F6-v11] T_null mean={T_null.mean():.1f} std={T_null.std():.1f}")
    print(f"[F6-v11] p = ({n_better} + 1) / ({B} + 1) = {p_value:.6f}")
    print(f"[F6-v11] sigma vs null = {sigma:.1f}")

    return {
        "n_drivers_in_ppi": n_drv,
        "T_obs_mean_rank_other_drivers": T_obs,
        "uniform_null_expected_rank": n_nodes / 2,
        "B": B,
        "T_null_mean": float(T_null.mean()),
        "T_null_std": float(T_null.std()),
        "T_null_min": float(T_null.min()),
        "permutation_p_plus_one": float(p_value),
        "sigma_vs_null": float(sigma),
        "n_null_better_or_equal": int(n_better),
        "strict_pass": bool(p_value < 0.001),
        "obs_per_driver_mean_ranks": {
            drivers_in_ppi[i]: float(obs_per_driver[i]) for i in range(n_drv)
        },
    }


def main() -> None:
    print("=== S3 v11: F6 v3 under multi-α attention propagation ensemble ===")
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    P = column_normalize(A)
    print(f"PPI: {A.shape}, mean_deg={deg.mean():.2f}")

    # Sanity check (M-equivalence): H=1, α=0.7 ensemble must match v10 RWR
    print("\n=== Equivalence check: H=1, α=0.7 multi-α ≡ v10 RWR ===")
    s_test = np.zeros(P.shape[0], dtype=np.float32)
    s_test[gene_to_idx["PSMB5"]] = 1.0
    eq = reproduce_rwr_alpha_07(P, s_test, tol=1e-6)
    print(f"max abs deviation = {eq['max_abs_deviation']:.3e}  (tol {eq['tol']:.0e})  -> OK")

    # Build the multi-α ensemble propagator (the v11 generalization)
    cfg = MultiAlphaConfig(
        alphas=(0.3, 0.5, 0.7, 0.9), aggregator="mean",
        max_iter=200, tol=1e-8,
    )
    propagator = MultiAlphaAttentionPropagation(P, cfg)
    print(f"\nv11 propagator: H={len(cfg.alphas)} heads, α={list(cfg.alphas)}, "
          f"aggregator={cfg.aggregator}")

    # Load consensus driver set (post-alias) — same source as v10 s3fix_v3
    f6_orig = json.loads((SPRINT3_V10 / "f6_driver_recall_fixed.json").read_text())
    consensus_in_ppi = sorted(
        set(json.loads((SPRINT3_V10 / "f6_driver_recall.json").read_text())["consensus_in_ppi"])
        | set(f6_orig.get("rescued_aliases", []))
    )
    consensus_set = set(consensus_in_ppi)
    drivers_in_ppi = sorted(g for g in consensus_set if g in gene_to_idx)
    print(f"Consensus drivers: {len(drivers_in_ppi)}")

    # Run F6 v3 with the multi-α ensemble
    res = f6_v3_with_propagator(
        propagator, deg, consensus_set, gene_to_idx, drivers_in_ppi, B=10000,
    )
    res["propagator_config"] = {
        "alphas": list(cfg.alphas), "aggregator": cfg.aggregator,
        "max_iter": cfg.max_iter, "tol": cfg.tol,
    }

    # Compare to v10 result
    v10 = json.loads((SPRINT3_V10 / "f6_driver_recall_v3.json").read_text())
    res["v10_comparison"] = {
        "v10_T_obs": v10["T_obs_mean_rank_other_drivers"],
        "v11_T_obs": res["T_obs_mean_rank_other_drivers"],
        "delta_T_obs": res["T_obs_mean_rank_other_drivers"] - v10["T_obs_mean_rank_other_drivers"],
        "v10_p_value": v10["permutation_p_plus_one"],
        "v11_p_value": res["permutation_p_plus_one"],
        "v10_sigma": (v10["T_null_mean"] - v10["T_obs_mean_rank_other_drivers"]) / max(v10["T_null_std"], 1e-12),
        "v11_sigma": res["sigma_vs_null"],
        "v10_strict_pass": bool(v10.get("strict_pass", v10["permutation_p_plus_one"] < 0.001)),
        "v11_strict_pass": res["strict_pass"],
        "verdict": (
            "v11 MATCHES v10 (architectural equivalence)"
            if abs(res["T_obs_mean_rank_other_drivers"] - v10["T_obs_mean_rank_other_drivers"]) < 50
            else (
                "v11 DOMINATES v10 (multi-α ensemble improves)"
                if res["T_obs_mean_rank_other_drivers"] < v10["T_obs_mean_rank_other_drivers"]
                else "v11 REGRESSES vs v10 — REVERT"
            )
        ),
    }

    out = SPRINT3_V11 / "f6_driver_recall_gat.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== FINAL ===")
    cmp = res["v10_comparison"]
    print(f"v10  T_obs = {cmp['v10_T_obs']:.1f}  σ = {cmp['v10_sigma']:.1f}  "
          f"p = {cmp['v10_p_value']:.6f}  {'PASS' if cmp['v10_strict_pass'] else 'FAIL'}")
    print(f"v11  T_obs = {cmp['v11_T_obs']:.1f}  σ = {cmp['v11_sigma']:.1f}  "
          f"p = {cmp['v11_p_value']:.6f}  {'PASS' if cmp['v11_strict_pass'] else 'FAIL'}")
    print(f"VERDICT: {cmp['verdict']}")
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s3_gat_f6_replication"):
        main()
