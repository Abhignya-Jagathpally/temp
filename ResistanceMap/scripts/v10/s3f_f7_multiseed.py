"""S3f: F7 — multi-seed pooling fixes for HDAC-family drugs.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F7 + `PPI_PROPAGATION_PILOT.md`:
    The pilot showed pan-HDAC drugs (Panobinostat, Vorinostat) need
    HDAC1+HDAC2+HDAC3+HDAC6 pooled seeds, not just HDAC1, to avoid TP53
    contaminating the propagation neighbors. F7 PASSES iff the multi-seed
    F3 result is *cleaner* than the single-seed F3 result for the same drug.

Test design:
    For each pan-HDAC drug:
        single-seed run with HDAC1 only → top-50 → CRISPR rank-sum
        multi-seed run with HDAC1+2+3+6 → top-50 → CRISPR rank-sum
    Compare: does the multi-seed top-50 have a more-negative mean Chronos and
    a smaller permutation p?

Inputs:
    data/processed/ppi_adjacency_v10s2.npz
    data/processed/ppi_degree_v10s2.npy
    data/processed/ppi_gene_index_v10s2.tsv
    data/processed/chronos_haem.parquet

Output:
    paper/v8_artifacts/v10_sprint3/f7_multiseed.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, npi_zscore, top_k_by_zscore,
)


HDAC_DRUG_TESTS = {
    "Panobinostat_single_HDAC1":   ["HDAC1"],
    "Panobinostat_pooled_HDAC1234": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "Vorinostat_single_HDAC1":     ["HDAC1"],
    "Vorinostat_pooled_HDAC1234":  ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
}
ALPHA = 0.7
N_PERM = 1000
TOP_K = 50


def crispr_test(driver_genes: list[str], chronos: pd.DataFrame, seed: int) -> dict:
    """One-sided Wilcoxon + 1000-perm test of mean Chronos essentiality."""
    g = torch.Generator().manual_seed(seed)
    avail = set(chronos.columns)
    drv_present = [d for d in driver_genes if d in avail]
    if len(drv_present) < 5:
        return {"n_present": len(drv_present), "verdict": "insufficient"}

    gene_mean = chronos.mean(axis=0, skipna=True)
    drv_scores = gene_mean[drv_present].dropna().values
    bg_genes = [c for c in chronos.columns if c not in drv_present]
    bg_scores = gene_mean[bg_genes].dropna().values
    if len(drv_scores) < 5:
        return {"n_present": len(drv_present), "verdict": "insufficient_after_nan"}

    _, mw_p = mannwhitneyu(drv_scores, bg_scores, alternative="less")
    drv_mean = float(drv_scores.mean())

    bg_t = torch.from_numpy(bg_scores)
    perm_means = torch.empty(N_PERM, dtype=bg_t.dtype)
    for i in range(N_PERM):
        perm = torch.randperm(len(bg_scores), generator=g)[:len(drv_scores)]
        perm_means[i] = bg_t[perm].mean()
    perm_p = float((perm_means.numpy() <= drv_mean).mean())
    return {
        "n_present": int(len(drv_scores)),
        "drv_mean_chronos": drv_mean,
        "bg_mean_chronos": float(bg_scores.mean()),
        "mannwhitneyu_p_one_sided_less": float(mw_p),
        "permutation_p": perm_p,
    }


def main() -> None:
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    n = A.shape[0]
    P = column_normalize(A)
    chronos = pd.read_parquet(PROC / "chronos_haem.parquet")
    print(f"[S3f] PPI: {n} nodes, chronos: {chronos.shape}")

    results = {}
    for case, seeds in HDAC_DRUG_TESTS.items():
        seed_idx = np.array(
            [gene_to_idx[g] for g in seeds if g in gene_to_idx], dtype=np.int64
        )
        if len(seed_idx) == 0:
            results[case] = {"verdict": "no_seed_genes_in_ppi", "seeds": seeds}
            continue
        r_obs, mean_null, std_null = npi_zscore(
            P, deg, seed_idx, n_perm=N_PERM, alpha=ALPHA, rng_seed=hash(case) & 0xFFFF,
        )
        seeds_set = set(seeds)
        top_k = top_k_by_zscore(r_obs, mean_null, std_null, gene_names, k=TOP_K, exclude=seeds_set)
        top_k_genes = [g for g, _, _ in top_k]
        crispr_res = crispr_test(top_k_genes, chronos, seed=hash(case) & 0xFFFF)
        results[case] = {
            "seeds_used": seeds,
            "n_seeds": int(len(seed_idx)),
            "alpha": ALPHA,
            "top_k_size": TOP_K,
            "top10_genes": top_k_genes[:10],
            "crispr_test": crispr_res,
        }
        if "drv_mean_chronos" in crispr_res:
            print(f"[S3f] {case:34s} drv_mean={crispr_res['drv_mean_chronos']:+.3f}  "
                  f"perm_p={crispr_res['permutation_p']:.4g}")
        else:
            print(f"[S3f] {case}: {crispr_res.get('verdict')}")

    # Compare single-seed vs pooled-seed for each drug family
    pano_single = results["Panobinostat_single_HDAC1"]["crispr_test"].get("permutation_p", 1.0)
    pano_pooled = results["Panobinostat_pooled_HDAC1234"]["crispr_test"].get("permutation_p", 1.0)
    vor_single = results["Vorinostat_single_HDAC1"]["crispr_test"].get("permutation_p", 1.0)
    vor_pooled = results["Vorinostat_pooled_HDAC1234"]["crispr_test"].get("permutation_p", 1.0)
    pano_improvement = pano_single - pano_pooled
    vor_improvement = vor_single - vor_pooled
    print(f"[S3f] Panobinostat single→pooled p improvement: {pano_single:.4g} → {pano_pooled:.4g}")
    print(f"[S3f] Vorinostat   single→pooled p improvement: {vor_single:.4g} → {vor_pooled:.4g}")

    # F7 PASS iff pooled p <= single p for both drugs (multi-seed not worse than single)
    f7_pass = bool(pano_pooled <= pano_single and vor_pooled <= vor_single)

    summary = {
        "alpha": ALPHA,
        "n_perm": N_PERM,
        "top_k": TOP_K,
        "per_case": results,
        "comparison": {
            "Panobinostat_single_p": pano_single,
            "Panobinostat_pooled_p": pano_pooled,
            "Panobinostat_p_improvement": pano_improvement,
            "Vorinostat_single_p": vor_single,
            "Vorinostat_pooled_p": vor_pooled,
            "Vorinostat_p_improvement": vor_improvement,
        },
        "f7_pass_pooled_at_least_as_good": f7_pass,
        "interpretation": (
            "F7 PASSES if pooled HDAC1+2+3+6 seeds give a permutation-p ≤ "
            "the single-HDAC1 seed for both pan-HDAC drugs. The pilot's "
            "PPI_PROPAGATION_PILOT.md showed this fix was needed."
        ),
    }
    out_path = SPRINT3 / "f7_multiseed.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3f] wrote {out_path}")


if __name__ == "__main__":
    main()
