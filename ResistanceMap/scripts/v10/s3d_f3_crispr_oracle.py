"""S3d: F3 — CRISPR rank-sum oracle for the 10-drug propagation panel.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F3:
    For each drug, take RWR top-k driver genes; one-sided Wilcoxon rank-sum
    of mean Chronos essentiality vs 1000 size-matched random gene sets drawn
    from the haematologic DepMap subset. Bonferroni-corrected. Pilot showed
    4/10 drugs significant on a smaller (3-MM-line) subset; with 64 haem
    lines we expect to maintain or exceed that.

Inputs:
    paper/v8_artifacts/v10_sprint3/drug_propagation.npz
    data/processed/chronos_haem.parquet      # 64 haem cell lines × 18,531 genes
Output:
    paper/v8_artifacts/v10_sprint3/f3_crispr_oracle.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"


def crispr_rank_sum(
    driver_genes: list[str],
    chronos_df: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Wilcoxon + permutation null on per-gene mean Chronos essentiality."""
    g = torch.Generator().manual_seed(seed)
    available = set(chronos_df.columns)
    drv_present = [d for d in driver_genes if d in available]
    if len(drv_present) < 5:
        return {"n_present": len(drv_present), "verdict": "insufficient"}

    gene_mean = chronos_df.mean(axis=0, skipna=True)  # nan-safe
    drv_scores = gene_mean[drv_present].dropna().values
    bg_genes = [c for c in chronos_df.columns if c not in drv_present]
    bg_scores = gene_mean[bg_genes].dropna().values

    if len(drv_scores) < 5 or len(bg_scores) < 100:
        return {"n_present": int(len(drv_present)), "verdict": "insufficient_after_nan"}

    u_stat, mw_p = mannwhitneyu(drv_scores, bg_scores, alternative="less")
    drv_mean = float(drv_scores.mean())
    bg_mean = float(bg_scores.mean())

    # Permutation null via torch RNG
    n_drv_eff = len(drv_scores)
    bg_t = torch.from_numpy(bg_scores)
    perm_means = torch.empty(n_perm, dtype=bg_t.dtype)
    for i in range(n_perm):
        perm = torch.randperm(len(bg_scores), generator=g)[:n_drv_eff]
        perm_means[i] = bg_t[perm].mean()
    perm_p = float((perm_means.numpy() <= drv_mean).mean())
    return {
        "n_input": int(len(driver_genes)),
        "n_present": int(len(drv_present)),
        "n_present_after_nan": int(len(drv_scores)),
        "drv_mean_chronos": drv_mean,
        "bg_mean_chronos": bg_mean,
        "mannwhitneyu_p_one_sided_less": float(mw_p),
        "permutation_p": perm_p,
        "n_permutations": int(n_perm),
        "significant_uncorrected": bool(perm_p < 0.05),
    }


def main() -> None:
    chronos = pd.read_parquet(PROC / "chronos_haem.parquet")
    print(f"[S3d] chronos: {chronos.shape}")

    prop = np.load(SPRINT3 / "drug_propagation.npz", allow_pickle=True)
    drugs = list(prop["drugs"])
    print(f"[S3d] drugs in panel: {drugs}")

    # Top-k cutoff for testing — use top-50 (between 25-100 is standard for MoA tests)
    K = 50
    results = {}
    for drug in drugs:
        top_k_full = list(prop[f"{drug}__top_k_genes"])[:K]
        res = crispr_rank_sum(top_k_full, chronos, n_perm=1000, seed=0)
        results[drug] = res
        if "drv_mean_chronos" in res:
            print(f"[S3d] {drug:14s} drv={res['drv_mean_chronos']:+.3f}  "
                  f"bg={res['bg_mean_chronos']:+.3f}  "
                  f"perm_p={res['permutation_p']:.4g}  "
                  f"{'SIGNIF' if res['significant_uncorrected'] else 'ns'}")
        else:
            print(f"[S3d] {drug}: {res.get('verdict')}")

    # Bonferroni correction on permutation_p across n_drugs
    sig_drugs = [d for d, r in results.items() if r.get("significant_uncorrected")]
    n_drugs = len(results)
    bonferroni_alpha = 0.05 / max(n_drugs, 1)
    sig_bonferroni = [d for d, r in results.items()
                      if r.get("permutation_p", 1.0) < bonferroni_alpha]
    print(f"[S3d] Bonferroni-corrected α={bonferroni_alpha:.4g}; "
          f"{len(sig_bonferroni)}/{n_drugs} drugs significant: {sig_bonferroni}")

    summary = {
        "top_k_tested": K,
        "n_drugs_panel": n_drugs,
        "n_significant_uncorrected_p_lt_0.05": len(sig_drugs),
        "n_significant_bonferroni_p_lt_0.05_div_n": len(sig_bonferroni),
        "bonferroni_alpha": bonferroni_alpha,
        "drugs_significant_uncorrected": sig_drugs,
        "drugs_significant_bonferroni": sig_bonferroni,
        "per_drug": results,
        "f3_pass": bool(len(sig_bonferroni) >= 4),  # spec: pilot showed 4/10
        "spec_threshold": "F3 PASS = ≥4/10 drugs Bonferroni-significant on Chronos rank-sum",
    }
    out_path = SPRINT3 / "f3_crispr_oracle.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3d] wrote {out_path}")


if __name__ == "__main__":
    main()
