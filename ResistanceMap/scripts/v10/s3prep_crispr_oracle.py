"""S3-prep companion: CRISPR-oracle skeleton for v10 falsification gate F3.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 row F3:
    "CRISPR rank-sum on top-k driver genes vs 1000 random gene sets.
     Below 95th percentile → attribution claim fails."

This script provides a callable `crispr_oracle(driver_genes, ...)` that
implements the one-sided Wilcoxon rank-sum test of Chronos essentiality
scores in the haematologic DepMap subset for a candidate driver-gene set
versus matched-size random gene sets drawn (with torch RNG) from the
remainder of the Chronos gene index.

The function signature matches what Sprint 3 will plug RWR-derived driver
sets into. For Sprint 1, it is run as a smoke test against the empirical
PSMB5→proteasome 26S-subunit set used in the pilot (PMID 21536909-style
known mechanism). Expected: rank-sum p < 0.001 against 1000 random sets,
matching the pilot's 9.2e-38 result on the smaller 3-MM subset.

Inputs:
    data/processed/chronos_haem.parquet
Outputs:
    paper/v8_artifacts/v10_sprint1/s3prep_oracle_smoke.json
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


def crispr_oracle(
    driver_genes: list[str],
    chronos_df: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Wilcoxon rank-sum essentiality test for a candidate driver-gene set.

    Per the spec: a driver gene set is "candidate" if its mean Chronos
    essentiality is significantly more-negative than random gene sets of
    matched size, drawn from the rest of the Chronos gene index.

    Returns:
        dict with mean_chronos, p_one_sided, perm_p, percentile, refuted.
    """
    g = torch.Generator().manual_seed(seed)
    available = set(chronos_df.columns)
    drivers_present = [d for d in driver_genes if d in available]
    if len(drivers_present) < 3:
        return {
            "n_driver_genes_present": len(drivers_present),
            "verdict": "insufficient_genes",
        }

    # Mean across cell lines per gene → driver vs background distribution
    gene_mean = chronos_df.mean(axis=0)  # (n_genes,)
    drv_scores = gene_mean[drivers_present].values
    drv_mean = float(drv_scores.mean())
    bg_genes = [g for g in available if g not in drivers_present]
    bg_scores_full = gene_mean[bg_genes].values

    # Mann-Whitney one-sided (drivers more negative than background)
    u_stat, mw_p = mannwhitneyu(drv_scores, bg_scores_full, alternative="less")

    # Permutation null: draw 1000 random size-matched gene sets, compare means
    n_drv = len(drivers_present)
    n_bg = len(bg_genes)
    bg_t = torch.from_numpy(bg_scores_full)
    perm_means = torch.empty(n_perm, dtype=bg_t.dtype)
    for i in range(n_perm):
        perm = torch.randperm(n_bg, generator=g)[:n_drv]
        perm_means[i] = bg_t[perm].mean()
    perm_means_np = perm_means.numpy()
    perm_p = float((perm_means_np <= drv_mean).mean())
    pct = float((perm_means_np <= drv_mean).mean() * 100)

    return {
        "n_driver_genes_input": len(driver_genes),
        "n_driver_genes_present": len(drivers_present),
        "drv_mean_chronos": drv_mean,
        "bg_mean_chronos": float(bg_scores_full.mean()),
        "mannwhitneyu_p_one_sided_less": float(mw_p),
        "permutation_p": perm_p,
        "percentile_in_null": pct,
        "n_permutations": n_perm,
        "refuted_at_5pct": bool(perm_p > 0.05),
    }


def main() -> None:
    chronos_path = PROC / "chronos_haem.parquet"
    if not chronos_path.exists():
        sys.exit(f"missing {chronos_path}")
    print(f"[S3-prep] reading {chronos_path}")
    chronos_df = pd.read_parquet(chronos_path)
    print(f"[S3-prep] chronos_haem shape: {chronos_df.shape}")

    # Smoke-test driver set: 26S proteasome subunits (PSMA*, PSMB*, PSMC*, PSMD*).
    # This matches the PPI-pilot result (PSMB5 → proteasome p=9.2e-38) and serves
    # as a positive-control sanity check that the oracle wiring is sound.
    proteasome = [c for c in chronos_df.columns if c.startswith(("PSMA", "PSMB", "PSMC", "PSMD"))]
    print(f"[S3-prep] proteasome smoke set: {len(proteasome)} genes (PSM*)")

    result = crispr_oracle(proteasome, chronos_df, n_perm=1000, seed=0)
    print("[S3-prep] proteasome oracle result:")
    for k, v in result.items():
        print(f"    {k}: {v}")

    # Negative control: random size-matched set should fail the test
    g = torch.Generator().manual_seed(42)
    n_drv = len(proteasome)
    all_genes = list(chronos_df.columns)
    perm = torch.randperm(len(all_genes), generator=g)[:n_drv]
    rand_set = [all_genes[int(i)] for i in perm]
    rand_result = crispr_oracle(rand_set, chronos_df, n_perm=1000, seed=1)
    print("[S3-prep] random-control oracle result:")
    for k, v in rand_result.items():
        print(f"    {k}: {v}")

    out = {
        "proteasome_positive_control": result,
        "random_negative_control": rand_result,
    }
    out_path = ROOT / "paper" / "v8_artifacts" / "v10_sprint1" / "s3prep_oracle_smoke.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[S3-prep] wrote {out_path}")


if __name__ == "__main__":
    main()
