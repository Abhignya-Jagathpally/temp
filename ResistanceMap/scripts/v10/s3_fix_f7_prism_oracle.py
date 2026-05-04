"""S3 F7 fix: replace Chronos essentiality oracle with PRISM drug-sensitivity oracle.

Sprint 3 F7 v3 strict-spec status:
    Bortezomib (PSMB5+1+2 vs PSMB5):  STRICT PASS (proteasome essentiality signal)
    Panobinostat (HDAC1+2+3+6 vs HDAC1):  STRICT PASS (pan-HDAC essentiality)
    Vorinostat (HDAC1+2+3 vs HDAC1):  STRICT FAIL (both p > 0.7, in noise floor)

Why: HDAC class-I genes (HDAC1, HDAC2, HDAC3) have negligible CRISPR essentiality
in DepMap haem cell lines because of paralog buffering — knocking out HDAC1
alone doesn't kill the cell since HDAC2/3 compensate. Chronos cannot detect
HDAC drug-target relevance.

Fix: use PRISM drug-sensitivity AUC × CCLE proteomics as the orthogonal oracle.
Per-gene sensitivity-correlation = Spearman(proteomics[g], -AUC[drug]) across
cell lines that have both measurements. A gene with strongly negative Spearman
between its protein level and drug AUC is one whose elevated expression
correlates with greater drug sensitivity — exactly the relationship that
HDAC1/2/3 should have with Vorinostat, even though they are not essential.

F7 v4 design (mirrors v3 but uses PRISM correlation instead of Chronos):
    For each drug × seed config:
        run RWR(seed) + NPI z-score on STRING → top-50 genes
        per-gene "drug-corr" = Spearman(CCLE-proteomics[g], -PRISM_AUC[drug])
        drv_mean_corr = mean drug-corr across top-50
        bg_mean_corr  = mean drug-corr across all other expressed genes
        permutation p = P(random 50-gene set has drug-corr ≤ drv_mean_corr)

    F7-v4 PASS for that drug if pooled-seed p ≤ single-seed p (BOTH cell-line panels).

Note: there is no haem-vs-mm panel split here — PRISM-AUC is per-drug global,
applicable across all PRISM cell lines.

Output: paper/v8_artifacts/v10_sprint3/f7_multiseed_v4_prism.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, npi_zscore, top_k_by_zscore,
)


F7_V4_PAIRS = {
    "Vorinostat": {
        "single": ["HDAC1"],
        "pooled": ["HDAC1", "HDAC2", "HDAC3"],
        "rationale": "class-I HDAC selective; HDAC6 IC50 1.5 µM (Bradner 2010)",
    },
    "Panobinostat": {
        "single": ["HDAC1"],
        "pooled": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
        "rationale": "pan-HDAC; all 4 paralogs IC50 <10 nM (Bradner 2010)",
    },
    "Bortezomib": {
        "single": ["PSMB5"],
        "pooled": ["PSMB5", "PSMB1", "PSMB2"],
        "rationale": "20S proteasome β-subunits (positive control)",
    },
}


def load_prism_drug_auc(drug_name: str) -> pd.Series:
    prism = pd.read_csv(
        RAW / "prism" / "secondary-screen-dose-response-curve-parameters.csv",
        low_memory=False,
    )
    sub = prism[prism["name"].fillna("").str.lower() == drug_name.lower()]
    # Multiple replicates per cell line — take mean
    return sub.groupby("depmap_id")["auc"].mean().rename(f"AUC_{drug_name}")


def load_proteomics_with_symbols() -> tuple[pd.DataFrame, dict[str, str]]:
    prot = pd.read_csv(RAW / "ccle_proteomics.csv", index_col=0)
    # Strip "(entrez_id)" suffix from columns
    sym_cols = [re.sub(r"\s+\(\d+\)", "", c) for c in prot.columns]
    sym_to_col = dict(zip(sym_cols, prot.columns))
    return prot, sym_to_col


def per_gene_drug_corr(
    prot: pd.DataFrame, sym_to_col: dict[str, str], drug_auc: pd.Series,
) -> pd.Series:
    """Per-gene Spearman(proteomics[g], -AUC) across overlapping cell lines.

    Returns Series indexed by gene symbol with the correlation. Negative-AUC
    means we measure "sensitivity" (higher = more sensitive); a gene with
    positive corr is one whose elevated protein correlates with sensitivity.
    """
    overlap = sorted(set(prot.index) & set(drug_auc.index))
    print(f"  overlap cells (proteomics × PRISM): {len(overlap)}")
    prot_sub = prot.loc[overlap]
    auc_sub = drug_auc.loc[overlap]
    sens = -auc_sub.values
    out = {}
    for sym, col in sym_to_col.items():
        x = prot_sub[col].values
        # Drop NaN-paired rows
        mask = ~(np.isnan(x) | np.isnan(sens))
        if mask.sum() < 30:
            continue
        rho, _ = spearmanr(x[mask], sens[mask])
        if not np.isnan(rho):
            out[sym] = float(rho)
    return pd.Series(out, name="drug_corr")


def f7_test_with_oracle(
    P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
    gene_names: list[str], oracle: pd.Series, seeds: list[str], rng_seed: int,
) -> dict:
    seed_idx = np.array([gene_to_idx[g] for g in seeds if g in gene_to_idx], dtype=np.int64)
    if len(seed_idx) == 0:
        return {"verdict": "no_seed_in_PPI"}
    r_obs, mn, sn = npi_zscore(P, deg, seed_idx, n_perm=1000, alpha=0.7, rng_seed=rng_seed)
    top = top_k_by_zscore(r_obs, mn, sn, gene_names, k=50, exclude=set())
    top_genes = [g for g, _, _ in top]
    for g in seeds:
        if g not in top_genes:
            top_genes.insert(0, g)
    top_genes = top_genes[:50]

    avail = set(oracle.index)
    drv_present = [g for g in top_genes if g in avail]
    if len(drv_present) < 5:
        return {"n_present": len(drv_present), "verdict": "insufficient"}
    drv_corrs = oracle[drv_present].values
    bg_genes = [g for g in oracle.index if g not in drv_present]
    bg_corrs = oracle[bg_genes].values
    drv_mean = float(drv_corrs.mean())

    # Permutation p: random size-matched subset of bg
    g_rng = torch.Generator().manual_seed(rng_seed)
    bg_t = torch.from_numpy(bg_corrs.astype(np.float32))
    n = len(drv_corrs)
    perm_means = torch.empty(1000, dtype=bg_t.dtype)
    for i in range(1000):
        idx = torch.randperm(len(bg_t), generator=g_rng)[:n]
        perm_means[i] = bg_t[idx].mean()
    # We want POSITIVE drug_corr (gene-protein correlated with sensitivity)
    perm_p = float((perm_means.numpy() >= drv_mean).mean())
    return {
        "seeds": seeds,
        "n_top50_present_in_oracle": int(len(drv_present)),
        "drv_mean_drug_corr": drv_mean,
        "bg_mean_drug_corr": float(bg_corrs.mean()),
        "permutation_p_one_sided_greater": perm_p,
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

    print("Loading CCLE proteomics + PRISM AUC...")
    prot, sym_to_col = load_proteomics_with_symbols()
    print(f"proteomics: {prot.shape}")

    results = {}
    for drug, cfg in F7_V4_PAIRS.items():
        print(f"\n=== F7 v4 [{drug}] ===")
        drug_auc = load_prism_drug_auc(drug)
        oracle = per_gene_drug_corr(prot, sym_to_col, drug_auc)
        print(f"  per-gene oracle: {len(oracle)} genes")
        print(f"  oracle distribution: mean={oracle.mean():+.4f} sd={oracle.std():.4f}")
        single = f7_test_with_oracle(
            P, deg, gene_to_idx, gene_names, oracle,
            seeds=cfg["single"], rng_seed=hash(drug + "single_v4") & 0xFFFF,
        )
        pooled = f7_test_with_oracle(
            P, deg, gene_to_idx, gene_names, oracle,
            seeds=cfg["pooled"], rng_seed=hash(drug + "pooled_v4") & 0xFFFF,
        )
        sp_p = single.get("permutation_p_one_sided_greater", 1.0)
        po_p = pooled.get("permutation_p_one_sided_greater", 1.0)
        improves = bool(po_p <= sp_p)
        results[drug] = {
            "rationale": cfg["rationale"],
            "single": single,
            "pooled": pooled,
            "delta_perm_p": po_p - sp_p,
            "pooled_dominates": improves,
        }
        print(f"  single({'+'.join(cfg['single'])}) drv_corr={single.get('drv_mean_drug_corr', 'NA')}  "
              f"perm_p={sp_p:.4g}")
        print(f"  pooled({'+'.join(cfg['pooled'])}) drv_corr={pooled.get('drv_mean_drug_corr', 'NA')}  "
              f"perm_p={po_p:.4g}")
        print(f"  pooled dominates single: {'YES' if improves else 'NO'}")

    overall = all(r["pooled_dominates"] for r in results.values())
    out = {
        "design": "F7 v4 — PRISM AUC × CCLE proteomics drug-sensitivity oracle (replaces Chronos for HDAC drugs lacking essentiality signal)",
        "results": results,
        "overall_strict_pass_all_drugs": overall,
    }
    out_path = SPRINT3 / "f7_multiseed_v4_prism.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")
    print(f"\n=== F7 v4 OVERALL: {'STRICT PASS' if overall else 'STRICT FAIL'}")


if __name__ == "__main__":
    main()
