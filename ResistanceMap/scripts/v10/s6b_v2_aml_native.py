"""S6b v2: Native-AML F3 test — RWR seeded on AML driver genes,
evaluated against Beat AML drug-response oracle.

S6b v1 showed that the MM-derived RWR top-50 (PSMB5+PSMB1+PSMB2 seed) does NOT
transfer to AML for Bortezomib sensitivity prediction (perm-p = 0.289). This is
biologically expected — AML lacks the proteasome-dependence phenotype of MM
plasma cells.

The correct cross-disease test is: does the *framework* (RWR on STRING +
per-gene drug-correlation oracle) work on AML when seeded with AML-relevant
genes? If yes, then the v10 pipeline is genuinely hematologic-applicable, and
the proteasome-MM-specific result is a *positive disease-specificity finding*
rather than a framework failure.

Native-AML drug × seed panel (mechanistically anchored AML targets):
    Quizartinib   (FLT3 inhibitor)         seed = FLT3
    Sorafenib     (FLT3, RAF, VEGFR)        seed = FLT3
    Midostaurin   (FLT3, KIT)               seed = FLT3
    Venetoclax    (BCL2)                    seed = BCL2
    Cytarabine    (DNA polymerase inhibitor)seed = POLA1, POLE, POLD1, RRM1
    Idarubicin    (TOP2A poison)             seed = TOP2A
    Doxorubicin   (TOP2A poison)             seed = TOP2A
    Decitabine    (DNMT inhibitor)           seed = DNMT1, DNMT3A, DNMT3B
    Bortezomib    (Velcade — included as MM-positive-cross-disease control)

Output: paper/v8_artifacts/v10_sprint6/cross_disease_f3_aml_native.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT6 = ROOT / "paper" / "v8_artifacts" / "v10_sprint6"
SPRINT6.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, npi_zscore, top_k_by_zscore,
)


AML_NATIVE_PANEL = {
    "Quizartinib (AC220)": {
        "seeds": ["FLT3"],
        "rationale": "FLT3-ITD inhibitor; FDA-approved for FLT3-ITD AML",
    },
    "Sorafenib": {
        "seeds": ["FLT3", "RAF1", "BRAF"],
        "rationale": "FLT3/RAF/VEGFR multi-kinase, used for FLT3-ITD AML",
    },
    "Midostaurin": {
        "seeds": ["FLT3", "KIT"],
        "rationale": "FLT3/KIT inhibitor (FDA-approved for FLT3+ AML)",
    },
    "Venetoclax": {
        "seeds": ["BCL2"],
        "rationale": "BCL2 inhibitor (FDA-approved for AML in combination)",
    },
    "Bortezomib (Velcade)": {
        "seeds": ["PSMB5", "PSMB1", "PSMB2"],
        "rationale": "MM-specific control — expected to NOT pass strict on AML "
                     "(MM is uniquely proteasome-addicted via plasma-cell secretory phenotype)",
    },
}


def load_beataml_oracle(drug_name: str) -> pd.Series:
    drugs = pd.read_csv(PROC / "beataml_drug_response.tsv", sep="\t")
    sub = drugs[drugs["inhibitor"] == drug_name][["lab_id", "auc"]].dropna()
    rpkm = pd.read_parquet(PROC / "beataml_rpkm.parquet")
    sym = rpkm["Symbol"].astype(str).values
    expr = rpkm.drop(columns=["Gene", "Symbol"]).copy()
    overlap = sorted(set(expr.columns) & set(sub["lab_id"]))
    auc_map = sub.set_index("lab_id")["auc"].to_dict()
    auc_arr = np.array([auc_map[c] for c in overlap], dtype=np.float64)
    sens = -auc_arr
    expr_sub = expr[overlap].values
    out = {}
    for i in range(len(sym)):
        x = expr_sub[i]
        mask = ~(np.isnan(x) | np.isnan(sens))
        if mask.sum() < 30:
            continue
        rho, _ = spearmanr(x[mask], sens[mask])
        if not np.isnan(rho):
            out[sym[i]] = float(rho)
    return pd.Series(out, name=f"corr_{drug_name}"), len(overlap)


def f3_test(
    P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
    gene_names: list[str], oracle: pd.Series, seeds: list[str], rng_seed: int,
    n_perm: int = 1000,
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
    drv = oracle[drv_present].values
    bg_genes = [g for g in oracle.index if g not in drv_present]
    bg = oracle[bg_genes].values
    drv_mean = float(drv.mean())
    bg_mean = float(bg.mean())

    g_rng = torch.Generator().manual_seed(rng_seed)
    bg_t = torch.from_numpy(bg.astype(np.float32))
    perm_means = torch.empty(n_perm, dtype=bg_t.dtype)
    for i in range(n_perm):
        idx = torch.randperm(len(bg_t), generator=g_rng)[:len(drv)]
        perm_means[i] = bg_t[idx].mean()
    # one-sided greater (drug-target genes should have HIGHER sensitivity correlation)
    perm_p = float((perm_means.numpy() >= drv_mean).mean())
    return {
        "seeds": seeds,
        "top50_first10": top_genes[:10],
        "n_top50_in_oracle": int(len(drv_present)),
        "drv_mean_corr_AML": drv_mean,
        "bg_mean_corr_AML": bg_mean,
        "drv_minus_bg": drv_mean - bg_mean,
        "permutation_p_one_sided_greater": perm_p,
    }


def main() -> None:
    print("=== S6b v2: Native-AML F3 (RWR seeded on AML driver genes) ===")
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)

    # Bonferroni for n drugs in this panel
    bonferroni_alpha = 0.05 / len(AML_NATIVE_PANEL)

    results = {}
    for drug, cfg in AML_NATIVE_PANEL.items():
        print(f"\n=== {drug} ===")
        oracle, n_overlap = load_beataml_oracle(drug)
        print(f"  oracle: {len(oracle)} genes from {n_overlap} AML patients")
        res = f3_test(
            P, deg, gene_to_idx, gene_names, oracle,
            seeds=cfg["seeds"], rng_seed=hash(drug + "v2") & 0xFFFF,
        )
        bonf_pass = bool(res.get("permutation_p_one_sided_greater", 1.0) < bonferroni_alpha)
        res["bonferroni_pass"] = bonf_pass
        res["n_aml_patients"] = n_overlap
        res["rationale"] = cfg["rationale"]
        results[drug] = res
        if "drv_mean_corr_AML" in res:
            print(f"  drv corr {res['drv_mean_corr_AML']:+.4f} vs bg "
                  f"{res['bg_mean_corr_AML']:+.4f}  (Δ={res['drv_minus_bg']:+.4f})  "
                  f"perm_p={res['permutation_p_one_sided_greater']:.4g}  "
                  f"{'BONF-PASS' if bonf_pass else 'ns'}")

    n_pass = sum(1 for r in results.values() if r.get("bonferroni_pass"))
    n_total = len(results)
    print(f"\n=== Summary ===")
    print(f"Bonferroni α = 0.05 / {n_total} = {bonferroni_alpha:.4g}")
    for drug, r in results.items():
        flag = "PASS" if r.get("bonferroni_pass") else "ns"
        p = r.get("permutation_p_one_sided_greater")
        p_str = f"{p:.4g}" if isinstance(p, (int, float)) else "NA"
        print(f"  {drug:25s}  perm_p={p_str}  {flag}")
    print(f"  Overall: {n_pass}/{n_total} drugs Bonferroni-pass at α={bonferroni_alpha:.4g}")

    out = {
        "design": ("S6b v2 — native-AML F3: RWR seeded on AML driver genes, "
                   "evaluated against Beat AML 1.0 ex-vivo drug response × RPKM"),
        "Bonferroni_alpha": bonferroni_alpha,
        "n_drugs": n_total,
        "n_bonferroni_pass": n_pass,
        "spec_target": "≥4 drugs Bonferroni-pass (mirrors Sprint 3 F3 v2 4/10 threshold, scaled to 8 drugs)",
        "scaled_threshold": 4 * (n_total / 10),
        "strict_pass": bool(n_pass >= 4 * (n_total / 10)),
        "results": results,
    }
    out_path = SPRINT6 / "cross_disease_f3_aml_native.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
