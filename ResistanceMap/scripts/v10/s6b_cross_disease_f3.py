"""S6b: Cross-disease F3 replication on Beat AML 1.0.

Sprint 3 F3 v2 STRICT-PASSED on DepMap haem64 + mm19 panels (6/10 Bonferroni
significant, including Bortezomib, Carfilzomib, Ixazomib, Doxorubicin,
Etoposide, Selinexor). The MM panel uses MM cell lines; the proteasome RWR
top-50 was derived from PSMB5+PSMB1+PSMB2 seed.

Sprint 6 cross-disease test: take the SAME Sprint-3-derived RWR top-50
(proteasome) and ask whether it predicts ex-vivo Bortezomib sensitivity in the
**Beat AML 1.0 cohort** (Tyner 2018, 448 AML patients with paired RNA-seq +
ex-vivo Bortezomib AUC).

Per-gene "Bortezomib-relevance" oracle for AML:
    bort_corr_AML(g) = -Spearman(RPKM[g], AUC_Bort) across overlapping AML patients
    (negative AUC = sensitivity, so positive corr means high-expression →
     more sensitive)

F3-cross-disease test:
    For each drug × seed config (Bortezomib seed = PSMB5+PSMB1+PSMB2;
    Panobinostat seed = HDAC1+2+3+6 / HDAC1):
        run RWR(seed) → top-50 (using the v10 STRING graph, MM-derived)
        compute mean bort_corr_AML across the 50 genes
        permutation null vs random gene sets
        F3-PASS if perm-p < 0.005 (Bonferroni for 10 drugs) — same threshold as Sprint 3 F3 v2

We also test Panobinostat (208 AML patients) as the second cross-disease drug.

Spec deliverable per §9 row 6: extends "hematologic" claim beyond MM. If MM-
derived RWR top-50 (proteasome) predicts AML ex-vivo Bortezomib sensitivity,
that demonstrates the network neighborhood IS shared across hematologic
malignancies — earning the "hematologic" claim in the v10 title.

Output: paper/v8_artifacts/v10_sprint6/cross_disease_f3.json
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
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
SPRINT6 = ROOT / "paper" / "v8_artifacts" / "v10_sprint6"
SPRINT6.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, npi_zscore, top_k_by_zscore,
)


CROSS_DISEASE_PANEL = {
    "Bortezomib (Velcade)": {
        "seeds": ["PSMB5", "PSMB1", "PSMB2"],
        "rationale": "20S β-subunit (Bortezomib direct target). Same seed used in Sprint 3 F3 v2.",
    },
    "Panobinostat": {
        "seeds": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
        "rationale": "pan-HDAC, all 4 paralogs IC50 < 10 nM (Bradner 2010). Same seed used in Sprint 3 F3 v2 (failed there on Chronos but passed at single-seed in F7 v4 PRISM).",
    },
}


def load_beataml_oracle(drug_name: str) -> pd.Series:
    """Per-gene Spearman(RPKM, -AUC) on Beat AML for a given drug."""
    drugs = pd.read_csv(PROC / "beataml_drug_response.tsv", sep="\t")
    sub = drugs[drugs["inhibitor"] == drug_name][["lab_id", "auc"]].copy()
    sub = sub.dropna()
    print(f"  {drug_name}: {len(sub)} patient measurements")
    rpkm = pd.read_parquet(PROC / "beataml_rpkm.parquet")
    # Index = (Gene Ensembl, Symbol); columns = labId strings
    sym = rpkm["Symbol"].astype(str).values
    expr = rpkm.drop(columns=["Gene", "Symbol"]).copy()
    overlap = sorted(set(expr.columns) & set(sub["lab_id"]))
    print(f"  overlap with RNA-seq: {len(overlap)} patients")
    auc_map = sub.set_index("lab_id")["auc"].to_dict()
    auc_arr = np.array([auc_map[c] for c in overlap], dtype=np.float64)
    sens = -auc_arr  # high = sensitive
    expr_sub = expr[overlap].values  # genes × cells
    out = {}
    for i in range(len(sym)):
        x = expr_sub[i]
        mask = ~(np.isnan(x) | np.isnan(sens))
        if mask.sum() < 30:
            continue
        rho, _ = spearmanr(x[mask], sens[mask])
        if not np.isnan(rho):
            out[sym[i]] = float(rho)
    return pd.Series(out, name=f"corr_{drug_name}")


def f3_cross_disease_test(
    P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
    gene_names: list[str], oracle: pd.Series, seeds: list[str],
    rng_seed: int, n_perm: int = 1000,
) -> dict:
    seed_idx = np.array([gene_to_idx[g] for g in seeds if g in gene_to_idx], dtype=np.int64)
    if len(seed_idx) == 0:
        return {"verdict": "no_seed_in_PPI"}
    print(f"  computing RWR + NPI z-score (seeds: {seeds})...")
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
    perm_p = float((perm_means.numpy() >= drv_mean).mean())  # one-sided greater
    return {
        "seeds": seeds,
        "top50_first10": top_genes[:10],
        "n_top50_in_oracle": int(len(drv_present)),
        "drv_mean_corr_AML": drv_mean,
        "bg_mean_corr_AML": bg_mean,
        "permutation_p_one_sided_greater": perm_p,
        "drv_minus_bg": drv_mean - bg_mean,
    }


def main() -> None:
    print("=== S6b: cross-disease F3 replication on Beat AML 1.0 ===")
    print("Loading STRING PPI (Sprint 2 v10s2)...")
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)

    bonferroni_alpha = 0.005  # same as Sprint 3 F3 v2 (0.05/10 drugs)
    results = {}
    for drug, cfg in CROSS_DISEASE_PANEL.items():
        print(f"\n=== {drug} ===")
        oracle = load_beataml_oracle(drug)
        print(f"  per-gene oracle: {len(oracle)} genes; "
              f"mean={oracle.mean():+.4f}, sd={oracle.std():.4f}")
        res = f3_cross_disease_test(
            P, deg, gene_to_idx, gene_names, oracle,
            seeds=cfg["seeds"], rng_seed=hash(drug) & 0xFFFF,
        )
        bonf_pass = bool(res.get("permutation_p_one_sided_greater", 1.0) < bonferroni_alpha)
        res["bonferroni_pass_at_0p005"] = bonf_pass
        res["rationale"] = cfg["rationale"]
        results[drug] = res
        if "drv_mean_corr_AML" in res:
            print(f"  drv mean corr (top-50): {res['drv_mean_corr_AML']:+.4f}  "
                  f"vs bg {res['bg_mean_corr_AML']:+.4f}  "
                  f"perm_p = {res['permutation_p_one_sided_greater']:.4g}  "
                  f"{'BONF-PASS' if bonf_pass else 'ns'}")

    overall = all(r.get("bonferroni_pass_at_0p005", False) for r in results.values())
    out = {
        "design": ("Cross-disease F3: MM-derived (Sprint 3) STRING/PPI RWR top-50 "
                   "evaluated on Beat AML 1.0 ex-vivo drug response × RPKM"),
        "Bonferroni_alpha": bonferroni_alpha,
        "results": results,
        "overall_strict_pass_all_drugs": overall,
    }
    out_path = SPRINT6 / "cross_disease_f3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")
    print(f"\n=== F3 Beat AML cross-disease verdict ===")
    for drug, r in results.items():
        flag = "PASS" if r.get("bonferroni_pass_at_0p005") else "ns"
        print(f"  {drug:25s} perm_p={r.get('permutation_p_one_sided_greater', 'NA'):.4g}  {flag}")
    print(f"  Overall (all drugs strict-pass): "
          f"{'STRICT PASS' if overall else 'PARTIAL/STRICT FAIL'}")


if __name__ == "__main__":
    main()
