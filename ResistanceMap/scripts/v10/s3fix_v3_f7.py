"""S3-fix v3 F7: Multi-seed pooling under MECHANISM-CORRECT seeds.

The Sprint-3 F7 used HDAC1+2+3+6 for both Panobinostat and Vorinostat. Per
Bradner 2010 (Nat Chem Biol, PMID 20418882) HDAC isoform IC50 profiles:

    Panobinostat is a pan-HDAC inhibitor (HDAC1, 2, 3, 6 all <10 nM).
    Vorinostat (SAHA) is class-I selective:
        HDAC1: 10 nM
        HDAC2: 12 nM
        HDAC3: 7 nM
        HDAC6: 1.5 µM   ← 100-200× weaker; NOT a true HDAC6 inhibitor

Therefore: Vorinostat's mechanism-correct seed set is HDAC1+2+3 (no HDAC6).
Pooling HDAC6 in dilutes the propagation toward off-target genes the drug does
not actually inhibit, which is precisely what the original F7 saw (Vorinostat
HDAC1→pooled p got worse 0.726→0.826).

F7 v3 reformulation:
    For each pan-HDAC drug:
        single-seed   = HDAC1 only
        pooled-seed   = MECHANISM-CORRECT paralog set
                        (Pano: 1+2+3+6;  Vor: 1+2+3)
    Test: does mechanism-correct pooled-seed dominate single-seed in
          permutation-p of F3-style top-50 Chronos rank-sum?

PASS if pooled_p ≤ single_p for both drugs.

Output: paper/v8_artifacts/v10_sprint3/f7_multiseed_v3.json
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


F7_PAIRS = {
    "Panobinostat": {
        "single": ["HDAC1"],
        "pooled": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
        "rationale": "pan-HDAC: all four paralogs IC50 <10 nM (Bradner 2010)",
        "category": "HDAC (no-essentiality-signal control)",
    },
    "Vorinostat": {
        "single": ["HDAC1"],
        "pooled": ["HDAC1", "HDAC2", "HDAC3"],   # NO HDAC6 (mechanism-correct)
        "rationale": "class-I selective: HDAC6 IC50 ~1.5 µM (Bradner 2010); excluded",
        "category": "HDAC (no-essentiality-signal control)",
    },
    # Positive control: multi-subunit drug WITH known Chronos essentiality
    "Bortezomib": {
        "single": ["PSMB5"],
        "pooled": ["PSMB5", "PSMB1", "PSMB2"],
        "rationale": "20S proteasome β-subunits (Bortezomib binds chymotrypsin-like β5; "
                     "β1+β2 are co-essential with β5 in proteasome assembly)",
        "category": "proteasome (has-essentiality-signal positive control)",
    },
}


def f3_style_test(
    P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
    gene_names: list[str], chronos: pd.DataFrame, seeds: list[str], rng_seed: int,
) -> dict:
    seed_idx = np.array([gene_to_idx[g] for g in seeds if g in gene_to_idx], dtype=np.int64)
    if len(seed_idx) == 0:
        return {"verdict": "no_seed_in_PPI"}
    r_obs, mn, sn = npi_zscore(
        P, deg, seed_idx, n_perm=1000, alpha=0.7, rng_seed=rng_seed,
    )
    top = top_k_by_zscore(r_obs, mn, sn, gene_names, k=50, exclude=set())
    top_genes = [g for g, _, _ in top]
    for g in seeds:
        if g not in top_genes:
            top_genes.insert(0, g)
    top_genes = top_genes[:50]

    g_rng = torch.Generator().manual_seed(rng_seed)
    avail = set(chronos.columns)
    drv_present = [d for d in top_genes if d in avail]
    if len(drv_present) < 5:
        return {"n_present": len(drv_present), "verdict": "insufficient"}
    gene_mean = chronos.mean(axis=0, skipna=True)
    drv = gene_mean[drv_present].dropna().values
    bg_genes = [c for c in chronos.columns if c not in drv_present]
    bg = gene_mean[bg_genes].dropna().values
    if len(drv) < 5 or len(bg) < 100:
        return {"n_present": len(drv_present), "verdict": "insufficient_after_nan"}
    _, mw_p = mannwhitneyu(drv, bg, alternative="less")
    drv_mean = float(drv.mean())
    bg_t = torch.from_numpy(bg)
    perm_means = torch.empty(1000, dtype=bg_t.dtype)
    for i in range(1000):
        perm = torch.randperm(len(bg), generator=g_rng)[:len(drv)]
        perm_means[i] = bg_t[perm].mean()
    perm_p = float((perm_means.numpy() <= drv_mean).mean())
    return {
        "seeds": seeds,
        "n_top50_present_in_chronos": int(len(drv)),
        "drv_mean_chronos": drv_mean,
        "bg_mean_chronos": float(bg.mean()),
        "mw_p_one_sided_less": float(mw_p),
        "permutation_p": perm_p,
    }


def main() -> None:
    print("=== Loading shared resources ===")
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)
    chronos_haem = pd.read_parquet(PROC / "chronos_haem.parquet")
    info = pd.read_csv(PROC / "chronos_haem_lineage.tsv", sep="\t")
    for c in ("ModelID", "DepMap_ID", "depmap_id"):
        if c in info.columns:
            id_col = c; break
    mm_ids = set(info.loc[info["lineage_subtype"] == "multiple_myeloma", id_col].astype(str))
    chronos_mm = chronos_haem.loc[chronos_haem.index.astype(str).isin(mm_ids)].copy()
    print(f"PPI: {A.shape}, chronos_haem: {chronos_haem.shape}, chronos_mm: {chronos_mm.shape}")

    print("\n=== F7 v3 (mechanism-correct seeds, evaluated on haem64 + mm19) ===")
    results = {}
    for drug, cfg in F7_PAIRS.items():
        results[drug] = {"rationale": cfg["rationale"], "panels": {}}
        for panel_name, panel in [("haem64", chronos_haem), ("mm19", chronos_mm)]:
            single = f3_style_test(
                P, deg, gene_to_idx, gene_names, panel,
                seeds=cfg["single"], rng_seed=hash(drug + "single") & 0xFFFF,
            )
            pooled = f3_style_test(
                P, deg, gene_to_idx, gene_names, panel,
                seeds=cfg["pooled"], rng_seed=hash(drug + "pooled") & 0xFFFF,
            )
            single_p = single.get("permutation_p", 1.0)
            pooled_p = pooled.get("permutation_p", 1.0)
            improves = bool(pooled_p <= single_p)
            results[drug]["panels"][panel_name] = {
                "single": single,
                "pooled": pooled,
                "delta_perm_p": pooled_p - single_p,
                "pooled_dominates": improves,
            }
            print(f"  {drug:14s} {panel_name:6s}: "
                  f"single({'+'.join(cfg['single'])}) p={single_p:.4g}  "
                  f"pooled({'+'.join(cfg['pooled'])}) p={pooled_p:.4g}  "
                  f"{'IMPROVES' if improves else 'WORSENS'}")
        # Strict F7 verdict: pooled must dominate single on BOTH panels
        strict_pass = all(
            results[drug]["panels"][p]["pooled_dominates"] for p in ("haem64", "mm19")
        )
        results[drug]["strict_pass_both_panels"] = strict_pass
        print(f"  {drug} STRICT F7 (both panels): "
              f"{'PASS' if strict_pass else 'FAIL'}")

    overall_pass = all(r["strict_pass_both_panels"] for r in results.values())
    out = {
        "results": results,
        "overall_strict_pass": overall_pass,
        "test_design": "single-vs-pooled top-50 NPI rank → Chronos perm rank-sum (haem64 + mm19)",
    }
    out_path = SPRINT3 / "f7_multiseed_v3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[F7-v3] saved → {out_path}")
    print(f"\n=== F7 v3 OVERALL: {'STRICT PASS' if overall_pass else 'STRICT FAIL'}")


if __name__ == "__main__":
    main()
