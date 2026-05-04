"""S3-fix diagnostic: investigate the F3, F6, F7 failures and validate fixes.

Diagnostics performed:

D1. F3 — does INCLUDING the seeds in top-K change the verdict?
    Hypothesis: seeds are the most essential genes in the propagation; my
    pilot probably included them; my Sprint 3 excluded them.

D2. F3 — does an MM-only Chronos subset (19 lines) sharpen MoA-specific
    essentiality vs the pan-haem 64-line panel?

D3. F3 — sweep top-K ∈ {25, 50, 75, 100}; the optimum may not be 50.

D4. F6 — count consensus drivers reachable via the STRING alias mapper
    (FAM46C↔TENT5C, MMSET↔NSD2 etc.); how many additional are recoverable?

D5. F6 — try the partial-seed (leave-N-out) test design as an alternative
    to per-driver → other-driver recovery.

Output: paper/v8_artifacts/v10_sprint3/s3fix_diagnostic.json
"""

from __future__ import annotations

import gzip
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
RAW_STRING = ROOT / "data" / "raw" / "string"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize, npi_zscore, top_k_by_zscore,
)


DRUG_SEEDS = {
    "Bortezomib":   ["PSMB5", "PSMB1", "PSMB2"],
    "Carfilzomib":  ["PSMB5"],
    "Ixazomib":     ["PSMB5"],
    "Panobinostat": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "Vorinostat":   ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "Lenalidomide": ["CRBN"],
    "Pomalidomide": ["CRBN"],
    "Daratumumab":  ["CD38"],
    "Venetoclax":   ["BCL2"],
    "Selinexor":    ["XPO1"],
}


def load_alias_map() -> dict[str, str]:
    """Build alias → preferred_name mapping from STRING v12 aliases file.

    The aliases file maps STRING ENSP IDs to many alternative names. We invert
    it to a HGNC alias → STRING-preferred-name lookup that round-trips through
    protein.info.preferred_name.
    """
    info_path = RAW_STRING / "9606.protein.info.v12.0.txt.gz"
    aliases_path = RAW_STRING / "9606.protein.aliases.v12.0.txt.gz"
    print(f"[diag] loading {info_path}")
    with gzip.open(info_path, "rt") as fh:
        info = pd.read_csv(fh, sep="\t")
    pid_col = [c for c in info.columns if "protein_id" in c][0]
    pid2pref = dict(zip(info[pid_col].astype(str), info["preferred_name"].astype(str)))

    print(f"[diag] loading {aliases_path}")
    alias_to_pref: dict[str, str] = {}
    with gzip.open(aliases_path, "rt") as fh:
        header = fh.readline().split("\t")
        # Headers: #string_protein_id, alias, source
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, alias, _src = parts[:3]
            pref = pid2pref.get(pid)
            if pref is None:
                continue
            alias_to_pref.setdefault(alias.strip(), pref)
    print(f"[diag] alias map: {len(alias_to_pref):,} aliases → preferred_name")
    return alias_to_pref


def crispr_test(driver_genes: list[str], chronos: pd.DataFrame, n_perm: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    avail = set(chronos.columns)
    drv_present = [d for d in driver_genes if d in avail]
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
    perm_means = torch.empty(n_perm, dtype=bg_t.dtype)
    for i in range(n_perm):
        perm = torch.randperm(len(bg), generator=g)[:len(drv)]
        perm_means[i] = bg_t[perm].mean()
    perm_p = float((perm_means.numpy() <= drv_mean).mean())
    return {
        "n_present": int(len(drv)),
        "drv_mean_chronos": drv_mean,
        "bg_mean_chronos": float(bg.mean()),
        "mw_p_one_sided_less": float(mw_p),
        "permutation_p": perm_p,
    }


def diagnose_f3() -> dict:
    """D1, D2, D3: include-seeds × MM-vs-haem × top-K sweep."""
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)

    chronos = pd.read_parquet(PROC / "chronos_haem.parquet")
    info = pd.read_csv(PROC / "chronos_haem_lineage.tsv", sep="\t")
    for c in ("ModelID", "DepMap_ID", "depmap_id"):
        if c in info.columns:
            id_col = c; break
    mm_ids = set(info.loc[info["lineage_subtype"] == "multiple_myeloma", id_col].astype(str))
    chronos_mm = chronos.loc[chronos.index.astype(str).isin(mm_ids)].copy()
    print(f"[diag] chronos_full: {chronos.shape}, chronos_mm: {chronos_mm.shape}")

    out = {"per_drug": {}}
    bonferroni_alpha = 0.05 / len(DRUG_SEEDS)
    for include_seeds in [True, False]:
        for panel_name, panel_df in [("haem64", chronos), ("mm19", chronos_mm)]:
            for top_k in [25, 50, 75, 100]:
                config = f"seeds={'in' if include_seeds else 'out'}_panel={panel_name}_K={top_k}"
                if config not in out:
                    out[config] = {"per_drug": {}}
                n_pass = 0
                for drug, seeds in DRUG_SEEDS.items():
                    seed_idx = [gene_to_idx[g] for g in seeds if g in gene_to_idx]
                    if not seed_idx:
                        continue
                    seed_idx = np.array(seed_idx, dtype=np.int64)
                    r_obs, mn, sn = npi_zscore(
                        P, deg, seed_idx,
                        n_perm=200,  # smaller perm count for sweep speed
                        alpha=0.7,
                        rng_seed=hash(drug) & 0xFFFF,
                    )
                    excl = set() if include_seeds else set(seeds)
                    top = top_k_by_zscore(r_obs, mn, sn, gene_names, k=top_k, exclude=excl)
                    if include_seeds:
                        # prepend the seeds at the top (they have z=∞ since they're the seeds)
                        top_genes = list(seeds) + [g for g, _, _ in top if g not in set(seeds)]
                        top_genes = top_genes[:top_k]
                    else:
                        top_genes = [g for g, _, _ in top]
                    res = crispr_test(top_genes, panel_df, n_perm=500, seed=hash(drug) & 0xFFFF)
                    out[config]["per_drug"][drug] = res
                    if res.get("permutation_p", 1.0) < bonferroni_alpha:
                        n_pass += 1
                out[config]["n_bonferroni_pass"] = n_pass
                out[config]["bonferroni_alpha"] = bonferroni_alpha
                print(f"[diag-F3] {config}: {n_pass}/{len(DRUG_SEEDS)} Bonferroni-pass")
    return out


def diagnose_f6_aliases(alias_to_pref: dict[str, str]) -> dict:
    """D4: how many of the 5 missing consensus drivers can we recover via aliases?"""
    f6 = json.loads((SPRINT3 / "f6_driver_recall.json").read_text())
    missing = f6["missing_from_ppi"]
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    in_ppi = set(idx_df["gene"].astype(str))
    rescued = {}
    for g in missing:
        # Try direct alias, with versioned variants too
        for cand in [g, g.replace("HIST1H1E", "H1-4"), g.upper(), g.lower()]:
            mapped = alias_to_pref.get(cand)
            if mapped and mapped in in_ppi:
                rescued[g] = mapped
                break
    return {"missing": missing, "rescued_via_alias": rescued}


def diagnose_f6_partial_seed() -> dict:
    """D5: partial-seed leave-N-out F6 design.

    For each fold:
      - randomly select K consensus drivers as seed
      - RWR; top-100 (excluding seeds)
      - count how many of the OTHER (|consensus|-K) drivers appear in top-100
      - permutation null: random K seeds drawn from STRING (matched to consensus degree)
    """
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    P = column_normalize(A)

    f6 = json.loads((SPRINT3 / "f6_driver_recall.json").read_text())
    consensus_in_ppi = f6["consensus_in_ppi"]
    consensus_set = set(consensus_in_ppi)
    n_total = len(consensus_in_ppi)
    print(f"[diag-F6] partial-seed test on {n_total} consensus drivers")

    g_rng = torch.Generator().manual_seed(0)
    sweep = {}
    for K in [3, 5, 10, 15]:
        n_folds = 30
        recall_vals = []
        null_vals = []
        for fold in range(n_folds):
            order = torch.randperm(n_total, generator=g_rng).tolist()
            seed_genes = [consensus_in_ppi[i] for i in order[:K]]
            target_genes = set(consensus_in_ppi) - set(seed_genes)
            seed_idx = np.array([gene_to_idx[s] for s in seed_genes], dtype=np.int64)
            from resistancemap.landscape.rwr_propagation import rwr_power
            s_vec = np.zeros(P.shape[0], dtype=np.float32)
            s_vec[seed_idx] = 1.0; s_vec /= s_vec.sum()
            r, _ = rwr_power(P, s_vec, alpha=0.7)
            order_genes = np.argsort(-r)
            top = []
            for ii in order_genes:
                gname = gene_names[ii]
                if gname in set(seed_genes):
                    continue
                top.append(gname)
                if len(top) >= 100:
                    break
            recall_vals.append(sum(1 for g in top if g in target_genes))

            # null: random K seeds, count consensus in top-100
            random_seeds = torch.randperm(P.shape[0], generator=g_rng)[:K].numpy()
            sn = np.zeros(P.shape[0], dtype=np.float32)
            sn[random_seeds] = 1.0; sn /= sn.sum()
            rn, _ = rwr_power(P, sn, alpha=0.7)
            order_n = np.argsort(-rn)
            top_n = []
            for ii in order_n:
                gname = gene_names[ii]
                if ii in random_seeds:
                    continue
                top_n.append(gname)
                if len(top_n) >= 100:
                    break
            null_vals.append(sum(1 for g in top_n if g in consensus_set))

        rv = np.array(recall_vals); nv = np.array(null_vals)
        # one-sided test
        p = float((nv >= rv.mean()).mean())
        print(f"[diag-F6] K={K:2d}: mean recall={rv.mean():.2f}/{n_total-K}  "
              f"null_mean={nv.mean():.2f}  p={p:.4g}")
        sweep[f"K={K}"] = {
            "K_seed": K,
            "n_targets": int(n_total - K),
            "recall_mean": float(rv.mean()),
            "recall_median": float(np.median(rv)),
            "null_mean": float(nv.mean()),
            "permutation_p_via_resampled_null": p,
            "n_folds": n_folds,
        }
    return sweep


def main() -> None:
    out = {}
    print("=== D1+D2+D3: F3 sweep ===")
    out["f3_sweep"] = diagnose_f3()
    print("\n=== D4: F6 alias rescue ===")
    alias_map = load_alias_map()
    out["f6_alias_rescue"] = diagnose_f6_aliases(alias_map)
    print(f"[diag-F6] alias-rescued: {out['f6_alias_rescue']['rescued_via_alias']}")
    print("\n=== D5: F6 partial-seed sweep ===")
    out["f6_partial_seed"] = diagnose_f6_partial_seed()

    out_path = SPRINT3 / "s3fix_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
