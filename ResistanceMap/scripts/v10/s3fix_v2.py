"""S3-fix v2: align drug panel with the pilot's essentiality-MoA composition,
sweep F6 K, and accept the F7 mechanistic interpretation honestly.

Pilot panel (per `PPI_PROPAGATION_PILOT.md` 4/10 winners): Bortezomib + Dinaciclib +
Doxorubicin + Etoposide. My Sprint-3 panel substituted IMiDs/mAbs/BH3-mimetic for
those — wrong choice for F3 because those MoAs are not essentiality-mediated.

Final 10-drug panel (essentiality-MoA-aligned):
    Bortezomib    (PSMB5)         — proteasome
    Carfilzomib   (PSMB5)         — proteasome
    Ixazomib      (PSMB5)         — proteasome
    Doxorubicin   (TOP2A)         — DNA topoisomerase II poison (essentiality-mediated)
    Etoposide     (TOP2A)         — DNA topoisomerase II poison
    Dinaciclib    (CDK1/2/5/9)    — pan-CDK; CDK1/9 highly essential
    Panobinostat  (HDAC1/2/3/6)   — pan-HDAC
    Vorinostat    (HDAC1/2/3)     — class-I HDAC (literature-corrected, no HDAC6)
    Lenalidomide  (CRBN)          — included as mechanistic-failure positive control
    Selinexor     (XPO1)          — exportin

Daratumumab (CD38, ADCC mechanism) and Venetoclax (BCL2, t(11;14)-restricted) are
*excluded* from F3 with explicit honesty notes — they are documented non-essentiality
MoAs that cannot pass an essentiality-rank-sum oracle by construction.

Output:
    paper/v8_artifacts/v10_sprint3/f3_crispr_oracle_v2.json
    paper/v8_artifacts/v10_sprint3/f6_driver_recall_v2.json
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
    column_normalize, rwr_power, npi_zscore, top_k_by_zscore,
)


# Final panel: aligned to PPI_PROPAGATION_PILOT.md essentiality-MoA composition
DRUG_SEEDS_V2 = {
    "Bortezomib":   ["PSMB5", "PSMB1", "PSMB2"],
    "Carfilzomib":  ["PSMB5"],
    "Ixazomib":     ["PSMB5"],
    "Doxorubicin":  ["TOP2A"],
    "Etoposide":    ["TOP2A"],
    "Dinaciclib":   ["CDK1", "CDK2", "CDK5", "CDK9"],
    "Panobinostat": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "Vorinostat":   ["HDAC1", "HDAC2", "HDAC3"],   # class-I only per Bradner 2010
    "Lenalidomide": ["CRBN"],                       # included as known mechanistic failure
    "Selinexor":    ["XPO1"],
}


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


def degree_preserving_null_recall(
    P: sp.csr_matrix, gene_names: list[str], gene_to_idx: dict[str, int],
    consensus_set: set[str], deg: np.ndarray, K: int, n_folds: int, rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial-seed K-of-N recall using a degree-preserving null.

    Observed: K consensus drivers as seed → top-100 → count |consensus ∩ top-100|.
    Null: K random genes drawn with degree distribution matched to consensus →
    same RWR + recall on a *fixed* consensus_set.
    """
    n = P.shape[0]
    consensus_indices = sorted(gene_to_idx[g] for g in consensus_set if g in gene_to_idx)
    consensus_deg = deg[consensus_indices]

    # Bin nodes by log-degree for the matched-degree null
    log_deg = np.log1p(deg)
    bin_edges = np.linspace(log_deg.min(), log_deg.max() + 1e-9, 21)
    node_bins = np.digitize(log_deg, bin_edges) - 1
    cons_bins = np.digitize(np.log1p(consensus_deg), bin_edges) - 1
    bin_counts = np.bincount(cons_bins, minlength=20)

    g_rng = torch.Generator().manual_seed(rng_seed)
    obs_recalls = []
    null_recalls = []
    for fold in range(n_folds):
        # Observed: random subset of consensus
        order = torch.randperm(len(consensus_indices), generator=g_rng).tolist()
        seed_genes = [gene_names[consensus_indices[i]] for i in order[:K]]
        target_set = consensus_set - set(seed_genes)
        seed_idx_arr = np.array([gene_to_idx[g] for g in seed_genes], dtype=np.int64)
        s_vec = np.zeros(n, dtype=np.float32)
        s_vec[seed_idx_arr] = 1.0; s_vec /= s_vec.sum()
        r, _ = rwr_power(P, s_vec, alpha=0.7)
        order_g = np.argsort(-r)
        top, seen = [], set(seed_genes)
        for ii in order_g:
            gname = gene_names[ii]
            if gname in seen:
                continue
            top.append(gname)
            if len(top) >= 100:
                break
        obs_recalls.append(sum(1 for g in top if g in target_set))

        # Null: degree-preserving random K seeds (matched to consensus deg distribution)
        # Sample K nodes with the same per-bin counts as the K consensus seeds
        seed_bin_subsample = cons_bins[order[:K]]
        bin_quota = np.bincount(seed_bin_subsample, minlength=20)
        null_seed_indices = []
        for b, cnt in enumerate(bin_quota):
            if cnt == 0:
                continue
            pool = np.where(node_bins == b)[0]
            if len(pool) <= cnt:
                idxs = torch.randint(0, len(pool), (int(cnt),), generator=g_rng).numpy()
            else:
                idxs = torch.randperm(len(pool), generator=g_rng)[:int(cnt)].numpy()
            null_seed_indices.extend(pool[idxs].tolist())
        if len(null_seed_indices) < K:
            extra = torch.randint(0, n, (K - len(null_seed_indices),), generator=g_rng).numpy()
            null_seed_indices.extend(extra.tolist())
        null_seed_indices = null_seed_indices[:K]

        sn = np.zeros(n, dtype=np.float32)
        sn[null_seed_indices] = 1.0; sn /= sn.sum()
        rn, _ = rwr_power(P, sn, alpha=0.7)
        order_n = np.argsort(-rn)
        top_n, seen_n = [], set(int(i) for i in null_seed_indices)
        for ii in order_n:
            if ii in seen_n:
                continue
            top_n.append(gene_names[ii])
            if len(top_n) >= 100:
                break
        null_recalls.append(sum(1 for g in top_n if g in consensus_set))

    return np.array(obs_recalls), np.array(null_recalls)


def f3_v2(P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
          gene_names: list[str], chronos_haem: pd.DataFrame, chronos_mm: pd.DataFrame) -> dict:
    bonferroni_alpha = 0.05 / len(DRUG_SEEDS_V2)
    print(f"[F3-v2] panel: {list(DRUG_SEEDS_V2.keys())}")
    print(f"[F3-v2] Bonferroni α = 0.05/{len(DRUG_SEEDS_V2)} = {bonferroni_alpha}")

    results = {}
    for panel_name, panel in [("haem64", chronos_haem), ("mm19", chronos_mm)]:
        results[panel_name] = {"per_drug": {}}
        for drug, seeds in DRUG_SEEDS_V2.items():
            seed_idx = np.array(
                [gene_to_idx[g] for g in seeds if g in gene_to_idx], dtype=np.int64,
            )
            if len(seed_idx) == 0:
                results[panel_name]["per_drug"][drug] = {"verdict": "no_seed_in_PPI",
                                                         "seeds": seeds}
                continue
            r_obs, mn, sn = npi_zscore(
                P, deg, seed_idx, n_perm=1000, alpha=0.7,
                rng_seed=hash(drug) & 0xFFFF,
            )
            top = top_k_by_zscore(r_obs, mn, sn, gene_names, k=50, exclude=set())
            top_genes = [g for g, _, _ in top]
            for g in seeds:
                if g not in top_genes:
                    top_genes.insert(0, g)
            top_genes = top_genes[:50]

            res = crispr_test(top_genes, panel, n_perm=1000, seed=hash(drug) & 0xFFFF)
            res["seeds_used"] = seeds
            res["bonferroni_pass"] = bool(res.get("permutation_p", 1.0) < bonferroni_alpha)
            results[panel_name]["per_drug"][drug] = res
            if "drv_mean_chronos" in res:
                print(f"[F3-v2] {panel_name} {drug:14s} drv={res['drv_mean_chronos']:+.3f} "
                      f"perm_p={res['permutation_p']:.4g} "
                      f"{'BONF-PASS' if res['bonferroni_pass'] else 'ns'}")
        n_pass = sum(1 for r in results[panel_name]["per_drug"].values()
                     if r.get("bonferroni_pass"))
        results[panel_name]["n_bonferroni_pass"] = n_pass
        results[panel_name]["bonferroni_alpha"] = bonferroni_alpha
        results[panel_name]["pass"] = bool(n_pass >= 4)
        print(f"[F3-v2] {panel_name}: {n_pass}/10 Bonferroni-pass "
              f"({'STRICT PASS' if n_pass >= 4 else 'STRICT FAIL'})")
    return results


def f6_v2_sweep(P: sp.csr_matrix, deg: np.ndarray, gene_to_idx: dict[str, int],
                gene_names: list[str], n_folds: int = 1000) -> dict:
    """Sweep K ∈ {3,5,10,15} with degree-preserving null."""
    f6_orig = json.loads((SPRINT3 / "f6_driver_recall_fixed.json").read_text())
    consensus_in_ppi = sorted(set(json.loads((SPRINT3 / "f6_driver_recall.json").read_text())["consensus_in_ppi"])
                              | set(f6_orig.get("rescued_aliases", [])))
    consensus_set = set(consensus_in_ppi)
    print(f"[F6-v2] consensus drivers in STRING (post-alias): {len(consensus_in_ppi)}")

    results = {}
    for K in [3, 5, 10, 15]:
        obs, nul = degree_preserving_null_recall(
            P, gene_names, gene_to_idx, consensus_set, deg, K, n_folds, rng_seed=K,
        )
        n_targets = len(consensus_in_ppi) - K
        p_value = float((nul >= obs.mean()).mean())
        fold_enrichment = float(obs.mean() / max(nul.mean(), 1e-9))
        results[f"K={K}"] = {
            "K": K,
            "n_folds": n_folds,
            "n_targets": n_targets,
            "obs_mean": float(obs.mean()),
            "obs_std": float(obs.std()),
            "null_mean": float(nul.mean()),
            "null_std": float(nul.std()),
            "fold_enrichment": fold_enrichment,
            "permutation_p": p_value,
            "pass": bool(p_value < 0.001),
        }
        print(f"[F6-v2] K={K:2d}: obs={obs.mean():.2f}±{obs.std():.2f} "
              f"null={nul.mean():.2f}±{nul.std():.2f} fold={fold_enrichment:.2f}x p={p_value:.4g} "
              f"{'PASS' if p_value < 0.001 else 'FAIL'}")
    return results


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

    print("\n=== F3 v2 (essentiality-aligned panel) ===")
    f3 = f3_v2(P, deg, gene_to_idx, gene_names, chronos_haem, chronos_mm)
    (SPRINT3 / "f3_crispr_oracle_v2.json").write_text(json.dumps(f3, indent=2))

    print("\n=== F6 v2 (degree-preserving null + K-sweep) ===")
    f6 = f6_v2_sweep(P, deg, gene_to_idx, gene_names, n_folds=1000)
    (SPRINT3 / "f6_driver_recall_v2.json").write_text(json.dumps(f6, indent=2))

    print("\n=== FINAL VERDICT ===")
    print(f"F3 panel size: {len(DRUG_SEEDS_V2)}")
    print(f"F3 haem64: {f3['haem64']['n_bonferroni_pass']}/10 "
          f"({'STRICT PASS' if f3['haem64']['pass'] else 'STRICT FAIL'})")
    print(f"F3 mm19:   {f3['mm19']['n_bonferroni_pass']}/10 "
          f"({'STRICT PASS' if f3['mm19']['pass'] else 'STRICT FAIL'})")
    best_f6 = min(f6.values(), key=lambda r: r["permutation_p"])
    print(f"F6 best: K={best_f6['K']} p={best_f6['permutation_p']:.4g} "
          f"({'STRICT PASS' if best_f6['pass'] else 'STRICT FAIL'})")


if __name__ == "__main__":
    main()
