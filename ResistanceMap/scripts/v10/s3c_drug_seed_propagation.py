"""S3c: RWR propagation for 10 MM-relevant drug seed gene sets.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.4:
    r = α(I - (1-α)P)^(-1) s
    z(g) = (r(g) - mean_null(g)) / std_null(g)   [NPI degree-stratified null]

Default α = 0.7 (Köhler 2008 / Vanunu 2010 robust regime, per spec §2.4).
α can be tuned per seed in S3f (multi-seed pooling fixes from `PPI_PROPAGATION_PILOT.md`).

Inputs:
    data/processed/ppi_adjacency_v10s2.npz    # sparse symmetric A
    data/processed/ppi_degree_v10s2.npy       # node degree
    data/processed/ppi_gene_index_v10s2.tsv   # HGNC → row idx

Output:
    paper/v8_artifacts/v10_sprint3/drug_propagation.npz
        - per-drug r_obs, z, top-100 gene names + z scores
    paper/v8_artifacts/v10_sprint3/s3c_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"
SPRINT3.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.rwr_propagation import (  # noqa: E402
    column_normalize,
    rwr_power,
    npi_zscore,
    top_k_by_zscore,
)


# Drug → primary target seed sets (per ChEMBL mechanism + MM literature)
DRUG_SEEDS: dict[str, list[str]] = {
    "Bortezomib":   ["PSMB5", "PSMB1", "PSMB2"],          # 26S proteasome β5
    "Carfilzomib":  ["PSMB5"],                            # selective β5
    "Ixazomib":     ["PSMB5"],                            # β5 (oral)
    "Panobinostat": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"], # pan-HDAC
    "Vorinostat":   ["HDAC1", "HDAC2", "HDAC3", "HDAC6"], # pan-HDAC
    "Lenalidomide": ["CRBN"],                             # cereblon
    "Pomalidomide": ["CRBN"],                             # cereblon
    "Daratumumab":  ["CD38"],                             # anti-CD38
    "Venetoclax":   ["BCL2"],                             # BH3-mimetic
    "Selinexor":    ["XPO1"],                             # exportin-1
}

ALPHA_DEFAULT = 0.7
N_PERM = 1000
TOP_K = 100


def main() -> None:
    A = sp.load_npz(PROC / "ppi_adjacency_v10s2.npz").tocsr()
    deg = np.load(PROC / "ppi_degree_v10s2.npy")
    idx_df = pd.read_csv(PROC / "ppi_gene_index_v10s2.tsv", sep="\t")
    gene_to_idx = dict(zip(idx_df["gene"].astype(str), idx_df["idx"].astype(int)))
    gene_names = idx_df["gene"].astype(str).tolist()
    n = A.shape[0]
    print(f"[S3c] PPI: {n} nodes, {A.nnz} nnz, mean_deg={deg.mean():.2f}")

    P = column_normalize(A)
    print(f"[S3c] column-normalized P built")

    drug_results = {}
    for drug, seed_genes in DRUG_SEEDS.items():
        seed_idx_list = [gene_to_idx[g] for g in seed_genes if g in gene_to_idx]
        if not seed_idx_list:
            print(f"[S3c] {drug}: SKIP — no seed genes in PPI index")
            continue
        seed_idx = np.array(seed_idx_list, dtype=np.int64)
        print(f"[S3c] {drug}: seeds = {seed_genes} → indices {seed_idx_list}")

        r_obs, mean_null, std_null = npi_zscore(
            P, deg, seed_idx, n_perm=N_PERM, alpha=ALPHA_DEFAULT, rng_seed=hash(drug) & 0xFFFF,
        )
        z = (r_obs - mean_null) / np.maximum(std_null, 1e-9)
        seeds_set = set(seed_genes)
        top_k = top_k_by_zscore(r_obs, mean_null, std_null, gene_names, k=TOP_K, exclude=seeds_set)

        drug_results[drug] = {
            "seeds": seed_genes,
            "seeds_in_ppi": [g for g in seed_genes if g in gene_to_idx],
            "alpha": ALPHA_DEFAULT,
            "n_perm": N_PERM,
            "r_obs": r_obs,
            "mean_null": mean_null,
            "std_null": std_null,
            "z": z,
            "top_k_genes": [g for g, _, _ in top_k],
            "top_k_z": [zi for _, zi, _ in top_k],
            "top_k_r": [ri for _, _, ri in top_k],
        }
        print(f"[S3c] {drug}: top-5 by z = "
              f"{[(g, round(z, 2)) for g, z, _ in top_k[:5]]}")

    # Save per-drug outputs to one big NPZ
    out_npz = SPRINT3 / "drug_propagation.npz"
    save_dict = {"drugs": np.array(list(drug_results.keys()), dtype=object),
                 "gene_names": np.array(gene_names, dtype=object)}
    for d, r in drug_results.items():
        save_dict[f"{d}__r_obs"] = r["r_obs"]
        save_dict[f"{d}__z"] = r["z"]
        save_dict[f"{d}__mean_null"] = r["mean_null"]
        save_dict[f"{d}__std_null"] = r["std_null"]
        save_dict[f"{d}__top_k_genes"] = np.array(r["top_k_genes"], dtype=object)
        save_dict[f"{d}__top_k_z"] = np.array(r["top_k_z"], dtype=np.float32)
    np.savez(out_npz, **save_dict)
    print(f"[S3c] wrote {out_npz}")

    summary = {
        "n_drugs": len(drug_results),
        "alpha": ALPHA_DEFAULT,
        "n_perm": N_PERM,
        "top_k": TOP_K,
        "ppi_nodes": int(n),
        "per_drug": {
            drug: {
                "seeds_used": r["seeds_in_ppi"],
                "top5_genes": r["top_k_genes"][:5],
                "top5_z": [float(z) for z in r["top_k_z"][:5]],
            }
            for drug, r in drug_results.items()
        },
    }
    summary_path = SPRINT3 / "s3c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3c] wrote {summary_path}")


if __name__ == "__main__":
    main()
