"""S2c: Encode paired-patient (z0, z1) latents via shared Sprint-1 PCA decoder.

Inputs:
    data/processed/mmrf_paired_patients.tsv     # patient_id, t0_aliquot, t1_aliquot
    data/raw/mmrf_commpass/gene_expression.tsv  # source TPM matrix (per-aliquot)
    data/processed/mmrf_z64_pca.npz             # frozen PCA-64 encoder (Sprint 1)
    data/processed/mmrf_gene_features.tsv       # top-5000 ENSG IDs in PCA fitting order

Outputs:
    data/processed/mmrf_paired_z64.npz          # arrays: patient_id, z0, z1, t0_tp, t1_tp, has_2nd_line
    paper/v8_artifacts/v10_sprint2/s2c_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
PROC = ROOT / "data" / "processed"
SPRINT2 = ROOT / "paper" / "v8_artifacts" / "v10_sprint2"
SPRINT2.mkdir(parents=True, exist_ok=True)


def main() -> None:
    paired = pd.read_csv(PROC / "mmrf_paired_patients.tsv", sep="\t")
    print(f"[S2c] paired patients: {len(paired)}")

    # Load full expression matrix (genes × aliquots)
    expr_path = RAW / "gene_expression.tsv"
    print(f"[S2c] reading {expr_path}")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)  # rows=Ensembl, cols=aliquot ids
    print(f"[S2c] expression: {expr.shape[0]} genes × {expr.shape[1]} aliquots")

    pca = np.load(PROC / "mmrf_z64_pca.npz")
    components = pca["components"]   # (64, 5000)
    pca_mean = pca["mean"]           # (5000,)
    feature_mu = pca["feature_mu"]   # (5000,)
    feature_sd = pca["feature_sd"]   # (5000,)
    gene_order = pca["gene_index"]   # indices into the original *full-expr* column order

    feat_df = pd.read_csv(PROC / "mmrf_gene_features.tsv", sep="\t")
    top_ensg = feat_df["ensembl_id"].tolist()
    print(f"[S2c] PCA encoder: top-{len(top_ensg)} genes, {components.shape[0]}-d latent")

    # Subset expression to top-5000 (in the same order PCA was fitted)
    expr_top = expr.loc[top_ensg].astype(np.float32)
    log_expr_top = np.log1p(expr_top.values).astype(np.float32)  # (5000, n_aliquots)

    # Aliquot-aligned encoding helper
    aliquot_to_idx = {a: i for i, a in enumerate(expr_top.columns)}

    def encode(aliquot: str) -> np.ndarray:
        idx = aliquot_to_idx.get(aliquot)
        if idx is None:
            return None
        x = log_expr_top[:, idx]
        x_std = (x - feature_mu) / feature_sd
        # X_centered = x_std - pca_mean ; z = X_centered @ components.T
        return ((x_std - pca_mean) @ components.T).astype(np.float32)

    z0_list, z1_list = [], []
    keep_idx = []
    for i, row in paired.iterrows():
        z0 = encode(row["t0_aliquot"])
        z1 = encode(row["t1_aliquot"])
        if z0 is None or z1 is None:
            continue
        z0_list.append(z0)
        z1_list.append(z1)
        keep_idx.append(i)
    Z0 = np.stack(z0_list, axis=0)
    Z1 = np.stack(z1_list, axis=0)
    paired_kept = paired.iloc[keep_idx].reset_index(drop=True)
    print(f"[S2c] encoded {len(keep_idx)} paired patients (z0={Z0.shape}, z1={Z1.shape})")

    out_npz = PROC / "mmrf_paired_z64.npz"
    np.savez(
        out_npz,
        z0=Z0,
        z1=Z1,
        patient_id=np.array(paired_kept["patient_id"].tolist(), dtype=object),
        t0_timepoint=paired_kept["t0_timepoint"].values.astype(np.int32),
        t1_timepoint=paired_kept["t1_timepoint"].values.astype(np.int32),
        has_2nd_line=paired_kept["has_2nd_line"].values.astype(np.int32),
    )
    print(f"[S2c] wrote {out_npz}")

    # Quick sanity: distribution of (t1 - t0) latent shifts
    shift = Z1 - Z0
    print(f"[S2c] shift norm — mean={float(np.linalg.norm(shift, axis=1).mean()):.3f}  "
          f"median={float(np.median(np.linalg.norm(shift, axis=1))):.3f}")

    summary = {
        "n_encoded": int(len(keep_idx)),
        "n_strict_paired": int(paired_kept["has_2nd_line"].sum()),
        "z_shift_norm_mean": float(np.linalg.norm(shift, axis=1).mean()),
        "z_shift_norm_median": float(np.median(np.linalg.norm(shift, axis=1))),
        "z_norm_t0_mean": float(np.linalg.norm(Z0, axis=1).mean()),
        "z_norm_t1_mean": float(np.linalg.norm(Z1, axis=1).mean()),
        "encoder": "Sprint-1 frozen PCA-64",
    }
    summary_path = SPRINT2 / "s2c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S2c] wrote {summary_path}")


if __name__ == "__main__":
    main()
