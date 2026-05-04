"""S1c: Encode MMRF baseline matrix to 64-d latents (Sprint-1 substitute).

Spec target (V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.1): frozen scGPT cell-state
encoder, z ∈ ℝ^64. Sprint-1 substitute: log1p-TPM → standardize → 64-d PCA.
This is a documented placeholder. The downstream U_θ math (DSM, PPI Tikhonov,
Helmholtz-Hodge) does not depend on the encoder choice — only on the latent
dimension. Sprint-2 swaps in scGPT once `tdc/scGPT@acf749f3` is environment-
pinned and a checkpoint is cached.

Inputs:
    data/processed/mmrf_baseline_expression.parquet   # patients × Ensembl genes (TPM)
    data/processed/ppi_gene_index.tsv                 # PPI nodes (HGNC symbols)

Outputs:
    data/processed/mmrf_z64.npy                       # (n_patients, 64) float32
    data/processed/mmrf_z64_sample_ids.json           # ordered patient ids
    data/processed/mmrf_z64_pca.npz                   # PCA components for projection
    data/processed/mmrf_gene_features.tsv             # gene symbols actually used

Note: Ensembl IDs in MMRF expression are version-stripped already (per
`scripts/preprocess_mmrf_gdc.py`). To intersect with the PPI gene index
(HGNC symbols), we use the GENCODE v36 mapping — but since the input
TSV has gene_id only, we use TPM-bytes here and defer symbol mapping to S1d's
PPI projection step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

LATENT_DIM = 64


def main() -> None:
    expr_path = PROC / "mmrf_baseline_expression.parquet"
    print(f"[S1c] reading {expr_path}")
    expr = pd.read_parquet(expr_path)  # rows=patients, cols=Ensembl genes (TPM)
    print(f"[S1c] expression: {expr.shape[0]} patients × {expr.shape[1]} Ensembl genes")

    # Filter to high-expression genes (top-5000 by mean log1p-TPM) to reduce noise
    log_expr = np.log1p(expr.values.astype(np.float32))
    mean_expr = log_expr.mean(axis=0)
    top_idx = np.argsort(-mean_expr)[:5000]
    log_expr_top = log_expr[:, top_idx]
    print(f"[S1c] top-5000 high-expression genes selected (mean log1p-TPM range "
          f"{mean_expr[top_idx[-1]]:.2f}..{mean_expr[top_idx[0]]:.2f})")

    # Standardize per-gene
    mu = log_expr_top.mean(axis=0)
    sd = log_expr_top.std(axis=0) + 1e-6
    X = (log_expr_top - mu) / sd

    # 64-d PCA
    pca = PCA(n_components=LATENT_DIM, random_state=0)
    Z = pca.fit_transform(X).astype(np.float32)
    print(f"[S1c] PCA explained variance ratio (top 10): "
          f"{np.round(pca.explained_variance_ratio_[:10], 3).tolist()}")
    print(f"[S1c] cumulative explained variance @ 64 components: "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    # Save
    z_path = PROC / "mmrf_z64.npy"
    np.save(z_path, Z)
    print(f"[S1c] wrote {z_path} (shape={Z.shape}, dtype={Z.dtype})")

    ids_path = PROC / "mmrf_z64_sample_ids.json"
    ids_path.write_text(json.dumps(expr.index.tolist()))
    print(f"[S1c] wrote {ids_path}")

    pca_path = PROC / "mmrf_z64_pca.npz"
    np.savez(
        pca_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        gene_index=top_idx.astype(np.int32),
        feature_mu=mu.astype(np.float32),
        feature_sd=sd.astype(np.float32),
    )
    print(f"[S1c] wrote {pca_path}")

    feat_path = PROC / "mmrf_gene_features.tsv"
    pd.DataFrame({
        "ensembl_id": expr.columns[top_idx],
        "rank_by_mean_log1p_tpm": np.arange(len(top_idx)),
        "mean_log1p_tpm": mean_expr[top_idx],
    }).to_csv(feat_path, sep="\t", index=False)
    print(f"[S1c] wrote {feat_path}")

    summary = {
        "encoder": "PCA-64 on log1p-TPM (Sprint-1 substitute for frozen scGPT)",
        "n_patients": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "n_genes_input": int(expr.shape[1]),
        "n_genes_used": int(len(top_idx)),
        "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "scgpt_substitution_note": (
            "scGPT swap-in is Sprint-2; downstream U_theta math is encoder-agnostic"
            " up to latent dimension; only encoder identity changes."
        ),
    }
    summary_path = ROOT / "paper" / "v8_artifacts" / "v10_sprint1" / "s1c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S1c] wrote {summary_path}")


if __name__ == "__main__":
    main()
