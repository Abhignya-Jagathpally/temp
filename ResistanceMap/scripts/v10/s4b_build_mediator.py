"""S4b: Build per-patient proteasome mediator score from baseline expression.

Mediator definition (per-patient):
    M_i = z-score-weighted mean of log1p TPM expression over the Bortezomib-RWR
          top-50 genes, where weights are NPI z-scores from Sprint 3.

Steps:
    1. Load Bortezomib RWR top-50 (paper/v8_artifacts/v10_sprint3/bortezomib_top50_for_mediator.json)
    2. Map symbols → Ensembl IDs (data/processed/mmrf_ensembl_to_symbol.tsv)
    3. Load MMRF baseline log1p-TPM matrix (data/processed/mmrf_baseline_expression.parquet)
    4. Z-score each gene column (per-patient comparable)
    5. Per patient: M = sum_g z_score_g * std_expression_ig / sum |z_score_g|
    6. Save data/processed/mmrf_proteasome_score.tsv (submitter_id, M_proteasome)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT3 = ROOT / "paper" / "v8_artifacts" / "v10_sprint3"


def main() -> None:
    print("=== S4b: building proteasome mediator score ===")

    top50 = json.loads((SPRINT3 / "bortezomib_top50_for_mediator.json").read_text())
    print(f"top-50 RWR size: {len(top50)}")

    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))

    # Map top-50 symbols to Ensembl
    rows = []
    for entry in top50:
        sym = entry["gene"]
        ens = sym_to_ens.get(sym)
        rows.append({"symbol": sym, "ensembl": ens, "z": entry["z"]})
    rwr_df = pd.DataFrame(rows)
    n_mapped = rwr_df["ensembl"].notna().sum()
    print(f"top-50 symbols mapped to MMRF Ensembl: {n_mapped}/{len(top50)}")

    # Load MMRF expression
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    print(f"MMRF expression shape: {expr.shape}  index name: {expr.index.name}")

    # Subset to RWR genes that are in expression columns
    rwr_df = rwr_df[rwr_df["ensembl"].isin(expr.columns)].copy()
    print(f"top-50 genes present in MMRF expression: {len(rwr_df)}")
    print(f"  total z-weight present: {rwr_df['z'].abs().sum():.1f}")

    sub = expr[rwr_df["ensembl"].tolist()].copy()
    # Z-score each column (gene)
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    # Weighted sum across genes per patient
    weights = rwr_df["z"].values
    weights_norm = weights / np.abs(weights).sum()
    M = sub.values @ weights_norm
    out = pd.DataFrame({"submitter_id": expr.index, "M_proteasome": M})
    print(f"per-patient M distribution: mean={M.mean():.3f} sd={M.std():.3f} "
          f"min={M.min():.3f} max={M.max():.3f}")
    out_path = PROC / "mmrf_proteasome_score.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"saved → {out_path}")

    # Also save the gene-list used for s4e negative-control reproducibility
    rwr_df.to_csv(PROC / "mmrf_proteasome_score_genes.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
