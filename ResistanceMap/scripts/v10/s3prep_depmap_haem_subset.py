"""S3-prep: Subset DepMap CRISPRGeneEffect.csv to haematologic cell lines.

Per PPI_PROPAGATION_PILOT.md fix #2: pilot ran with 3 MM cell lines; expand to
~50 haematologic lines for Sprint 3 falsification gates F3 (CRISPR rank-sum
oracle), F6 (Walker drivers ≥35/50 in top-100), and F7 (multi-seed pooling).

DepMap sample_info.csv tags hematological lineages with `lineage` ∈
{`leukemia`, `lymphoma`, `plasma_cell`, ...}; we keep all of them so AML/CML
support the "hematologic" claim in the v10 title.

Inputs:
    data/raw/depmap/CRISPRGeneEffect.csv
    data/raw/sample_info.csv

Output:
    data/processed/chronos_haem.parquet
    data/processed/chronos_haem_lineage.tsv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# DepMap "lineage" values that count as hematologic for v10
HAEM_LINEAGES = {
    "Lymphoid",
    "Myeloid",
    "leukemia",
    "lymphoma",
    "plasma_cell",
    "lymphocyte",
    "myeloid",
}


def main() -> None:
    chronos_path = RAW / "depmap" / "CRISPRGeneEffect.csv"
    info_path = RAW / "sample_info.csv"
    if not chronos_path.exists():
        sys.exit(f"missing {chronos_path}")
    if not info_path.exists():
        sys.exit(f"missing {info_path}")

    print(f"[S3-prep] reading {info_path}")
    info = pd.read_csv(info_path)
    print(f"[S3-prep] sample_info: {info.shape}; columns sample: {info.columns.tolist()[:8]}")

    # Find the lineage column — DepMap renames it across versions
    lineage_cols = [c for c in info.columns if "lineage" in c.lower() or c == "OncotreePrimaryDisease"]
    print(f"[S3-prep] lineage candidate cols: {lineage_cols}")

    # Use the most-permissive: any lineage column matching a haem keyword
    haem_mask = pd.Series(False, index=info.index)
    for col in lineage_cols:
        haem_mask |= info[col].astype(str).str.contains(
            "|".join(HAEM_LINEAGES) + "|leukemia|lymphoma|myeloma|plasma|lymphoid|myeloid",
            case=False,
            regex=True,
            na=False,
        )
    haem_info = info[haem_mask].copy()
    print(f"[S3-prep] haematologic cell lines flagged: {len(haem_info)}")

    # Identify the cell-line-id column DepMap uses (ModelID or DepMap_ID)
    id_col = None
    for cand in ("ModelID", "DepMap_ID", "depmap_id"):
        if cand in haem_info.columns:
            id_col = cand
            break
    if id_col is None:
        sys.exit("no DepMap cell-line ID column found in sample_info")
    haem_ids = set(haem_info[id_col].astype(str).tolist())

    print(f"[S3-prep] reading {chronos_path}")
    # The CSV is wide: rows=cell-lines, cols=GENE_SYMBOL (HUGO_(ENTREZ))
    chronos = pd.read_csv(chronos_path, index_col=0)
    print(f"[S3-prep] chronos: {chronos.shape}")
    chronos.index = chronos.index.astype(str)
    keep = [r for r in chronos.index if r in haem_ids]
    print(f"[S3-prep] cell-lines kept: {len(keep)}")
    chronos_haem = chronos.loc[keep].copy()

    # Strip "(ENTREZ)" suffix from columns to make them join-able with PPI gene index
    new_cols = chronos_haem.columns.str.replace(r"\s*\(\d+\)$", "", regex=True)
    chronos_haem.columns = new_cols

    out_path = PROC / "chronos_haem.parquet"
    chronos_haem.to_parquet(out_path)
    print(f"[S3-prep] wrote {out_path} (shape={chronos_haem.shape})")

    out_lineage = PROC / "chronos_haem_lineage.tsv"
    haem_info.to_csv(out_lineage, sep="\t", index=False)
    print(f"[S3-prep] wrote {out_lineage}")

    summary = {
        "n_cell_lines": int(len(keep)),
        "n_genes": int(chronos_haem.shape[1]),
        "id_column": id_col,
        "lineage_keywords": sorted(HAEM_LINEAGES),
    }
    summary_path = ROOT / "paper" / "v8_artifacts" / "v10_sprint1" / "s3prep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S3-prep] wrote {summary_path}")


if __name__ == "__main__":
    main()
