"""S6a: Extract Beat AML 1.0 (Tyner 2018) supplement into fast-load formats.

Source: Tyner et al. 2018 Nature, PMID 30333627.
Supplement file: 41586_2018_623_MOESM3_ESM.xlsx (290 MB), 24 sheets.

Pulled tables:
    Table S5 (Clinical Summary)  → data/processed/beataml_clinical.tsv  (672 × 159)
    Table S8 (Gene Counts RPKM)  → data/processed/beataml_rpkm.parquet (22,843 × 451)
    Table S10 (Drug Responses)   → data/processed/beataml_drug_response.tsv (47,650 × 4)

Sprint 6 deliverable: Beat AML is the second hematologic disease for the v10
"hematologic foundation model" claim — without it, the title says hematologic
but the data is MM-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "beataml"
PROC = ROOT / "data" / "processed"


def main() -> None:
    src = RAW / "tyner2018_supplement.xlsx"
    print(f"=== S6a: extracting {src.name} ===")
    print("loading workbook (~ 290 MB)...")
    xl = pd.ExcelFile(src)

    # Clinical (S5)
    print("  Table S5 — Clinical Summary...")
    clin = xl.parse("Tabe S5-Clinical Summary")
    out_clin = PROC / "beataml_clinical.tsv"
    clin.to_csv(out_clin, sep="\t", index=False)
    print(f"     → {out_clin}  shape={clin.shape}")

    # RPKM gene counts (S8)
    print("  Table S8 — Gene Counts RPKM...")
    rpkm = xl.parse("Table S8-Gene Counts RPKM")
    # Pivot: rows = genes (Gene, Symbol), cols = patient labIds
    out_rpkm = PROC / "beataml_rpkm.parquet"
    rpkm.to_parquet(out_rpkm, index=False)
    print(f"     → {out_rpkm}  shape={rpkm.shape}")

    # Drug response (S10)
    print("  Table S10 — Drug Responses...")
    drugs = xl.parse("Table S10-Drug Responses")
    out_drugs = PROC / "beataml_drug_response.tsv"
    drugs.to_csv(out_drugs, sep="\t", index=False)
    print(f"     → {out_drugs}  shape={drugs.shape}")

    # Cohort overview
    print("\n=== Cohort summary ===")
    print(f"  patient labIds in clinical: {clin['LabId'].nunique() if 'LabId' in clin.columns else clin.shape[0]}")
    print(f"  patients with RNA-seq: {len(rpkm.columns) - 2}  (cols minus Gene + Symbol)")
    print(f"  unique drugs measured: {drugs['inhibitor'].nunique()}")
    drug_counts = drugs["inhibitor"].value_counts()
    print(f"  drugs of interest:")
    for d in ("Bortezomib (Velcade)", "Panobinostat", "Lenalidomide",
              "Cytarabine", "Idarubicin", "Quizartinib"):
        if d in drug_counts.index:
            print(f"    {d:30s}  N={drug_counts[d]}")


if __name__ == "__main__":
    main()
