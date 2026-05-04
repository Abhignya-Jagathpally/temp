"""S4c: Merge outcome+treatment, mediator, and confounders into one analysis-ready table.

Output: data/processed/mmrf_sprint4_analysis.tsv

Columns:
    submitter_id            (pkey)
    A: bort_1L              (binary 0/1)
    M: M_proteasome         (continuous z-score weighted)
    Y: tt2L_days, had_2L    (right-censored survival)
    C: iss_stage            (I/II/III/Unknown → ordinal 1/2/3/0)
       age_at_dx_years
       gender_male           (0/1)
       cyto_del17p           (0/1; NaN→0 with explicit flag)
       cyto_chr1q21_gain     (0/1)
       cyto_del13q           (0/1)
       cyto_t_4_14           (0/1)
       cyto_t_11_14          (0/1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
PROC = ROOT / "data" / "processed"


def main() -> None:
    print("=== S4c: building analysis-ready table ===")
    out_treat = pd.read_csv(PROC / "mmrf_outcomes_treatment.tsv", sep="\t")
    M = pd.read_csv(PROC / "mmrf_proteasome_score.tsv", sep="\t")
    cyto = pd.read_csv(RAW / "cytogenetics.tsv", sep="\t")

    df = out_treat.merge(M, on="submitter_id", how="inner")
    df = df.merge(
        cyto[["submitter_id", "del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"]],
        on="submitter_id", how="left",
    )

    # Confounder cleanup
    iss_map = {"I": 1, "II": 2, "III": 3, "Unknown": 0}
    df["iss_stage_ord"] = df["iss_stage"].map(iss_map).fillna(0).astype(int)
    df["age_at_dx_years"] = (df["age_at_diagnosis_days"].fillna(0) / 365.25).round(1)
    df["gender_male"] = (df["gender"] == "male").astype(int)
    for col in ("del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"):
        df[f"cyto_{col}"] = df[col].fillna(0).astype(int)

    keep = [
        "submitter_id", "bort_1L", "M_proteasome", "tt2L_days", "had_2L",
        "iss_stage_ord", "age_at_dx_years", "gender_male",
        "cyto_del17p", "cyto_chr1q21_gain", "cyto_del13q",
        "cyto_t_4_14", "cyto_t_11_14",
    ]
    out = df[keep].copy()
    print(f"analysis cohort N: {len(out)}")
    print(f"  bort_1L=1 : {(out['bort_1L']==1).sum()}")
    print(f"  bort_1L=0 : {(out['bort_1L']==0).sum()}")
    print(f"  had_2L=1  : {(out['had_2L']==1).sum()}")
    print(f"  M sd      : {out['M_proteasome'].std():.3f}")
    out_path = PROC / "mmrf_sprint4_analysis.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
