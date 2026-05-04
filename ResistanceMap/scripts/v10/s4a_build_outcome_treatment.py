"""S4a: Build the analysis-ready Sprint-4 dataset.

Per spec §9 row 4 ("Single-mediator NIE: proteasome → time-to-2nd-line Bortezomib"),
we need three pieces per MMRF patient:

    A  : Bortezomib first-line indicator (1 if any 1L row mentions Bortezomib)
    Y  : (TT2L_days, had_2L)  — time-to-2nd-line therapy with right-censoring
                                 censored at days_to_last_follow_up if no 2L

This script also emits an Ensembl→HGNC mapping for the mediator step (s4b).

Outputs:
    data/processed/mmrf_outcomes_treatment.tsv
    data/processed/mmrf_ensembl_to_symbol.tsv
"""

from __future__ import annotations

import gzip
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
PROC = ROOT / "data" / "processed"


def build_outcome_treatment() -> pd.DataFrame:
    treat = pd.read_csv(RAW / "treatments.tsv", sep="\t")
    clin = pd.read_csv(RAW / "clinical.tsv", sep="\t")

    # Bortezomib first-line indicator
    bort_1L = (
        treat[
            (treat["regimen_or_line_of_therapy"] == "First line of therapy")
            & (treat["therapeutic_agents"].fillna("").str.contains("Bortezomib", case=False))
        ]["submitter_id"].drop_duplicates().tolist()
    )
    bort_set = set(bort_1L)

    # Time-to-2nd-line: earliest 2L start
    tl2 = (
        treat[treat["regimen_or_line_of_therapy"] == "Second line of therapy"]
        .groupby("submitter_id")["days_to_treatment_start"]
        .min()
        .rename("tt2L_days_raw")
        .reset_index()
    )

    df = clin[["submitter_id", "vital_status", "days_to_death",
               "days_to_last_follow_up", "iss_stage", "age_at_diagnosis_days",
               "gender", "n_treatments"]].copy()
    df["bort_1L"] = df["submitter_id"].isin(bort_set).astype(int)
    df = df.merge(tl2, on="submitter_id", how="left")

    # Build right-censored TT2L: tt2L if observed else last_follow_up
    df["had_2L"] = df["tt2L_days_raw"].notna().astype(int)
    df["tt2L_days"] = df["tt2L_days_raw"]
    # Censor at min(last_follow_up, days_to_death) for unobserved 2L
    censor_at = df["days_to_last_follow_up"]
    censor_at = censor_at.fillna(df["days_to_death"])
    df.loc[df["had_2L"] == 0, "tt2L_days"] = censor_at[df["had_2L"] == 0].values
    df = df[df["tt2L_days"].notna() & (df["tt2L_days"] > 0)].copy()
    df["tt2L_days"] = df["tt2L_days"].astype(int)
    return df


def build_ensembl_symbol_map() -> pd.DataFrame:
    """Use any one MMRF STAR gene-counts file to extract a stable Ensembl→symbol map."""
    rna_dir = RAW / "rna"
    sample_file = next(rna_dir.glob("*.rna_seq.augmented_star_gene_counts.tsv"), None)
    if sample_file is None:
        raise FileNotFoundError(f"No STAR gene-counts file in {rna_dir}")
    df = pd.read_csv(sample_file, sep="\t", comment="#", usecols=["gene_id", "gene_name"])
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    # gene_id has version suffix (e.g. ENSG00000000003.15) — strip
    df["ensembl_base"] = df["gene_id"].str.split(".").str[0]
    df = df[["ensembl_base", "gene_name"]].drop_duplicates(subset="ensembl_base")
    return df


def main() -> None:
    print("=== S4a: building outcome + treatment table ===")
    df = build_outcome_treatment()
    print(f"patients: {len(df)}")
    print(f"  bort_1L=1 : {(df['bort_1L'] == 1).sum()}")
    print(f"  bort_1L=0 : {(df['bort_1L'] == 0).sum()}")
    print(f"  had_2L=1  : {(df['had_2L'] == 1).sum()}")
    print(f"  had_2L=0  (censored): {(df['had_2L'] == 0).sum()}")
    print(f"  TT2L median (had_2L=1) : "
          f"{df.loc[df['had_2L'] == 1, 'tt2L_days'].median():.0f} days")
    print(f"  Censoring time median   : "
          f"{df.loc[df['had_2L'] == 0, 'tt2L_days'].median():.0f} days")
    out_path = PROC / "mmrf_outcomes_treatment.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"saved → {out_path}")

    print("\n=== S4a: building Ensembl→symbol map ===")
    sym = build_ensembl_symbol_map()
    print(f"map entries: {len(sym)}")
    for g in ("PSMB5", "PSMB1", "PSMB2", "TOP2A", "XPO1"):
        hits = sym[sym["gene_name"] == g]
        if len(hits):
            print(f"  {g} → {hits['ensembl_base'].tolist()}")
        else:
            print(f"  {g}: NOT FOUND")
    sym_path = PROC / "mmrf_ensembl_to_symbol.tsv"
    sym.to_csv(sym_path, sep="\t", index=False)
    print(f"saved → {sym_path}")


if __name__ == "__main__":
    main()
