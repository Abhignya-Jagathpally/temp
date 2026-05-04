"""S2b: Identify paired baseline→relapse MMRF patients.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.1:
    N_p = 42 strict-paired patients with both ≥2 RNA-Seq timepoints AND a
    2nd-line transition; N_r = 319 Recurrent-BM (any paired); N_b = 994
    baseline-only.

Actual on-disk counts (RNA-seq files limit the cohort):
    paired (≥2 distinct timepoints): 29 patients
    strict-paired (paired + 2nd-line therapy record): 20 patients
    baseline-only (=N_total_with_RNA - 29): 758 patients

Outputs:
    data/processed/mmrf_paired_patients.tsv     # patient_id, t0_aliquot, t1_aliquot, has_2nd_line
    paper/v8_artifacts/v10_sprint2/s2b_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
PROC = ROOT / "data" / "processed"
SPRINT2 = ROOT / "paper" / "v8_artifacts" / "v10_sprint2"
SPRINT2.mkdir(parents=True, exist_ok=True)

ALIQUOT_TIMEPOINT_RE = re.compile(r"_T(\d+)_")


def parse_tp(aliquot_id: str) -> int:
    m = ALIQUOT_TIMEPOINT_RE.search(aliquot_id)
    return int(m.group(1)) if m else 999


def main() -> None:
    f2c = pd.read_csv(RAW / "file_to_case.tsv", sep="\t")
    f2c = f2c[f2c["modality"] == "rna"].copy()
    f2c["timepoint"] = f2c["aliquot_submitter_id"].map(parse_tp)

    # Per patient, find earliest two distinct timepoints with RNA-seq
    rows = []
    for patient_id, grp in f2c.groupby("patient_submitter_id"):
        tps = grp.sort_values("timepoint")
        unique_tps = tps.drop_duplicates(subset="timepoint")
        if len(unique_tps) < 2:
            continue
        t0_row = unique_tps.iloc[0]
        t1_row = unique_tps.iloc[1]
        rows.append({
            "patient_id": patient_id,
            "t0_aliquot": t0_row["aliquot_submitter_id"],
            "t0_timepoint": int(t0_row["timepoint"]),
            "t1_aliquot": t1_row["aliquot_submitter_id"],
            "t1_timepoint": int(t1_row["timepoint"]),
        })
    paired = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    print(f"[S2b] paired patients (≥2 distinct timepoints): {len(paired)}")

    # 2nd-line filter
    treat = pd.read_csv(RAW / "treatments.tsv", sep="\t")
    treat = treat[treat["treatment_or_therapy"] == "yes"]
    treat_lines = treat.groupby("submitter_id")["regimen_or_line_of_therapy"].apply(set).to_dict()
    def has_2nd(p: str) -> bool:
        s = treat_lines.get(p, set())
        return any("second" in (v or "").lower() for v in s)
    paired["has_2nd_line"] = paired["patient_id"].map(has_2nd)
    n_strict = int(paired["has_2nd_line"].sum())
    print(f"[S2b] strict-paired (paired + 2nd-line therapy): {n_strict}")

    out_path = PROC / "mmrf_paired_patients.tsv"
    paired.to_csv(out_path, sep="\t", index=False)
    print(f"[S2b] wrote {out_path}")

    summary = {
        "n_paired": int(len(paired)),
        "n_strict_paired_with_2nd_line": n_strict,
        "spec_target_N_p_strict": 42,
        "spec_target_N_r_paired": 319,
        "actual_vs_spec_strict_ratio": round(n_strict / 42.0, 3),
        "honest_disclosure": (
            "Spec assumed N_p=42 strict-paired and N_r=319 paired; on-disk RNA-seq "
            "files only support N=29 paired and N=20 strict-paired. Power for F5, F1, "
            "and L2-NIE in v10 is reduced proportionally."
        ),
    }
    summary_path = SPRINT2 / "s2b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S2b] wrote {summary_path}")


if __name__ == "__main__":
    main()
