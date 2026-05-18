#!/usr/bin/env python3
"""
scripts/mortfm_survival_baseline_suite.py
==========================================
v17.3 — mandatory survival baseline suite for the MMRF 29-patient TT2L cohort.

Comparators (all LOO on the same 29-patient split):
  1. Clinical-only Cox PH (L2-regularised, fit by Adam) on
     (ISS_ordinal, age_zscore, gender_male, bort_1L, n_treatments_norm)
  2. Permutation null (shuffles event labels, keeps marginals)
  3. Reference: the four LENS variants from
     logs/mortfm/lens_resistance_{baseline,clinical_only,graph_only,clinical_plus_graph}.json

The script writes:
  * results/mortfm/survival_baseline_suite.csv
  * logs/mortfm/survival_baseline_suite.json

The honest verdict the v17 plan asks for: NO patient-level claim is allowed
unless the full model beats clinical-only Cox AND the permutation null.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.longitudinal.clinical_features import encode_for_lens
from resistancemap.mortfm.survival import (
    concordance_index_bootstrap_ci, cox_fit_predict_loo,
    permutation_null_cindex,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_survival_baseline_suite")


def _load_mmrf_paired_with_outcomes():
    z = np.load("data/processed/mmrf_paired_z64.npz", allow_pickle=True)
    pids = [str(p) for p in z["patient_id"]]
    outcomes = pd.read_csv("data/processed/mmrf_outcomes_treatment.tsv", sep="\t")
    out_map = outcomes.set_index("submitter_id").to_dict("index")
    rows = []
    for i, pid in enumerate(pids):
        rec = out_map.get(pid)
        if rec is None:
            continue
        tt2l = rec.get("tt2L_days"); had_2l = rec.get("had_2L")
        if pd.isna(tt2l) or pd.isna(had_2l):
            continue
        rows.append({
            "patient_id": pid,
            "z0": z["z0"][i].astype(np.float32),
            "tt2l_days": float(tt2l),
            "event_observed": int(had_2l),
            "bort_1L": int(rec.get("bort_1L", 0) or 0),
        })
    return rows


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="results/mortfm/survival_baseline_suite.csv")
    ap.add_argument("--out-summary", default="logs/mortfm/survival_baseline_suite.json")
    ap.add_argument("--cox-l2", type=float, default=1e-2)
    ap.add_argument("--cox-epochs", type=int, default=300)
    ap.add_argument("--cox-lr", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    rows = _load_mmrf_paired_with_outcomes()
    if not rows:
        logger.error("No paired patients with outcomes found.")
        return 1
    logger.info("Loaded %d paired patients (%d observed)", len(rows),
                sum(r["event_observed"] for r in rows))

    enc = encode_for_lens(
        "data/processed/mmrf_outcomes_treatment.tsv",
        patient_ids=[r["patient_id"] for r in rows],
    )
    X_clin = enc.features  # (n, 5)
    event_time = [r["tt2l_days"] for r in rows]
    event_obs = [r["event_observed"] for r in rows]

    # --- 1. Clinical-only Cox LOO ----------------------------------------
    logger.info("Running clinical-only Cox LOO ...")
    cox_risks = cox_fit_predict_loo(
        X_clin, event_time, event_obs,
        epochs=args.cox_epochs, lr=args.cox_lr, l2=args.cox_l2,
    )
    cox_c, cox_lo, cox_hi = concordance_index_bootstrap_ci(
        event_time, event_obs, cox_risks, n_boot=1000,
    )
    cox_null = permutation_null_cindex(
        cox_risks, event_time, event_obs, n_permutations=args.n_perm,
    )
    logger.info("Clinical-only Cox: C=%.3f [%.3f, %.3f], null mean=%.3f, p=%.3f",
                cox_c, cox_lo, cox_hi, cox_null["null_mean"], cox_null["p_value"])

    # --- 2. Reference LENS variants --------------------------------------
    lens_variants = {}
    for variant in ["baseline", "clinical_only", "graph_only", "clinical_plus_graph"]:
        p = Path(f"logs/mortfm/lens_resistance_{variant}.json")
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        c_lo, c_hi = d["loo_cindex_95ci"]
        lens_variants[variant] = {
            "loo_cindex": d["loo_cindex"],
            "loo_cindex_95ci": [c_lo, c_hi],
            "ibs": d["integrated_brier_score"],
        }

    # --- 3. Build the comparator table -----------------------------------
    rows_out = [{
        "model": "clinical_only_cox",
        "n_features": int(X_clin.shape[1]),
        "loo_cindex": cox_c,
        "loo_cindex_lo": cox_lo,
        "loo_cindex_hi": cox_hi,
        "permutation_null_mean": cox_null["null_mean"],
        "permutation_null_std": cox_null["null_std"],
        "permutation_p_value": cox_null["p_value"],
        "beats_null_at_p_lt_0_05": cox_null["p_value"] < 0.05,
    }]
    for variant, m in lens_variants.items():
        rows_out.append({
            "model": f"LENS_{variant}",
            "n_features": None,
            "loo_cindex": m["loo_cindex"],
            "loo_cindex_lo": m["loo_cindex_95ci"][0],
            "loo_cindex_hi": m["loo_cindex_95ci"][1],
            "permutation_null_mean": None,
            "permutation_null_std": None,
            "permutation_p_value": None,
            "beats_null_at_p_lt_0_05": None,
        })
    df = pd.DataFrame(rows_out).sort_values("loo_cindex", ascending=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s", args.out_csv)
    print(df.to_string(index=False))

    # --- 4. Verdict --------------------------------------------------------
    best_lens = max(
        (m["loo_cindex"] for m in lens_variants.values()),
        default=float("nan"),
    )
    best_lens_name = max(
        lens_variants, key=lambda v: lens_variants[v]["loo_cindex"],
        default=None,
    ) if lens_variants else None
    lens_beats_cox = (best_lens >= cox_c) if not math.isnan(best_lens) else False
    lens_beats_null = (
        (lens_variants.get(best_lens_name, {}).get("loo_cindex_95ci", [0, 1])[0] or 0)
        > (cox_null["null_mean"] + 2 * cox_null["null_std"])
    ) if best_lens_name else False

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-survival-baselines",
        "n_patients": int(len(rows)),
        "n_observed_events": int(sum(event_obs)),
        "clinical_only_cox": {
            "loo_cindex": cox_c,
            "loo_cindex_95ci": [cox_lo, cox_hi],
            "permutation_null": cox_null,
            "beats_null_at_p_lt_0_05": cox_null["p_value"] < 0.05,
        },
        "lens_variants": lens_variants,
        "best_lens": {"name": best_lens_name, "cindex": best_lens} if best_lens_name else None,
        "lens_beats_cox_on_point_cindex": bool(lens_beats_cox),
        "lens_beats_cox_null_band": bool(lens_beats_null),
        "patient_level_claim_allowed": bool(lens_beats_cox and lens_beats_null and cox_null["p_value"] < 0.05),
        "honest_note": (
            "At n=29, no clinical-prediction claim is granted unless the best "
            "LENS variant point-estimate beats clinical-only Cox AND the LENS "
            "95% CI lower bound exceeds the clinical-only-Cox permutation null "
            "+ 2sd AND clinical-only Cox itself beats the null at p<0.05. "
            "This rule operationalises the v17 steering directive: 'Make "
            "clinical-only Cox/LENS the required baseline for every patient-level claim.'"
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Verdict: patient_level_claim_allowed = %s",
                summary["patient_level_claim_allowed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
