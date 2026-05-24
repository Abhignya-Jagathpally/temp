#!/usr/bin/env python3
"""
scripts/mortfm_build_longitudinal_dataset.py
============================================
Lane 4+5 — build the MORT-FM longitudinal training substrate from MMRF.

This is the script the trajectory + resistance-emergence claim gates
read from. It:

  1. Loads MMRF paired-patient + outcomes tables.
  2. Builds LongitudinalPair records (29 from MMRF v15 data).
  3. Writes a tidy parquet that future Block-F training will consume.
  4. Reports — honestly — what each evidence level this corpus supports.

Honest verdict
--------------
* MMRF gives us **same-patient baseline → followup molecular pairs**, which
  is Level 3 evidence per the v15 plan (true molecular trajectory).
* BUT — and the gate needs to see this — MMRF's t0/t1 timepoint labels are
  **enum strings** ("Baseline" / "Pre-maintenance" / "Relapse"), not real
  calendar days. We back-fill delta_t_days with tt2L_days when an observed
  2L event exists; otherwise delta_t_days stays 0 and the trajectory gate
  refuses to grant a calendar-time trajectory claim from that pair.
* time_to_next_treatment (TT2L) is a clinical resistance proxy — observed
  2L means 1L failed — and supports the resistance_emergence claim under
  ``progression_free_survival``-equivalent endpoint semantics.

Outputs:
  * data/processed/longitudinal/temporal_pairs.parquet
  * logs/mortfm/longitudinal_data_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.longitudinal import (
    LongitudinalDataset, build_pairs_from_mmrf,
)
from resistancemap.mortfm.longitudinal.temporal_pair_builder import summarise_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_build_longitudinal_dataset")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="data/processed/mmrf_paired_patients.tsv")
    ap.add_argument("--outcomes", default="data/processed/mmrf_outcomes_treatment.tsv")
    ap.add_argument("--out-parquet",
                    default="data/processed/longitudinal/temporal_pairs.parquet")
    ap.add_argument("--out-report",
                    default="logs/mortfm/longitudinal_data_report.json")
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas",
                    help="Processing engine (default: pandas). Use 'spark' for "
                         "large-scale runs via PySpark.")
    args = ap.parse_args()

    # ----- Spark engine path ------------------------------------------------
    if args.engine == "spark":
        from resistancemap.data.spark_longitudinal import (
            build_longitudinal_panel_spark,
        )
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            logger.error(
                "PySpark is required for --engine spark. "
                "Install with:  pip install 'pyspark>=3.4'"
            )
            return 1
        spark = SparkSession.builder.appName(
            "mortfm_build_longitudinal_dataset"
        ).getOrCreate()
        data_dir = str(Path(args.paired).parent)
        build_longitudinal_panel_spark(spark, data_dir, args.out_parquet)
        logger.info("Spark engine: wrote panel -> %s", args.out_parquet)
        spark.stop()
        return 0

    # ----- Pandas engine path (default) ------------------------------------
    pairs = build_pairs_from_mmrf(paired_tsv=args.paired, outcomes_tsv=args.outcomes)
    ds = LongitudinalDataset(pairs)
    ds.to_parquet(args.out_parquet)
    s = summarise_pairs(pairs)

    # --- Gate decisions --------------------------------------------------
    # gate_longitudinal_trajectory thresholds (registry):
    #   min_longitudinal_pairs: 100
    #   min_unique_patients_with_2_timepoints: 50
    #   require_real_time_units: True
    LONGITUDINAL_TRAJ_PASS = (
        s["n_pairs"] >= 100
        and s["n_unique_patients"] >= 50
        and s["n_with_real_calendar_time"] == s["n_pairs"]
    )
    # gate_resistance_claim with TT2L endpoint (an observed 2L proxy is a
    # legitimate clinical resistance event).
    RESISTANCE_EMERGENCE_PASS = (
        s["n_with_endpoint"] >= 10
        and s["n_with_observed_event"] >= 5
    )

    report = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-longitudinal-dataset",
        "cohort": "MMRF",
        "summary": s,
        "evidence_levels": {
            "Level_0_pseudotime_only": False,  # MMRF is not pseudotime
            "Level_1_repeated_cross_sectional": True,
            "Level_2_clinical_longitudinal_outcome": s["n_with_endpoint"] > 0,
            "Level_3_patient_level_molecular_pairs": s["n_pairs"] > 0,
            "Level_4_perturbation_time_series": False,  # MMRF has no perturbation
        },
        "gate_decisions": {
            "longitudinal_trajectory": {
                "thresholds": {
                    "min_longitudinal_pairs": 100,
                    "min_unique_patients_with_2_timepoints": 50,
                    "require_real_time_units": True,
                },
                "pass": bool(LONGITUDINAL_TRAJ_PASS),
                "honest_reason": (
                    f"n_pairs={s['n_pairs']} (need 100); "
                    f"n_unique_patients={s['n_unique_patients']} (need 50); "
                    f"n_with_real_calendar_time={s['n_with_real_calendar_time']} "
                    "(MMRF timepoints are enum strings, not calendar days; "
                    "TT2L was used to back-fill delta_t_days for observed 2L "
                    "events only)."
                ),
            },
            "resistance_emergence_TT2L": {
                "thresholds": {
                    "min_pairs_with_endpoint": 10,
                    "min_observed_events": 5,
                },
                "pass": bool(RESISTANCE_EMERGENCE_PASS),
                "endpoint": "time_to_next_treatment",
                "note": (
                    "An observed 2L switch implies 1L regimen failed — "
                    "this IS a resistance-emergence-class endpoint per the "
                    "endpoint registry (progression / progression_free_survival "
                    "semantics). To grant the resistance_emergence claim we "
                    "additionally need a trained MORT-FM-LENS run reporting "
                    "C-index, calibration, and patient-disjoint test."
                ),
            },
        },
        "outputs": {
            "parquet": args.out_parquet,
            "report": args.out_report,
        },
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Longitudinal data report -> %s", args.out_report)
    logger.info("Summary: %s", json.dumps(report["summary"], indent=2, default=str))
    logger.info("Gate decisions: %s",
                json.dumps({k: v["pass"] for k, v in report["gate_decisions"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
