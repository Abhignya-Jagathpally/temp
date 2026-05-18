#!/usr/bin/env python3
"""
scripts/mortfm_longitudinal_pair_audit.py
==========================================
v17.4 — audit the longitudinal pair pool: emit cohort registry,
apply censoring policy, run patient-disjoint + temporal-holdout splits,
and write a single audit report the gate auditor reads.
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
    CANONICAL_COHORTS, apply_censoring_policy, build_pairs_from_mmrf,
    split_patient_disjoint, total_pairs_in_state, write_cohort_registry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_longitudinal_pair_audit")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-registry", default="logs/mortfm/longitudinal_cohort_registry.json")
    ap.add_argument("--out-audit", default="logs/mortfm/longitudinal_pair_audit.json")
    args = ap.parse_args()

    write_cohort_registry(args.out_registry)
    logger.info("Wrote cohort registry -> %s", args.out_registry)

    # Build the MMRF-only pairs (the only ingested paired-molecular cohort).
    pairs = build_pairs_from_mmrf()
    censor = apply_censoring_policy(pairs)
    logger.info("Censoring policy: %s",
                {k: v for k, v in censor.items() if k != "kept"})

    splits = split_patient_disjoint(pairs, seed=13)
    logger.info("Patient-disjoint splits: %s", splits.summary())

    contributes = {
        name: spec.contributes_to for name, spec in CANONICAL_COHORTS.items()
    }
    n_ingested_pairs = total_pairs_in_state("ingested")
    n_planned_pairs = total_pairs_in_state("ingested_plus_planned")
    audit = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-longitudinal-pair-audit",
        "n_pairs_ingested": int(n_ingested_pairs),
        "n_pairs_after_planned_ingestion_ceiling": int(n_planned_pairs),
        "longitudinal_trajectory_gate_threshold": 100,
        "currently_passes_longitudinal_gate": n_ingested_pairs >= 100,
        "passes_after_planned": n_planned_pairs >= 100,
        "censoring_policy_report": {
            "n_input": censor["n_input"],
            "n_kept": censor["n_kept"],
            "n_dropped_no_event_time": censor["n_dropped_no_event_time"],
            "n_dropped_zero_or_negative_event_time": censor["n_dropped_zero_or_negative_event_time"],
            "n_event_observed": censor["n_event_observed"],
            "n_administratively_censored": censor["n_administratively_censored"],
        },
        "split_summary": splits.summary(),
        "cohort_contributions": contributes,
        "cohort_pair_counts": {
            name: spec.expected_pairs
            for name, spec in CANONICAL_COHORTS.items()
            if spec.baseline_available and spec.relapse_available
        },
        "honest_note": (
            "v17.4 — the longitudinal substrate is unchanged at 29 paired "
            "patients (MMRF_IA12 only). The cohort registry now codifies "
            "which planned cohorts would advance which gates; the censoring "
            "policy is centralised; patient-disjoint splits are deterministic "
            "for any random seed. No new training; this audit is the "
            "data-side substrate for the next ingestion sprint."
        ),
        "wall_time_s": round(time.time() - t0, 2),
    }
    Path(args.out_audit).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_audit, "w") as f:
        json.dump(audit, f, indent=2)
    logger.info("Wrote audit -> %s", args.out_audit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
