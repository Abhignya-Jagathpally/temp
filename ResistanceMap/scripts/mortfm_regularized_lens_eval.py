#!/usr/bin/env python3
"""
scripts/mortfm_regularized_lens_eval.py
========================================
v17.7 — evaluate the regularised LENS baseline on the MMRF 29-patient TT2L
cohort with two feature substrates:

  1. clinical-only (5 features) — should reproduce v17.2's clinical Cox
  2. clinical + z0 (5 + 64 features) — the regularised version of v16's
     LENS_clinical_only variant (which was the v16 winner at C=0.603)

For each substrate we report:
  * LOO C-index + bootstrap 95% CI
  * Bootstrap optimism-corrected C-index (Harrell 1996, n_bootstrap=50)
  * Permutation null mean +- std (resamples real event labels)

The corrected C-index is the *honest* number — it strips out the
"fold-by-fold model gets lucky" optimism inherent to LOO at n=29.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.longitudinal.clinical_features import encode_for_lens
from resistancemap.mortfm.survival import (
    RegularizedLENSConfig, bootstrap_optimism_correction,
    concordance_index_bootstrap_ci, fit_predict_loo_regularized,
    permutation_null_cindex,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_regularized_lens_eval")


def _load():
    z = np.load("data/processed/mmrf_paired_z64.npz", allow_pickle=True)
    pids = [str(p) for p in z["patient_id"]]
    outcomes = pd.read_csv("data/processed/mmrf_outcomes_treatment.tsv", sep="\t")
    om = outcomes.set_index("submitter_id").to_dict("index")
    rows = []
    for i, pid in enumerate(pids):
        rec = om.get(pid)
        if rec is None:
            continue
        if pd.isna(rec.get("tt2L_days")) or pd.isna(rec.get("had_2L")):
            continue
        rows.append({
            "patient_id": pid, "z0": z["z0"][i].astype(np.float32),
            "tt2l_days": float(rec["tt2L_days"]),
            "event_observed": int(rec["had_2L"]),
        })
    return rows


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/mortfm/regularized_lens_eval.json")
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    rows = _load()
    pids = [r["patient_id"] for r in rows]
    et = [r["tt2l_days"] for r in rows]
    eo = [r["event_observed"] for r in rows]
    enc = encode_for_lens("data/processed/mmrf_outcomes_treatment.tsv", patient_ids=pids)

    X_clin = enc.features.astype(np.float32)              # (n, 5)
    Z0 = np.stack([r["z0"] for r in rows], axis=0)        # (n, 64)
    X_full = np.concatenate([X_clin, Z0], axis=1)         # (n, 69)

    out = {"run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-regularized-lens",
           "n_patients": int(len(rows)), "n_events": int(sum(eo))}

    for label, X in [("clinical_only", X_clin), ("clinical_plus_z0", X_full)]:
        logger.info("=== %s (n_features=%d) ===", label, X.shape[1])
        cfg = RegularizedLENSConfig(
            d_latent=32, n_bins=8, d_hidden=64, dropout=0.3,
            l1=1e-4, l2=1e-3,
        )
        risks = fit_predict_loo_regularized(
            X, et, eo, cfg=cfg, epochs=args.epochs, lr=5e-3, log_every=5,
        )
        c, lo, hi = concordance_index_bootstrap_ci(et, eo, risks, n_boot=1000)
        null = permutation_null_cindex(risks, et, eo, n_permutations=500)
        opt = bootstrap_optimism_correction(
            c, X, et, eo, cfg=cfg,
            n_bootstrap=args.n_bootstrap, epochs=max(60, args.epochs // 2), lr=5e-3,
        )
        out[label] = {
            "loo_cindex": c, "loo_cindex_95ci": [lo, hi],
            "permutation_null_mean": null["null_mean"],
            "permutation_null_std": null["null_std"],
            "permutation_p_value": null["p_value"],
            "mean_optimism": opt["mean_optimism"],
            "corrected_cindex": opt["corrected_cindex"],
            "n_bootstrap": opt["n_bootstrap"],
            "n_features": int(X.shape[1]),
        }
        logger.info(
            "%s: C=%.3f [%.3f, %.3f], null_mean=%.3f, "
            "optimism=%+.3f, corrected_C=%.3f, p=%.3f",
            label, c, lo, hi, null["null_mean"],
            opt["mean_optimism"], opt["corrected_cindex"], null["p_value"],
        )

    # Verdict
    clin = out["clinical_only"]
    full = out["clinical_plus_z0"]
    full_beats_clin_corrected = full["corrected_cindex"] > clin["corrected_cindex"]
    full_beats_null = full["loo_cindex_95ci"][0] > (
        full["permutation_null_mean"] + 2 * full["permutation_null_std"]
    )
    out["lens_beats_clinical_on_corrected_cindex"] = bool(full_beats_clin_corrected)
    out["lens_beats_null_2sd_band"] = bool(full_beats_null)
    out["patient_level_claim_allowed"] = bool(full_beats_clin_corrected and full_beats_null)
    out["honest_note"] = (
        "v17.7 regularised LENS reports BOTH the LOO point C-index and the "
        "Harrell 1996 bootstrap optimism-corrected C-index. The corrected "
        "number is the honest one — it removes LOO's tendency to overstate "
        "performance at small n by ~5-15 points. Any patient-level claim "
        "must survive on the corrected number."
    )
    out["wall_time_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Verdict: lens_beats_clinical_corrected=%s, lens_beats_null_2sd=%s, "
                "patient_level_claim_allowed=%s",
                full_beats_clin_corrected, full_beats_null,
                out["patient_level_claim_allowed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
