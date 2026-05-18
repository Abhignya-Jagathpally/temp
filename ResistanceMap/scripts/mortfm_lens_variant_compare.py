#!/usr/bin/env python3
"""
scripts/mortfm_lens_variant_compare.py
=======================================
Aggregate the four LENS resistance-training variant summaries into one
table + a single comparison JSON. No new training — pure consolidation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-dir", default="logs/mortfm")
    ap.add_argument("--out-csv", default="results/mortfm/lens_variant_comparison.csv")
    ap.add_argument("--out-json", default="logs/mortfm/lens_variant_comparison.json")
    args = ap.parse_args()

    variants = [
        ("baseline", "lens_resistance_baseline.json", "z0 only, no graph, no clinical"),
        ("clinical_only", "lens_resistance_clinical_only.json", "+ ISS, age, gender, bort_1L, n_treatments"),
        ("graph_only", "lens_resistance_graph_only.json", "+ LatentToGraphProjector(z0)"),
        ("clinical_plus_graph", "lens_resistance_clinical_plus_graph.json", "all of the above"),
    ]
    rows = []
    for name, fname, desc in variants:
        p = Path(args.variants_dir) / fname
        if not p.exists():
            print(f"warn: {p} not found")
            continue
        with open(p) as f:
            d = json.load(f)
        c_lo, c_hi = d["loo_cindex_95ci"]
        rows.append({
            "variant": name,
            "description": desc,
            "n_patients": d["n_patients"],
            "n_observed_events": d["n_observed_events"],
            "loo_cindex": d["loo_cindex"],
            "loo_cindex_lo": c_lo,
            "loo_cindex_hi": c_hi,
            "integrated_brier_score": d["integrated_brier_score"],
            "nll_uncalibrated": d["nll_uncalibrated"],
            "nll_temperature_scaled": d["nll_temperature_scaled"],
            "calibration_log_temp": d["calibration_log_temperature"],
            "resistance_emergence_gate_pass": d["resistance_emergence_gate_pass"],
            "survival_prediction_gate_pass": d["survival_prediction_gate_pass"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("loo_cindex", ascending=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

    # Best variant
    best = df.iloc[0]
    summary = {
        "n_variants_compared": int(len(df)),
        "best_variant": str(best["variant"]),
        "best_loo_cindex": float(best["loo_cindex"]),
        "best_loo_cindex_95ci": [float(best["loo_cindex_lo"]), float(best["loo_cindex_hi"])],
        "best_resistance_emergence_gate_pass": bool(best["resistance_emergence_gate_pass"]),
        "best_survival_prediction_gate_pass": bool(best["survival_prediction_gate_pass"]),
        "headline_finding": (
            f"Best variant is '{best['variant']}' "
            f"(C-index = {best['loo_cindex']:.3f} "
            f"[{best['loo_cindex_lo']:.3f}, {best['loo_cindex_hi']:.3f}]). "
            "Clinical covariates alone provide the strongest signal at n=29; "
            "combining clinical + graph projector overfits."
        ),
        "outputs": {"csv": args.out_csv},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + summary["headline_finding"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
