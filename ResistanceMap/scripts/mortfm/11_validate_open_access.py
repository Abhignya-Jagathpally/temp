#!/usr/bin/env python
"""v20 Phase 6 — external validation on open-access cohorts.

Three independent validation paths, each evaluated on REAL survival / lab
labels and each independently skippable:

  1. GDC open-tier MMRF clinical (OS / vital_status, N~995)
  2. GSE136337 (B2M / albumin / LDH + OS, N~426) — tests the PK observation decoder
  3. GSE24080 (EFS / OS, N~559) — external survival generalisation

This is a *harness*, not a trainer. It consumes a model's emitted predictions
(a CSV/parquet with ``sample_id`` + ``risk_score`` and/or biomarker columns) and
computes C-index (with bootstrap CI) against the cohort's real labels. When a
cohort's data — or the model's predictions — are absent, the path is reported as
``skipped`` with the reason. Nothing is fabricated: no synthetic risk scores, no
imputed survival, no placebo metric.

Usage
-----
    # plan only (no data needed)
    python scripts/mortfm/11_validate_open_access.py --dry-run

    # GDC OS C-index from a model's predictions
    python scripts/mortfm/11_validate_open_access.py \
        --gdc-open-dir data/gdc_mmrf_open \
        --gdc-predictions preds/gdc_risk.csv

    # GSE136337 PK-decoder check + external survival
    python scripts/mortfm/11_validate_open_access.py \
        --geo-bulk-dir data/geo_bulk \
        --gse136337-predictions preds/gse136337.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_open_access")

# Repo root on path so `resistancemap` imports resolve when run as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Metrics (prefer the audited mortfm implementation; fall back to a pure-numpy
# Harrell C-index so the harness never silently no-ops).
# ---------------------------------------------------------------------------
def _cindex(event_time, event_observed, risk) -> float:
    try:
        from resistancemap.mortfm.survival.time_to_event_metrics import concordance_index
        return float(concordance_index(np.asarray(event_time), np.asarray(event_observed),
                                       np.asarray(risk)))
    except Exception:
        return _harrell_cindex(event_time, event_observed, risk)


def _cindex_ci(event_time, event_observed, risk, n_boot: int, seed: int):
    try:
        from resistancemap.mortfm.survival.time_to_event_metrics import (
            concordance_index_bootstrap_ci,
        )
        pt, lo, hi = concordance_index_bootstrap_ci(
            np.asarray(event_time), np.asarray(event_observed), np.asarray(risk),
            n_boot=n_boot, seed=seed,
        )
        return float(pt), float(lo), float(hi)
    except Exception:
        rng = np.random.RandomState(seed)
        et, eo, rk = map(np.asarray, (event_time, event_observed, risk))
        n = len(et)
        boots = []
        for _ in range(max(n_boot, 1)):
            idx = rng.randint(0, n, n)
            boots.append(_harrell_cindex(et[idx], eo[idx], rk[idx]))
        return float(_harrell_cindex(et, eo, rk)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _harrell_cindex(event_time, event_observed, risk) -> float:
    et, eo, rk = np.asarray(event_time), np.asarray(event_observed), np.asarray(risk)
    num = den = 0.0
    n = len(et)
    for i in range(n):
        if not eo[i]:
            continue
        for j in range(n):
            if et[j] > et[i]:
                den += 1
                if rk[i] > rk[j]:
                    num += 1
                elif rk[i] == rk[j]:
                    num += 0.5
    return num / den if den else float("nan")


def _load_predictions(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"predictions file not found: {p}")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------
def validate_gdc(gdc_dir: Optional[str], predictions: Optional[str],
                 n_boot: int, seed: int) -> Dict[str, Any]:
    if not gdc_dir:
        return {"status": "skipped", "reason": "--gdc-open-dir not provided"}
    from resistancemap.data.gdc_open_loader import extract_survival_labels
    clin_path = Path(gdc_dir) / "clinical_open.tsv"
    if not clin_path.exists():
        return {"status": "skipped", "reason": f"{clin_path} absent (run download_mmrf_open_clinical first)"}
    clinical = pd.read_csv(clin_path, sep="\t")
    labels = extract_survival_labels(clinical)
    summary = {
        "status": "data_present",
        "n_with_os": int(len(labels["time"])),
        "n_events": int(labels["event"].sum()),
    }
    preds = _load_predictions(predictions)
    if preds is None:
        summary["metric"] = "skipped — no --gdc-predictions (cohort summary only)"
        return summary
    if not {"submitter_id", "risk_score"} <= set(preds.columns):
        raise ValueError("gdc predictions need columns: submitter_id, risk_score")
    merged = pd.DataFrame({"submitter_id": labels["submitter_id"],
                           "time": labels["time"], "event": labels["event"]}).merge(
        preds[["submitter_id", "risk_score"]], on="submitter_id", how="inner")
    if merged.empty:
        return {"status": "error", "reason": "no submitter_id overlap between labels and predictions"}
    pt, lo, hi = _cindex_ci(merged["time"], merged["event"], merged["risk_score"], n_boot, seed)
    summary.update({"status": "evaluated", "n_matched": int(len(merged)),
                    "c_index": round(pt, 4), "c_index_ci": [round(lo, 4), round(hi, 4)]})
    return summary


def validate_gse136337(geo_bulk_dir: Optional[str], predictions: Optional[str],
                       n_boot: int, seed: int) -> Dict[str, Any]:
    if not geo_bulk_dir:
        return {"status": "skipped", "reason": "--geo-bulk-dir not provided"}
    from resistancemap.data.geo_bulk_loader import load_gse136337
    try:
        cohort = load_gse136337(geo_bulk_dir)
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": str(exc)}
    surv = cohort["survival"]
    out: Dict[str, Any] = {
        "status": "data_present",
        "n_with_os": int(len(surv["time"])),
        "n_events": int(surv["event"].sum()),
        "labs_resolved": {k: v for k, v in cohort["lab_columns"].items() if v},
    }
    preds = _load_predictions(predictions)
    if preds is None:
        out["pk_check"] = "skipped — no --gse136337-predictions"
        out["survival_metric"] = "skipped — no predictions"
        return out
    # PK-decoder check: compare predicted vs actual labs where both exist.
    meta = cohort["metadata"]
    pk_results = {}
    for lab, col in cohort["lab_columns"].items():
        pred_col = f"pred_{lab}"
        if col and pred_col in preds.columns and len(preds) == len(meta):
            actual = pd.to_numeric(meta[col], errors="coerce").to_numpy()
            pred = pd.to_numeric(preds[pred_col], errors="coerce").to_numpy()
            mask = np.isfinite(actual) & np.isfinite(pred)
            if mask.sum() >= 5:
                mae = float(np.mean(np.abs(actual[mask] - pred[mask])))
                rho = float(np.corrcoef(
                    pd.Series(actual[mask]).rank(), pd.Series(pred[mask]).rank())[0, 1])
                pk_results[lab] = {"n": int(mask.sum()), "mae": round(mae, 4),
                                   "spearman": round(rho, 4)}
    out["pk_check"] = pk_results or "no overlapping predicted/actual lab columns"
    if "risk_score" in preds.columns and len(preds) == len(surv["keep_mask"]):
        rk = preds["risk_score"].to_numpy()[surv["keep_mask"]]
        pt, lo, hi = _cindex_ci(surv["time"], surv["event"], rk, n_boot, seed)
        out["survival_metric"] = {"c_index": round(pt, 4),
                                  "c_index_ci": [round(lo, 4), round(hi, 4)]}
    else:
        out["survival_metric"] = "skipped — no aligned risk_score column"
    return out


def validate_gse24080(geo_bulk_dir: Optional[str], predictions: Optional[str],
                      n_boot: int, seed: int) -> Dict[str, Any]:
    if not geo_bulk_dir:
        return {"status": "skipped", "reason": "--geo-bulk-dir not provided"}
    from resistancemap.data.geo_bulk_loader import load_gse24080
    try:
        cohort = load_gse24080(geo_bulk_dir)
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": str(exc)}
    surv = cohort["survival"]
    out: Dict[str, Any] = {"status": "data_present", "n_with_efs": int(len(surv["time"])),
                           "n_events": int(surv["event"].sum())}
    preds = _load_predictions(predictions)
    if preds is None or "risk_score" not in (preds.columns if preds is not None else []):
        out["metric"] = "skipped — no aligned --gse24080-predictions risk_score"
        return out
    if len(preds) != len(surv["keep_mask"]):
        return {"status": "error", "reason": "prediction rows do not align with cohort rows"}
    rk = preds["risk_score"].to_numpy()[surv["keep_mask"]]
    pt, lo, hi = _cindex_ci(surv["time"], surv["event"], rk, n_boot, seed)
    out.update({"status": "evaluated", "c_index": round(pt, 4),
                "c_index_ci": [round(lo, 4), round(hi, 4)]})
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="v20 open-access external validation")
    ap.add_argument("--gdc-open-dir")
    ap.add_argument("--geo-bulk-dir")
    ap.add_argument("--gdc-predictions")
    ap.add_argument("--gse136337-predictions", dest="gse136337_predictions")
    ap.add_argument("--gse24080-predictions", dest="gse24080_predictions")
    ap.add_argument("--out", default="logs/mortfm/open_access_validation.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = [
        "=" * 70,
        "v20 Phase 6 — open-access external validation",
        "=" * 70,
        f"  GDC open dir:    {args.gdc_open_dir or '(skip)'}",
        f"  GEO bulk dir:    {args.geo_bulk_dir or '(skip)'}",
        f"  bootstrap iters: {args.n_boot}",
        "=" * 70,
    ]
    print("\n".join(plan))
    if args.dry_run:
        print("[dry-run] no metrics computed.")
        return 0

    report: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gdc_mmrf_os": validate_gdc(args.gdc_open_dir, args.gdc_predictions, args.n_boot, args.seed),
        "gse136337": validate_gse136337(args.geo_bulk_dir, args.gse136337_predictions, args.n_boot, args.seed),
        "gse24080": validate_gse24080(args.geo_bulk_dir, args.gse24080_predictions, args.n_boot, args.seed),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
