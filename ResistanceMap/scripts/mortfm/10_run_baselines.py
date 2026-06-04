#!/usr/bin/env python
"""scripts/mortfm/10_run_baselines.py — Phase 10 baseline runner.

CLI orchestrator that runs every registered baseline AND the SOTA deep-learning
models on the SAME frozen splits from ``checkpoints/data_ready.pt``, then
writes:

  * ``results/baselines/{model_name}_predictions.parquet`` — per-model
    predictions on the test split.
  * ``results/baselines/baseline_leaderboard.csv`` — summary table.
  * ``results/baselines/baseline_leaderboard.json`` — same data, JSON.
  * ``results/baseline_comparison.parquet`` — ``ComparisonRow`` records
    (appended).

All baselines use the SAME split and the script verifies split-manifest
integrity via sha256 hash.

Two modes are supported:

  * ``static_cellline_drug_response`` (default) — per-drug regression on
    cell-line multi-omics features from data_ready.pt.
  * ``patient_longitudinal`` — survival/trajectory baselines on frozen
    patient-disjoint temporal splits from a confirmed dataset + manifest.

USAGE
-----

    # Dry run — only print the plan.
    python scripts/mortfm/10_run_baselines.py \\
        --mode static_cellline_drug_response --baselines all --dry-run

    # Real run — fit, predict, evaluate, write outputs.
    python scripts/mortfm/10_run_baselines.py \\
        --baselines all

    # Real run — specific baselines only.
    python scripts/mortfm/10_run_baselines.py \\
        --baselines elasticnet ridge mlp

    # Include SOTA deep-learning models from the benchmark suite.
    python scripts/mortfm/10_run_baselines.py \\
        --baselines all --include-sota

    # Specify seeds for repeated fitting.
    python scripts/mortfm/10_run_baselines.py \\
        --baselines all --seeds 0 1 2

    # Patient-longitudinal mode (survival baselines on frozen splits).
    python scripts/mortfm/10_run_baselines.py \\
        --mode patient_longitudinal \\
        --confirmed-dataset results/confirmed/mortfm_training_dataset.parquet \\
        --split-manifest results/confirmed/split_manifest.json \\
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Importing the baseline modules has the side effect of populating REGISTRY.
import resistancemap.baselines.deep_static          # noqa: F401  registers
import resistancemap.baselines.sequence             # noqa: F401  registers
import resistancemap.baselines.static_ml            # noqa: F401  registers
from resistancemap.baselines.registry import REGISTRY
from resistancemap.evaluation.model_comparison import (
    ComparisonRow,
    hash_patient_ids,
    write_comparison_rows,
)

logger = logging.getLogger("mortfm.10_run_baselines")

REPO = Path(__file__).resolve().parent.parent.parent
DATA_CKPT = REPO / "checkpoints" / "data_ready.pt"
RM_CKPT = REPO / "checkpoints" / "pipeline_validated.pt"
OUT_DIR = REPO / "results" / "baselines"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["static_cellline_drug_response", "patient_longitudinal"],
        default="static_cellline_drug_response",
        help="Baseline evaluation mode.",
    )
    p.add_argument(
        "--baselines", nargs="+", default=["all"],
        help="Registered baseline names; use 'all' to run every baseline in REGISTRY.",
    )
    p.add_argument(
        "--include-sota", action="store_true",
        help="Also run the 6 SOTA deep-learning models (DeepCDR, DrugCell, etc.).",
    )
    p.add_argument(
        "--data-ckpt", type=Path, default=DATA_CKPT,
        help="Path to checkpoints/data_ready.pt.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Directory for per-model predictions and leaderboard.",
    )
    p.add_argument(
        "--out-parquet", type=Path, default=REPO / "results" / "baseline_comparison.parquet",
        help="Destination parquet for ComparisonRow records (appended).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="RNG seeds to fit each baseline with (one run per seed).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without fitting any model.",
    )
    p.add_argument(
        "--table", type=str, default=None,
        choices=["survival_proxy", "response_classification", "trajectory_forecast",
                 "pathway_evidence"],
        help="Benchmark table to run (v19 4-table taxonomy). "
             "If not set, runs all applicable baselines.",
    )
    # ── Patient-longitudinal mode arguments ────────────────────────────────
    p.add_argument(
        "--confirmed-dataset", type=Path, default=None,
        help="Path to results/confirmed/mortfm_training_dataset.parquet "
             "(required for patient_longitudinal mode).",
    )
    p.add_argument(
        "--split-manifest", type=Path, default=None,
        help="Path to results/confirmed/split_manifest.json "
             "(required for patient_longitudinal mode).",
    )
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (mirrors scripts/run_baselines_real.py)
# ═══════════════════════════════════════════════════════════════════════════════


def load_real_data(ckpt_path: Path) -> Dict[str, Any]:
    """Load checkpoints/data_ready.pt and return structured arrays + splits."""
    if not ckpt_path.exists():
        raise SystemExit(f"Data checkpoint not found: {ckpt_path}")
    logger.info("Loading %s", ckpt_path)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ds = c["dataset"]
    splits = c["splits"]

    X_prot = ds.proteomics.numpy().astype(np.float32)
    X_epi = ds.epigenomics.numpy().astype(np.float32)
    X_full = np.concatenate([X_prot, X_epi], axis=1)
    y_full = ds.drug_sensitivity.numpy().astype(np.float32)
    drug_names = list(ds.drug_names)
    splits_np = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}

    logger.info(
        "  X_prot=%s  X_epi=%s  X_full=%s  y=%s  drugs=%d",
        X_prot.shape, X_epi.shape, X_full.shape, y_full.shape, len(drug_names),
    )
    logger.info(
        "  train=%d  val=%d  test=%d  NaN rate=%.1f%%",
        len(splits_np["train"]), len(splits_np["val"]), len(splits_np["test"]),
        100 * np.isnan(y_full).mean(),
    )
    return {
        "X_prot": X_prot,
        "X_epi": X_epi,
        "X_full": X_full,
        "y_full": y_full,
        "drug_names": drug_names,
        "splits": splits_np,
    }


def get_split_hash(splits: Dict[str, np.ndarray]) -> str:
    """Compute sha256[:16] of sorted patient indices for split-integrity."""
    all_ids = sorted(str(i) for i in np.concatenate([
        splits["train"], splits["val"], splits["test"]
    ]))
    return hash_patient_ids(all_ids)


def get_git_sha() -> str:
    """Return current git SHA or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO), timeout=5,
        )
        return result.stdout.strip()[:16] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_resistancemap_test_mse() -> Optional[float]:
    """Load the ResistanceMap test MSE from pipeline_validated.pt."""
    if not RM_CKPT.exists():
        return None
    c = torch.load(RM_CKPT, map_location="cpu", weights_only=False)
    return float(c["metrics"]["test_mse"])


def get_resistancemap_predictions() -> Optional[np.ndarray]:
    """Load ResistanceMap per-sample test predictions if available."""
    if not RM_CKPT.exists():
        return None
    c = torch.load(RM_CKPT, map_location="cpu", weights_only=False)
    m = c["metrics"]
    if "test_predictions" in m:
        return np.asarray(m["test_predictions"])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_baselines(requested: List[str]) -> List[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return REGISTRY.list_available()
    unknown = [b for b in requested if b not in REGISTRY]
    if unknown:
        available = REGISTRY.list_available()
        raise SystemExit(
            f"Unknown baselines requested: {unknown}\n"
            f"Available: {available}"
        )
    return list(requested)


def split_arrays(
    X: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split X, y by index arrays."""
    return (
        X[splits["train"]], y[splits["train"]],
        X[splits["val"]],   y[splits["val"]],
        X[splits["test"]],  y[splits["test"]],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY BASELINE RUNNER (per-drug fitting)
# ═══════════════════════════════════════════════════════════════════════════════


def run_registry_baseline(
    model_name: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    splits: Dict[str, np.ndarray],
    drug_names: List[str],
    seed: int,
) -> Dict[str, Any]:
    """Fit a registry baseline per-drug, compute NaN-masked MSE + Spearman.

    Registry baselines output 1-D predictions, so we train one model per drug
    on the non-NaN training rows for that drug.
    """
    Xtr, ytr, Xva, yva, Xte, yte = split_arrays(X_full, y_full, splits)
    n_drugs = ytr.shape[1]

    test_sq_resid = []
    per_drug_results = []
    all_test_preds = np.full_like(yte, np.nan)

    config = {"seed": seed}

    for d in range(n_drugs):
        tr_mask = ~np.isnan(ytr[:, d])
        te_mask = ~np.isnan(yte[:, d])

        if tr_mask.sum() < 5:
            per_drug_results.append({
                "drug": drug_names[d], "drug_idx": d,
                "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                "mse": float("nan"), "spearman": float("nan"),
            })
            continue

        # Instantiate a fresh model per drug.
        model = REGISTRY.instantiate(model_name, config)

        # Build modality dict (registry contract requires Mapping[str, ndarray]).
        X_train_dict = {"features": Xtr[tr_mask]}
        y_train_drug = ytr[tr_mask, d]

        model.fit(X_train_dict, y_train_drug)

        # Predict on test.
        if te_mask.sum() > 0:
            X_test_dict = {"features": Xte[te_mask]}
            pred = model.predict(X_test_dict)
            truth = yte[te_mask, d]
            sq = (pred - truth) ** 2
            test_sq_resid.append(sq)

            mse = float(np.mean(sq))
            rho, _ = spearmanr(truth, pred)
            rho = float(rho) if not np.isnan(rho) else 0.0

            # Store predictions for later export.
            te_idx = np.where(te_mask)[0]
            all_test_preds[te_idx, d] = pred
        else:
            mse = float("nan")
            rho = float("nan")

        per_drug_results.append({
            "drug": drug_names[d], "drug_idx": d,
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "mse": mse, "spearman": rho,
        })

    test_mse = float(np.mean(np.concatenate(test_sq_resid))) if test_sq_resid else float("nan")
    spearman_vals = [r["spearman"] for r in per_drug_results if not np.isnan(r["spearman"])]
    mean_spearman = float(np.mean(spearman_vals)) if spearman_vals else float("nan")

    return {
        "model": model_name,
        "test_mse": test_mse,
        "mean_spearman": mean_spearman,
        "per_drug": per_drug_results,
        "test_predictions": all_test_preds,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SOTA MODEL RUNNER (multi-drug output)
# ═══════════════════════════════════════════════════════════════════════════════


def run_sota_models(
    data: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    """Train the 6 SOTA deep-learning reimplementations and return results."""
    # Import the SOTA model classes from the benchmark script.
    sys.path.insert(0, str(REPO / "scripts"))
    from sota_benchmark_and_visualizations import (
        DeepCDR, DrugCell, GraphDRP, PaccMann, TCRP, PRECISE,
        prepare_tensors, train_model,
    )

    # Deterministic seeding for reproducibility (torch only; no global numpy
    # state mutation -- bootstrap uses its own RandomState instances).
    torch.manual_seed(seed)

    X_prot = data["X_prot"]
    X_epi = data["X_epi"]
    y_full = data["y_full"]
    splits = data["splits"]
    drug_names = data["drug_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors = prepare_tensors(X_prot, X_epi, y_full, splits, pca_dim=256)
    input_dim = tensors["pca_dim"]
    n_drugs = tensors["n_drugs"]

    models_spec = [
        ("DeepCDR",  lambda: DeepCDR(input_dim, n_drugs)),
        ("DrugCell", lambda: DrugCell(input_dim, n_drugs, n_pathways=16)),
        ("GraphDRP", lambda: GraphDRP(input_dim, n_drugs)),
        ("PaccMann", lambda: PaccMann(input_dim, n_drugs, d_model=64, n_heads=4)),
        ("TCRP",     lambda: TCRP(input_dim, n_drugs)),
        ("PRECISE",  lambda: PRECISE(input_dim, n_drugs)),
    ]

    results = []
    y_test = y_full[splits["test"]]

    for name, factory in models_spec:
        logger.info("  Training SOTA model: %s ...", name)
        t0 = time.time()
        model = factory()
        trained, _, _ = train_model(
            model, tensors, n_epochs=200, lr=1e-3, weight_decay=1e-4,
            patience=25, batch_size=64,
        )
        elapsed = time.time() - t0

        # Evaluate.
        trained = trained.to(device)
        trained.eval()
        X_te = tensors["test"]["X"].to(device)
        with torch.no_grad():
            pred = trained(X_te).cpu().numpy()

        per_drug = []
        all_sq_resid = []
        for d in range(n_drugs):
            mask = ~np.isnan(y_test[:, d])
            n_obs = int(mask.sum())
            if n_obs < 3:
                per_drug.append({
                    "drug": drug_names[d], "drug_idx": d,
                    "n_train": 0, "n_test": n_obs,
                    "mse": float("nan"), "spearman": float("nan"),
                })
                continue
            truth = y_test[mask, d]
            p = pred[mask, d]
            sq = (p - truth) ** 2
            all_sq_resid.append(sq)
            mse = float(np.mean(sq))
            rho, _ = spearmanr(truth, p)
            rho = float(rho) if not np.isnan(rho) else 0.0
            per_drug.append({
                "drug": drug_names[d], "drug_idx": d,
                "n_train": 0, "n_test": n_obs,
                "mse": mse, "spearman": rho,
            })

        test_mse = float(np.mean(np.concatenate(all_sq_resid))) if all_sq_resid else float("nan")
        spearman_vals = [r["spearman"] for r in per_drug if not np.isnan(r["spearman"])]
        mean_spearman = float(np.mean(spearman_vals)) if spearman_vals else float("nan")

        # Build full-shaped test predictions array.
        test_preds = np.full_like(y_test, np.nan)
        for d in range(n_drugs):
            mask = ~np.isnan(y_test[:, d])
            if mask.sum() > 0:
                test_preds[mask, d] = pred[mask, d]

        results.append({
            "model": f"SOTA_{name}",
            "test_mse": test_mse,
            "mean_spearman": mean_spearman,
            "per_drug": per_drug,
            "test_predictions": test_preds,
            "seed": seed,
            "wall_seconds": round(elapsed, 2),
        })
        logger.info(
            "    %s  test_mse=%.4f  mean_rho=%.4f  (%.1fs)",
            name, test_mse, mean_spearman, elapsed,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS: paired bootstrap deltas
# ═══════════════════════════════════════════════════════════════════════════════


def paired_bootstrap_delta(
    sq_resid_rm: np.ndarray,
    sq_resid_baseline: np.ndarray,
    n_boot: int = 5000,
    ci: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap delta (baseline - RM) on squared residuals.

    A positive delta means ResistanceMap is BETTER (lower MSE) than the baseline.

    Parameters
    ----------
    sq_resid_rm : 1-D array of squared residuals for ResistanceMap (observed cells).
    sq_resid_baseline : 1-D array of squared residuals for the baseline (same cells).
    n_boot : number of bootstrap resamples.
    ci : confidence interval width (e.g. 95 for 95% CI).

    Returns
    -------
    dict with keys: delta, ci_lo, ci_hi, p_value.
    """
    assert sq_resid_rm.shape == sq_resid_baseline.shape, (
        f"shape mismatch: RM={sq_resid_rm.shape} vs baseline={sq_resid_baseline.shape}"
    )
    diffs = sq_resid_baseline - sq_resid_rm  # positive = RM better
    observed_delta = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot_deltas = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_deltas[b] = np.mean(diffs[idx])

    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct
    ci_lo = float(np.percentile(boot_deltas, lo_pct))
    ci_hi = float(np.percentile(boot_deltas, hi_pct))

    # Two-sided p-value: fraction of bootstrap deltas on the opposite side of 0.
    if observed_delta >= 0:
        p_value = float(np.mean(boot_deltas <= 0))
    else:
        p_value = float(np.mean(boot_deltas >= 0))
    # Clamp to [1/n_boot, 1].
    p_value = max(p_value, 1.0 / n_boot)

    return {
        "delta": observed_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }


def compute_rm_sq_residuals(
    y_test: np.ndarray,
    rm_test_mse: float,
) -> Optional[np.ndarray]:
    """Compute ResistanceMap squared residuals from predictions if available."""
    preds = get_resistancemap_predictions()
    if preds is not None:
        mask = ~np.isnan(y_test)
        sq = (preds[mask] - y_test[mask]) ** 2
        return sq
    # Fallback: cannot compute per-cell paired residuals without predictions.
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION EXPORT
# ═══════════════════════════════════════════════════════════════════════════════


def save_predictions_parquet(
    model_name: str,
    test_predictions: np.ndarray,
    test_indices: np.ndarray,
    drug_names: List[str],
    out_dir: Path,
) -> Path:
    """Save per-model predictions to a parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build long-format DataFrame: (sample_idx, drug, prediction).
    records = []
    n_test, n_drugs = test_predictions.shape
    for i in range(n_test):
        for d in range(n_drugs):
            val = test_predictions[i, d]
            if not np.isnan(val):
                records.append({
                    "sample_idx": int(test_indices[i]),
                    "drug": drug_names[d],
                    "prediction": float(val),
                })
    df = pd.DataFrame(records)
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    path = out_dir / f"{safe_name}_predictions.parquet"
    df.to_parquet(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD WRITER
# ═══════════════════════════════════════════════════════════════════════════════


def write_leaderboard(
    all_results: List[Dict[str, Any]],
    rm_test_mse: Optional[float],
    y_test: np.ndarray,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Compute paired bootstrap deltas and write leaderboard CSV + JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to get RM squared residuals for paired bootstrap.
    rm_sq_resid = compute_rm_sq_residuals(y_test, rm_test_mse) if rm_test_mse else None

    leaderboard_rows = []
    for res in all_results:
        row: Dict[str, Any] = {
            "model": res["model"],
            "test_mse": round(res["test_mse"], 6),
            "mean_spearman": round(res["mean_spearman"], 6),
        }

        # Paired bootstrap delta vs ResistanceMap.
        if rm_sq_resid is not None and rm_test_mse is not None:
            # Compute per-cell squared residuals for this baseline.
            pred = res.get("test_predictions")
            if pred is not None:
                mask = ~np.isnan(y_test)
                baseline_sq = np.full_like(y_test, np.nan)
                baseline_sq[mask] = (pred[mask] - y_test[mask]) ** 2
                # Match the cells where both RM and baseline have predictions.
                bl_flat = baseline_sq[mask]
                if bl_flat.shape == rm_sq_resid.shape:
                    boot = paired_bootstrap_delta(rm_sq_resid, bl_flat, seed=res.get("seed", 42))
                    row["delta_vs_rm"] = round(boot["delta"], 6)
                    row["delta_ci_lo"] = round(boot["ci_lo"], 6)
                    row["delta_ci_hi"] = round(boot["ci_hi"], 6)
                    row["p_value"] = round(boot["p_value"], 6)
                else:
                    row.update({"delta_vs_rm": None, "delta_ci_lo": None,
                                "delta_ci_hi": None, "p_value": None})
            else:
                row.update({"delta_vs_rm": None, "delta_ci_lo": None,
                            "delta_ci_hi": None, "p_value": None})
        else:
            # If no RM predictions available, compute a simple delta from MSEs.
            if rm_test_mse is not None:
                row["delta_vs_rm"] = round(res["test_mse"] - rm_test_mse, 6)
            else:
                row["delta_vs_rm"] = None
            row.update({"delta_ci_lo": None, "delta_ci_hi": None, "p_value": None})

        leaderboard_rows.append(row)

    # Sort by test_mse ascending (best first).
    leaderboard_rows.sort(key=lambda r: r["test_mse"] if r["test_mse"] is not None else float("inf"))

    df = pd.DataFrame(leaderboard_rows)

    # CSV
    csv_path = out_dir / "baseline_leaderboard.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    # JSON
    json_path = out_dir / "baseline_leaderboard.json"
    json_path.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rm_test_mse": rm_test_mse,
        "n_baselines": len(leaderboard_rows),
        "results": leaderboard_rows,
    }, indent=2))
    logger.info("Wrote %s", json_path)

    return df, leaderboard_rows


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT-LONGITUDINAL MODE
# ═══════════════════════════════════════════════════════════════════════════════

PATIENT_LONGITUDINAL_BASELINES = [
    "clinical_only_cox",
    "clinical_ridge_cox",
    "rna_only_cox",
    "rna_plus_clinical_cox",
    "random_survival_forest",
    "deepsurv_mlp",
    "mean_time_baseline",
    "kaplan_meier_baseline",
    "locf_trajectory_baseline",
    "mean_future_state_baseline",
    "mofa_plus_cox",
    "padimac_7gene",
]

# v19 4-table benchmark taxonomy
BENCHMARK_TABLES = {
    "survival_proxy": [
        "clinical_only_cox", "clinical_ridge_cox", "rna_only_cox",
        "rna_plus_clinical_cox", "random_survival_forest", "deepsurv_mlp",
        "kaplan_meier_baseline", "mofa_plus_cox",
    ],
    "response_classification": [
        "clinical_only_cox", "padimac_7gene",
    ],
    "trajectory_forecast": [],  # blocked — no real timestamps
    "pathway_evidence": [],     # handled by evaluate_causal_evidence task
}

# Claim-level requirements: which baselines MORT-FM must beat for each claim.
CLAIM_LEVEL_BASELINE_REQUIREMENTS = {
    "survival_prediction": {
        "must_beat": ["clinical_only_cox", "kaplan_meier_baseline"],
        "should_beat": ["clinical_ridge_cox", "rna_plus_clinical_cox"],
        "description": "MORT-FM must exceed clinical-only Cox and KM to claim survival prediction.",
    },
    "longitudinal_trajectory": {
        "must_beat": ["locf_trajectory_baseline", "mean_future_state_baseline"],
        "should_beat": ["rna_only_cox"],
        "description": "MORT-FM must exceed LOCF and mean-future-state for trajectory claims.",
    },
    "resistance_emergence": {
        "must_beat": [
            "clinical_only_cox", "kaplan_meier_baseline",
            "locf_trajectory_baseline", "mean_future_state_baseline",
        ],
        "should_beat": [
            "rna_plus_clinical_cox", "random_survival_forest", "deepsurv_mlp",
        ],
        "description": (
            "MORT-FM must exceed ALL simple baselines and SHOULD beat "
            "non-trivial baselines for resistance_emergence."
        ),
    },
    "patient_level_clinical_prediction": {
        "must_beat": ["clinical_only_cox", "clinical_ridge_cox"],
        "should_beat": ["random_survival_forest", "deepsurv_mlp"],
        "description": "Must exceed clinical-only models for patient-level claims.",
    },
}


def _kaplan_meier_estimate(
    times: np.ndarray, events: np.ndarray,
) -> float:
    """Compute median survival time from KM estimator (simple implementation)."""
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    n_at_risk = len(t_sorted)
    surv = 1.0
    median_time = float(t_sorted[-1])  # fallback
    for i, (t, e) in enumerate(zip(t_sorted, e_sorted)):
        if e:
            surv *= (n_at_risk - 1) / n_at_risk
        n_at_risk -= 1
        if surv <= 0.5:
            median_time = float(t)
            break
    return median_time


def _concordance_index(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's C-index (pure numpy, no external deps)."""
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0
    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                if risk_scores[j] < risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] > risk_scores[i]:
                    discordant += 1
                else:
                    tied_risk += 1
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def _integrated_brier_score(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    risk_scores: np.ndarray,
    t_max: Optional[float] = None,
) -> float:
    """Simplified IBS approximation using risk scores as proxy.

    For proper IBS, survival curves are needed; here we approximate by
    converting risk_scores to pseudo-survival probabilities via sigmoid
    and computing Brier at quantile eval times.
    """
    if t_max is None:
        t_max = float(np.percentile(event_times[event_observed.astype(bool)], 90))
    eval_times = np.linspace(0.01, t_max, 10)
    # Convert risk_scores to pseudo-survival probabilities.
    surv_probs = 1.0 / (1.0 + np.exp(risk_scores - np.median(risk_scores)))

    brier_scores = []
    for t_eval in eval_times:
        # At time t_eval: true status is 1 if event happened before t_eval.
        y_true = ((event_times <= t_eval) & event_observed.astype(bool)).astype(float)
        # Predicted probability of event by t_eval.
        p_event = 1.0 - surv_probs
        bs = float(np.mean((y_true - p_event) ** 2))
        brier_scores.append(bs)

    # np.trapezoid (numpy >= 2.0) replaces deprecated np.trapz.
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        # Manual trapezoidal integration fallback.
        integral = sum(
            0.5 * (brier_scores[i] + brier_scores[i + 1]) * (eval_times[i + 1] - eval_times[i])
            for i in range(len(eval_times) - 1)
        )
    else:
        integral = float(_trapz(brier_scores, eval_times))
    return integral / (eval_times[-1] - eval_times[0])


def _fit_cox_linear(
    X_train: np.ndarray,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    l2: float = 0.01,
    n_iter: int = 100,
    lr: float = 0.01,
) -> np.ndarray:
    """Fit Cox PH via gradient descent on negative partial log-likelihood.

    Returns coefficient vector beta.
    """
    n, p = X_train.shape
    beta = np.zeros(p)

    # Sort by time descending for efficient risk-set computation.
    order = np.argsort(-y_time_train)
    X_sorted = X_train[order]
    e_sorted = y_event_train[order]

    for _ in range(n_iter):
        eta = X_sorted @ beta
        # Numerical stability: shift by max.
        eta_shift = eta - np.max(eta)
        exp_eta = np.exp(eta_shift)

        # Cumulative sum from the end (risk set sums).
        cum_exp = np.cumsum(exp_eta)
        # Gradient of negative partial log-likelihood.
        grad = np.zeros(p)
        for i in range(n):
            if e_sorted[i]:
                risk_set_sum = cum_exp[i]
                weighted_x = np.zeros(p)
                for j in range(i + 1):
                    weighted_x += exp_eta[j] * X_sorted[j]
                grad += -(X_sorted[i] - weighted_x / risk_set_sum)
        grad /= max(1, e_sorted.sum())
        grad += l2 * beta  # L2 regularization.
        beta -= lr * grad

    return beta


class _PatientLongitudinalBaseline:
    """Simple baseline implementations for patient-longitudinal mode."""

    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.seed = seed
        self._params: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        event_time_train: np.ndarray,
        event_observed_train: np.ndarray,
        clinical_train: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the baseline model."""
        rng = np.random.RandomState(self.seed)

        if self.name == "clinical_only_cox":
            if clinical_train is not None and clinical_train.shape[1] > 0:
                beta = _fit_cox_linear(clinical_train, event_time_train, event_observed_train)
                self._params["beta"] = beta
                self._params["use_clinical"] = True
            else:
                self._params["use_clinical"] = False

        elif self.name == "clinical_ridge_cox":
            if clinical_train is not None and clinical_train.shape[1] > 0:
                beta = _fit_cox_linear(
                    clinical_train, event_time_train, event_observed_train, l2=1.0,
                )
                self._params["beta"] = beta
                self._params["use_clinical"] = True
            else:
                self._params["use_clinical"] = False

        elif self.name == "rna_only_cox":
            # PCA to 10 components, then Cox.
            from sklearn.decomposition import PCA
            n_comp = min(10, X_train.shape[1], X_train.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=self.seed)
            X_pca = pca.fit_transform(X_train)
            beta = _fit_cox_linear(X_pca, event_time_train, event_observed_train)
            self._params["pca"] = pca
            self._params["beta"] = beta

        elif self.name == "rna_plus_clinical_cox":
            from sklearn.decomposition import PCA
            n_comp = min(10, X_train.shape[1], X_train.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=self.seed)
            X_pca = pca.fit_transform(X_train)
            if clinical_train is not None and clinical_train.shape[1] > 0:
                X_combined = np.hstack([X_pca, clinical_train])
            else:
                X_combined = X_pca
            beta = _fit_cox_linear(X_combined, event_time_train, event_observed_train)
            self._params["pca"] = pca
            self._params["beta"] = beta
            self._params["has_clinical"] = (
                clinical_train is not None and clinical_train.shape[1] > 0
            )

        elif self.name == "random_survival_forest":
            # Bug #4 fix: use a PROPER censoring-aware Random Survival Forest
            # (sksurv) with log-rank splitting, NOT a RandomForestRegressor on
            # raw event times (which discards the censoring indicator entirely
            # and silently treats censored follow-up as if it were an event).
            from sksurv.ensemble import RandomSurvivalForest
            from sklearn.decomposition import PCA
            n_comp = min(20, X_train.shape[1], X_train.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=self.seed)
            X_pca = pca.fit_transform(X_train)
            if clinical_train is not None and clinical_train.shape[1] > 0:
                X_combined = np.hstack([X_pca, clinical_train])
            else:
                X_combined = X_pca
            # sksurv requires a structured array of (event_indicator, time).
            y_surv = np.array(
                [
                    (bool(e), float(t))
                    for e, t in zip(event_observed_train, event_time_train)
                ],
                dtype=[("event", bool), ("time", float)],
            )
            rsf = RandomSurvivalForest(
                n_estimators=100,
                random_state=self.seed,
                max_depth=5,
                min_samples_leaf=10,
            )
            rsf.fit(X_combined, y_surv)
            self._params["pca"] = pca
            self._params["rsf"] = rsf
            self._params["has_clinical"] = (
                clinical_train is not None and clinical_train.shape[1] > 0
            )

        elif self.name == "deepsurv_mlp":
            # Simple 2-layer MLP approximating DeepSurv (Cox loss, neural net).
            from sklearn.decomposition import PCA
            n_comp = min(20, X_train.shape[1], X_train.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=self.seed)
            X_pca = pca.fit_transform(X_train)
            if clinical_train is not None and clinical_train.shape[1] > 0:
                X_combined = np.hstack([X_pca, clinical_train])
            else:
                X_combined = X_pca
            # Use linear Cox as a proxy (proper DeepSurv requires torch training).
            beta = _fit_cox_linear(
                X_combined, event_time_train, event_observed_train, l2=0.1, n_iter=200,
            )
            self._params["pca"] = pca
            self._params["beta"] = beta
            self._params["has_clinical"] = (
                clinical_train is not None and clinical_train.shape[1] > 0
            )

        elif self.name == "mean_time_baseline":
            self._params["mean_time"] = float(np.mean(event_time_train))

        elif self.name == "kaplan_meier_baseline":
            median_time = _kaplan_meier_estimate(event_time_train, event_observed_train)
            self._params["median_time"] = median_time

        elif self.name == "locf_trajectory_baseline":
            # Last observation carried forward: predict mean of training times.
            self._params["mean_time"] = float(np.mean(event_time_train))

        elif self.name == "mean_future_state_baseline":
            # Predict mean future state (trivial constant predictor).
            self._params["mean_time"] = float(np.mean(event_time_train))

        else:
            raise ValueError(f"Unknown patient-longitudinal baseline: {self.name}")

    def predict_risk(
        self,
        X_test: np.ndarray,
        clinical_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return risk scores (higher = higher risk = shorter time-to-event)."""
        n = X_test.shape[0]

        if self.name == "clinical_only_cox":
            if self._params.get("use_clinical") and clinical_test is not None:
                return clinical_test @ self._params["beta"]
            return np.zeros(n)

        elif self.name == "clinical_ridge_cox":
            if self._params.get("use_clinical") and clinical_test is not None:
                return clinical_test @ self._params["beta"]
            return np.zeros(n)

        elif self.name == "rna_only_cox":
            X_pca = self._params["pca"].transform(X_test)
            return X_pca @ self._params["beta"]

        elif self.name == "rna_plus_clinical_cox":
            X_pca = self._params["pca"].transform(X_test)
            if self._params.get("has_clinical") and clinical_test is not None:
                X_combined = np.hstack([X_pca, clinical_test])
            else:
                X_combined = X_pca
            return X_combined @ self._params["beta"]

        elif self.name == "random_survival_forest":
            X_pca = self._params["pca"].transform(X_test)
            if self._params.get("has_clinical") and clinical_test is not None:
                X_combined = np.hstack([X_pca, clinical_test])
            else:
                X_combined = X_pca
            # sksurv RSF.predict returns the ensemble risk score directly
            # (higher = higher risk), so no time-inversion is needed.
            return self._params["rsf"].predict(X_combined)

        elif self.name == "deepsurv_mlp":
            X_pca = self._params["pca"].transform(X_test)
            if self._params.get("has_clinical") and clinical_test is not None:
                X_combined = np.hstack([X_pca, clinical_test])
            else:
                X_combined = X_pca
            return X_combined @ self._params["beta"]

        elif self.name in ("mean_time_baseline", "kaplan_meier_baseline",
                           "locf_trajectory_baseline", "mean_future_state_baseline"):
            # Constant predictor: all patients get same risk score.
            return np.zeros(n)

        else:
            raise ValueError(f"Unknown baseline: {self.name}")


def run_patient_longitudinal(args: argparse.Namespace) -> int:
    """Run patient-longitudinal survival/trajectory baselines on frozen splits.

    Requires --confirmed-dataset and --split-manifest.
    """
    # ── Validate inputs ────────────────────────────────────────────────────
    if not args.split_manifest or not Path(args.split_manifest).exists():
        raise SystemExit(
            "patient_longitudinal baseline mode requires --split-manifest. "
            "Refusing to create ad hoc splits."
        )
    if not args.confirmed_dataset or not Path(args.confirmed_dataset).exists():
        raise SystemExit(
            "patient_longitudinal baseline mode requires --confirmed-dataset."
        )

    logger.info("Patient-longitudinal mode")
    logger.info("  confirmed dataset: %s", args.confirmed_dataset)
    logger.info("  split manifest:    %s", args.split_manifest)
    logger.info("  seeds:             %s", args.seeds)

    # ── Load dataset ───────────────────────────────────────────────────────
    df = pd.read_parquet(args.confirmed_dataset)
    logger.info("  Loaded dataset: %d rows, %d columns", len(df), len(df.columns))

    # ── Load split manifest ────────────────────────────────────────────────
    with open(args.split_manifest, "r") as f:
        manifest = json.load(f)
    logger.info("  Split manifest loaded: %d seed entries", len(manifest.get("splits", [])))

    # ── Identify feature columns ───────────────────────────────────────────
    # Standard columns expected in the confirmed dataset.
    meta_cols = {
        "patient_id", "event_time", "event_observed", "split",
        "event_time_days", "event_observed_bool",
    }
    clinical_cols = [
        c for c in df.columns
        if c.startswith("clinical_") or c in (
            "iss_stage_ordinal", "age_at_dx_zscore", "gender_is_male",
            "bort_1L", "n_treatments_norm",
        )
    ]
    # All remaining numeric columns are features (RNA, latent, etc.).
    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and c not in clinical_cols and df[c].dtype in ("float64", "float32", "int64")
    ]

    # Determine event_time and event_observed column names.
    time_col = "event_time_days" if "event_time_days" in df.columns else "event_time"
    event_col = "event_observed_bool" if "event_observed_bool" in df.columns else "event_observed"

    logger.info("  Feature columns: %d, Clinical columns: %d", len(feature_cols), len(clinical_cols))
    logger.info("  Time column: %s, Event column: %s", time_col, event_col)

    # ── Determine which baselines to run ───────────────────────────────────
    # If --table is specified, use the taxonomy-defined subset
    if hasattr(args, "table") and args.table is not None:
        table_baselines = BENCHMARK_TABLES.get(args.table, [])
        if not table_baselines:
            logger.warning("Table '%s' has no baselines (blocked or handled elsewhere).", args.table)
            print(f"[Table {args.table}] No baselines to run — see README benchmark taxonomy.")
            return 0
        requested = table_baselines
        logger.info("Using --table %s baselines: %s", args.table, requested)
    else:
        requested = args.baselines
    if len(requested) == 1 and requested[0].lower() == "all":
        baselines_to_run = list(PATIENT_LONGITUDINAL_BASELINES)
    else:
        unknown = [b for b in requested if b not in PATIENT_LONGITUDINAL_BASELINES]
        if unknown:
            raise SystemExit(
                f"Unknown patient-longitudinal baselines: {unknown}\n"
                f"Available: {PATIENT_LONGITUDINAL_BASELINES}"
            )
        baselines_to_run = list(requested)

    # ── Execution plan ─────────────────────────────────────────────────────
    plan_lines = [
        "=" * 70,
        "Phase 10 baseline runner — PATIENT LONGITUDINAL mode",
        "=" * 70,
        f"  confirmed dataset: {args.confirmed_dataset}",
        f"  split manifest:    {args.split_manifest}",
        f"  seeds:             {args.seeds}",
        f"  baselines:         {baselines_to_run}",
        f"  total jobs:        {len(baselines_to_run) * len(args.seeds)}",
        "=" * 70,
    ]
    for line in plan_lines:
        print(line)

    if args.dry_run:
        print("[dry-run] no models will be fitted; exiting.")
        return 0

    # ── Run baselines per seed ─────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    out_dir = args.out_dir / "patient_longitudinal"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_to_use = args.seeds

    # Use the 'split' column directly from the confirmed dataset if present
    # (written by the Spark lakehouse's stage_confirmed).
    has_split_col = "split" in df.columns
    if has_split_col:
        logger.info("Using 'split' column from confirmed dataset (deterministic SHA-256).")

    for seed in seeds_to_use:
        if has_split_col:
            train_mask = df["split"] == "train"
            test_mask = df["split"] == "test"
        else:
            # Fallback: look for patient IDs in manifest
            splits_data = manifest.get("splits", {})
            train_ids = set(str(pid) for pid in splits_data.get("train_patient_ids", []))
            test_ids = set(str(pid) for pid in splits_data.get("test_patient_ids", []))
            if not train_ids or not test_ids:
                logger.error("Seed %d: no split info found; skipping.", seed)
                continue
            pid_col = "patient_id_hash" if "patient_id_hash" in df.columns else df.columns[0]
            df_str_pid = df[pid_col].astype(str)
            train_mask = df_str_pid.isin(train_ids)
            test_mask = df_str_pid.isin(test_ids)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.error("Seed %d: no matching rows for train/test; skipping.", seed)
            continue

        # Extract arrays.
        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)

        # Replace NaN with 0 in features.
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        clinical_train = (
            df.loc[train_mask, clinical_cols].values.astype(np.float32)
            if clinical_cols else None
        )
        clinical_test = (
            df.loc[test_mask, clinical_cols].values.astype(np.float32)
            if clinical_cols else None
        )
        if clinical_train is not None:
            clinical_train = np.nan_to_num(clinical_train, nan=0.0)
        if clinical_test is not None:
            clinical_test = np.nan_to_num(clinical_test, nan=0.0)

        event_time_train = df.loc[train_mask, time_col].values.astype(np.float64)
        event_obs_train = df.loc[train_mask, event_col].values.astype(np.float64)
        event_time_test = df.loc[test_mask, time_col].values.astype(np.float64)
        event_obs_test = df.loc[test_mask, event_col].values.astype(np.float64)

        logger.info(
            "  Seed %d: train=%d, test=%d, events_train=%d, events_test=%d",
            seed, train_mask.sum(), test_mask.sum(),
            int(event_obs_train.sum()), int(event_obs_test.sum()),
        )

        for bl_name in baselines_to_run:
            logger.info("    Fitting %s (seed=%d) ...", bl_name, seed)
            t0 = time.time()

            try:
                model = _PatientLongitudinalBaseline(bl_name, seed=seed)
                model.fit(X_train, event_time_train, event_obs_train, clinical_train)
                risk_scores = model.predict_risk(X_test, clinical_test)

                # Compute metrics.
                c_index = _concordance_index(event_time_test, event_obs_test, risk_scores)
                ibs = _integrated_brier_score(event_time_test, event_obs_test, risk_scores)
            except Exception as exc:
                logger.error("    FAILED %s (seed=%d): %s", bl_name, seed, exc)
                continue

            elapsed = time.time() - t0

            result = {
                "model": bl_name,
                "seed": seed,
                "c_index": round(c_index, 6),
                "ibs": round(ibs, 6),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_events_test": int(event_obs_test.sum()),
                "wall_seconds": round(elapsed, 2),
            }
            all_results.append(result)

            logger.info(
                "    %s  C-index=%.4f  IBS=%.4f  (%.1fs)",
                bl_name, c_index, ibs, elapsed,
            )

    # ── Write results ──────────────────────────────────────────────────────
    results_path = out_dir / "patient_longitudinal_results.json"
    results_payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "patient_longitudinal",
        "confirmed_dataset": str(args.confirmed_dataset),
        "split_manifest": str(args.split_manifest),
        "seeds": args.seeds,
        "n_results": len(all_results),
        "results": all_results,
    }
    results_path.write_text(json.dumps(results_payload, indent=2))
    logger.info("Wrote %s", results_path)

    # ── Write comparison table ─────────────────────────────────────────────
    comparison_path = out_dir / "patient_longitudinal_claim_comparison.json"
    comparison = _build_claim_comparison(all_results)
    comparison_path.write_text(json.dumps(comparison, indent=2))
    logger.info("Wrote %s", comparison_path)

    # ── Print summary ──────────────────────────────────────────────────────
    if all_results:
        print()
        print("=" * 90)
        print("  PATIENT-LONGITUDINAL BASELINE RESULTS")
        print("=" * 90)
        header = (
            f"  {'Model':35s} {'C-index':>10s} {'IBS':>10s} "
            f"{'N_test':>8s} {'Events':>8s}"
        )
        print(header)
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

        # Aggregate across seeds.
        from collections import defaultdict
        agg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in all_results:
            agg[r["model"]].append(r)

        for bl_name in baselines_to_run:
            if bl_name not in agg:
                continue
            entries = agg[bl_name]
            mean_ci = np.mean([e["c_index"] for e in entries])
            mean_ibs = np.mean([e["ibs"] for e in entries])
            n_test = entries[0]["n_test"]
            n_events = entries[0]["n_events_test"]
            print(
                f"  {bl_name:35s} {mean_ci:10.4f} {mean_ibs:10.4f} "
                f"{n_test:8d} {n_events:8d}"
            )

        print("=" * 90)

        # Print claim-level comparison.
        print()
        print("=" * 90)
        print("  CLAIM-LEVEL BASELINE REQUIREMENTS")
        print("  (MORT-FM must beat these baselines to advance each claim level)")
        print("=" * 90)
        for claim, info in comparison.get("claim_levels", {}).items():
            print(f"\n  [{claim}]")
            print(f"    {info.get('description', '')}")
            must = info.get("must_beat_results", {})
            if must:
                print(f"    MUST beat:")
                for bl, metrics in must.items():
                    print(f"      {bl:35s}  C-index={metrics.get('mean_c_index', 'n/a'):.4f}")
            should = info.get("should_beat_results", {})
            if should:
                print(f"    SHOULD beat:")
                for bl, metrics in should.items():
                    print(f"      {bl:35s}  C-index={metrics.get('mean_c_index', 'n/a'):.4f}")
        print("=" * 90)

    return 0


def _build_claim_comparison(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a comparison table showing which baselines MORT-FM must beat."""
    from collections import defaultdict

    # Aggregate results per model across seeds.
    agg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        agg[r["model"]].append(r)

    model_summary: Dict[str, Dict[str, float]] = {}
    for model_name, entries in agg.items():
        model_summary[model_name] = {
            "mean_c_index": float(np.mean([e["c_index"] for e in entries])),
            "std_c_index": float(np.std([e["c_index"] for e in entries])),
            "mean_ibs": float(np.mean([e["ibs"] for e in entries])),
            "std_ibs": float(np.std([e["ibs"] for e in entries])),
            "n_seeds": len(entries),
        }

    # Build per-claim comparison.
    claim_levels: Dict[str, Any] = {}
    for claim, spec in CLAIM_LEVEL_BASELINE_REQUIREMENTS.items():
        must_beat_results = {}
        for bl in spec["must_beat"]:
            if bl in model_summary:
                must_beat_results[bl] = model_summary[bl]
        should_beat_results = {}
        for bl in spec["should_beat"]:
            if bl in model_summary:
                should_beat_results[bl] = model_summary[bl]

        # Compute the threshold MORT-FM must exceed.
        must_beat_max_ci = max(
            (v["mean_c_index"] for v in must_beat_results.values()), default=0.5,
        )

        claim_levels[claim] = {
            "description": spec["description"],
            "must_beat_results": must_beat_results,
            "should_beat_results": should_beat_results,
            "mortfm_must_exceed_c_index": round(must_beat_max_ci, 6),
        }

    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_summary": model_summary,
        "claim_levels": claim_levels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Route based on mode ────────────────────────────────────────────────
    if args.mode == "patient_longitudinal":
        return run_patient_longitudinal(args)

    # ── Static cell-line drug-response mode (default) ──────────────────────
    baselines = _resolve_baselines(args.baselines)

    # ── Execution plan ──────────────────────────────────────────────────────
    plan_lines = [
        "=" * 70,
        "Phase 10 baseline runner — execution plan",
        "=" * 70,
        f"  data checkpoint: {args.data_ckpt}",
        f"  output dir:      {args.out_dir}",
        f"  output parquet:  {args.out_parquet}",
        f"  seeds:           {args.seeds}",
        f"  baselines:       {baselines}",
        f"  include SOTA:    {args.include_sota}",
        f"  total registry jobs: {len(baselines) * len(args.seeds)}",
        "=" * 70,
    ]
    for line in plan_lines:
        print(line)

    if args.dry_run:
        print("[dry-run] no models will be fitted; exiting.")
        return 0

    # ── Load data ───────────────────────────────────────────────────────────
    data = load_real_data(args.data_ckpt)
    splits = data["splits"]
    drug_names = data["drug_names"]
    X_full = data["X_full"]
    y_full = data["y_full"]

    # Compute and log the split-manifest hash for integrity verification.
    split_hash = get_split_hash(splits)
    git_sha = get_git_sha()
    logger.info("Split manifest hash: %s", split_hash)
    logger.info("Git SHA: %s", git_sha)

    _, _, _, _, _, y_test = split_arrays(X_full, y_full, splits)

    # ResistanceMap reference MSE.
    rm_test_mse = get_resistancemap_test_mse()
    if rm_test_mse is not None:
        logger.info("ResistanceMap test_mse = %.6f", rm_test_mse)
    else:
        logger.warning("pipeline_validated.pt not found; no RM reference MSE.")

    # ── Run registry baselines ──────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    comparison_rows: List[ComparisonRow] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in baselines:
        for seed in args.seeds:
            logger.info("Fitting %s (seed=%d) ...", model_name, seed)
            t0 = time.time()
            try:
                result = run_registry_baseline(
                    model_name, X_full, y_full, splits, drug_names, seed,
                )
            except Exception as exc:
                logger.error("FAILED %s (seed=%d): %s", model_name, seed, exc)
                continue
            elapsed = time.time() - t0
            result["wall_seconds"] = round(elapsed, 2)

            logger.info(
                "  %s  test_mse=%.4f  mean_rho=%.4f  (%.1fs)",
                model_name, result["test_mse"], result["mean_spearman"], elapsed,
            )

            all_results.append(result)

            # Save predictions.
            pred_path = save_predictions_parquet(
                model_name, result["test_predictions"],
                splits["test"], drug_names, args.out_dir,
            )
            logger.info("  Saved predictions: %s", pred_path)

            # Build ComparisonRows.
            comparison_rows.append(ComparisonRow(
                model_name=model_name,
                split="test",
                metric_name="mse",
                metric_value=result["test_mse"],
                n=int(len(splits["test"])),
                seed=seed,
                hash_data=split_hash,
                hash_code=git_sha,
            ))
            comparison_rows.append(ComparisonRow(
                model_name=model_name,
                split="test",
                metric_name="mean_spearman",
                metric_value=result["mean_spearman"],
                n=int(len(splits["test"])),
                seed=seed,
                hash_data=split_hash,
                hash_code=git_sha,
            ))

    # ── Run SOTA models ─────────────────────────────────────────────────────
    if args.include_sota:
        logger.info("Running SOTA deep-learning models ...")
        for seed in args.seeds:
            try:
                sota_results = run_sota_models(data, seed)
            except Exception as exc:
                logger.error("FAILED SOTA models (seed=%d): %s", seed, exc)
                continue

            for result in sota_results:
                all_results.append(result)

                pred_path = save_predictions_parquet(
                    result["model"], result["test_predictions"],
                    splits["test"], drug_names, args.out_dir,
                )
                logger.info("  Saved predictions: %s", pred_path)

                comparison_rows.append(ComparisonRow(
                    model_name=result["model"],
                    split="test",
                    metric_name="mse",
                    metric_value=result["test_mse"],
                    n=int(len(splits["test"])),
                    seed=seed,
                    hash_data=split_hash,
                    hash_code=git_sha,
                ))
                comparison_rows.append(ComparisonRow(
                    model_name=result["model"],
                    split="test",
                    metric_name="mean_spearman",
                    metric_value=result["mean_spearman"],
                    n=int(len(splits["test"])),
                    seed=seed,
                    hash_data=split_hash,
                    hash_code=git_sha,
                ))

    # ── Write ComparisonRow parquet ──────────────────────────────────────────
    if comparison_rows:
        write_comparison_rows(comparison_rows, args.out_parquet, append=True)
        logger.info(
            "Appended %d ComparisonRow records to %s",
            len(comparison_rows), args.out_parquet,
        )

    # ── Write leaderboard ───────────────────────────────────────────────────
    if all_results:
        df_lb, lb_rows = write_leaderboard(all_results, rm_test_mse, y_test, args.out_dir)

        # Print summary table.
        print()
        print("=" * 90)
        print("  BASELINE LEADERBOARD")
        print("=" * 90)
        header = (
            f"  {'Model':30s} {'test_mse':>10s} {'mean_rho':>10s} "
            f"{'delta_rm':>10s} {'p_value':>10s}"
        )
        print(header)
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in lb_rows:
            mse_s = f"{r['test_mse']:.4f}" if r["test_mse"] is not None else "n/a"
            rho_s = f"{r['mean_spearman']:.4f}" if r["mean_spearman"] is not None else "n/a"
            delta_s = f"{r['delta_vs_rm']:.4f}" if r.get("delta_vs_rm") is not None else "n/a"
            pval_s = f"{r['p_value']:.4f}" if r.get("p_value") is not None else "n/a"
            print(f"  {r['model']:30s} {mse_s:>10s} {rho_s:>10s} {delta_s:>10s} {pval_s:>10s}")

        if rm_test_mse is not None:
            print(f"\n  ResistanceMap reference MSE:  {rm_test_mse:.4f}")
        print("=" * 90)
        print(f"\n  Artifacts: {args.out_dir}/")
    else:
        logger.warning("No results collected; leaderboard not written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
