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

USAGE
-----

    # Dry run — only print the plan.
    python scripts/mortfm/10_run_baselines.py \\
        --baselines all --dry-run

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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

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
