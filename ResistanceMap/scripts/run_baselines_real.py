#!/usr/bin/env python3
"""Real-data regression baselines vs ResistanceMap.

Loads checkpoints/data_ready.pt (built by the agentic DAG from real
proteomics + epigenomics + drug-sensitivity data) and trains a panel of
regression baselines on the SAME train/val/test splits ResistanceMap used.

Drug-sensitivity is ~46% NaN (sparse clinical drug screens). To match
ResistanceMap's evaluation (resistancemap/main.py:1440-1443), each baseline
is trained per-drug on the observed (non-NaN) train rows for that drug, and
test_mse is computed over observed test cells only — same denominator
ResistanceMap uses.

Output: paper/tables/baseline_comparison.{json,md}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
DATA_CKPT = REPO / "checkpoints" / "data_ready.pt"
RM_CKPT = REPO / "checkpoints" / "pipeline_validated.pt"
OUT_DIR = REPO / "paper" / "tables"


def load_real_data():
    c = torch.load(DATA_CKPT, map_location="cpu", weights_only=False)
    ds = c["dataset"]
    splits = c["splits"]
    X_full = torch.cat([ds.proteomics, ds.epigenomics], dim=1).numpy().astype(np.float32)
    y_full = ds.drug_sensitivity.numpy().astype(np.float32)
    drug_names = list(ds.drug_names)
    splits_np = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}
    return X_full, y_full, splits_np, drug_names


def split_arrays(X, y, splits):
    return (X[splits["train"]], y[splits["train"]],
            X[splits["val"]],   y[splits["val"]],
            X[splits["test"]],  y[splits["test"]])


def masked_mse_per_drug(model_factory, Xtr, ytr, Xva, yva, Xte, yte):
    """Train one model per drug on observed cells; pool squared residuals."""
    n_drugs = ytr.shape[1]
    test_residuals = []
    val_residuals = []
    for d in range(n_drugs):
        tr_mask = ~np.isnan(ytr[:, d])
        va_mask = ~np.isnan(yva[:, d])
        te_mask = ~np.isnan(yte[:, d])
        if tr_mask.sum() < 5:
            continue
        m = model_factory()
        m.fit(Xtr[tr_mask], ytr[tr_mask, d])
        if te_mask.sum() > 0:
            pred_te = m.predict(Xte[te_mask])
            test_residuals.append(pred_te - yte[te_mask, d])
        if va_mask.sum() > 0:
            pred_va = m.predict(Xva[va_mask])
            val_residuals.append(pred_va - yva[va_mask, d])
    test_mse = float(np.mean(np.concatenate(test_residuals) ** 2)) if test_residuals else float("nan")
    val_mse = float(np.mean(np.concatenate(val_residuals) ** 2)) if val_residuals else float("nan")
    return test_mse, val_mse


def baseline_global_mean(ytr, yva, yte):
    """Predict the per-drug train mean for every (sample, drug) cell."""
    drug_means = np.nanmean(ytr, axis=0)  # (n_drugs,)
    pred_te = np.broadcast_to(drug_means, yte.shape)
    pred_va = np.broadcast_to(drug_means, yva.shape)
    te_mask = ~np.isnan(yte)
    va_mask = ~np.isnan(yva)
    test_mse = float(np.mean((pred_te[te_mask] - yte[te_mask]) ** 2))
    val_mse = float(np.mean((pred_va[va_mask] - yva[va_mask]) ** 2))
    return test_mse, val_mse


def baseline_global_zero(ytr, yva, yte):
    """Predict zero everywhere (drug sensitivity is z-scored ⇒ population mean ≈ 0)."""
    te_mask = ~np.isnan(yte); va_mask = ~np.isnan(yva)
    return float(np.mean(yte[te_mask] ** 2)), float(np.mean(yva[va_mask] ** 2))


def get_resistancemap_test_mse():
    if not RM_CKPT.exists():
        return None
    c = torch.load(RM_CKPT, map_location="cpu", weights_only=False)
    return float(c["metrics"]["test_mse"])


def run_panel(Xtr, ytr, Xva, yva, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, ElasticNet, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    pca = PCA(n_components=min(256, Xtr_s.shape[0] - 1), random_state=42)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xva_p = pca.transform(Xva_s)
    Xte_p = pca.transform(Xte_s)
    print("[panel] features=%d  PCA=%d  n_train=%d  n_val=%d  n_test=%d  n_drugs=%d"
          % (Xtr.shape[1], Xtr_p.shape[1], Xtr.shape[0], Xva.shape[0], Xte.shape[0], ytr.shape[1]))

    panel = []

    def record(name, t0, test_mse, val_mse, notes=""):
        panel.append({"model": name, "test_mse": float(test_mse),
                      "val_mse": float(val_mse) if val_mse is not None else None,
                      "wall_seconds": round(time.time() - t0, 2), "notes": notes})
        print(f"  {name:36s} test_mse={test_mse:.4f}  val_mse={val_mse:.4f}  ({panel[-1]['wall_seconds']}s)")

    t0 = time.time(); te, va = baseline_global_zero(ytr, yva, yte)
    record("Zero (z-scored target floor)", t0, te, va, "predicts 0 everywhere")

    t0 = time.time(); te, va = baseline_global_mean(ytr, yva, yte)
    record("PerDrugTrainMean", t0, te, va, "predicts per-drug train mean")

    t0 = time.time()
    te, va = masked_mse_per_drug(lambda: Ridge(alpha=1.0, random_state=42), Xtr_s, ytr, Xva_s, yva, Xte_s, yte)
    record("Ridge (full proteomics+epi)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(lambda: Ridge(alpha=1.0, random_state=42), Xtr_p, ytr, Xva_p, yva, Xte_p, yte)
    record("Ridge (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
                                 Xtr_p, ytr, Xva_p, yva, Xte_p, yte)
    record("ElasticNet (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(lambda: Lasso(alpha=0.05, max_iter=5000, random_state=42),
                                 Xtr_p, ytr, Xva_p, yva, Xte_p, yte)
    record("Lasso (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, n_jobs=-1, random_state=42),
        Xtr_p, ytr, Xva_p, yva, Xte_p, yte,
    )
    record("RandomForest (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(
        lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
        Xtr_p, ytr, Xva_p, yva, Xte_p, yte,
    )
    record("GradientBoosting (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(
        lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             n_jobs=-1, random_state=42, tree_method="hist", verbosity=0),
        Xtr_p, ytr, Xva_p, yva, Xte_p, yte,
    )
    record("XGBoost (PCA-256)", t0, te, va)

    t0 = time.time()
    te, va = masked_mse_per_drug(
        lambda: MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                             learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                             validation_fraction=0.15, random_state=42),
        Xtr_p, ytr, Xva_p, yva, Xte_p, yte,
    )
    record("MLP-256-128 (PCA-256)", t0, te, va)

    return panel


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] %s" % DATA_CKPT)
    X, y, splits, drug_names = load_real_data()
    Xtr, ytr, Xva, yva, Xte, yte = split_arrays(X, y, splits)

    n_train_obs = int((~np.isnan(ytr)).sum())
    n_test_obs = int((~np.isnan(yte)).sum())
    print(f"[data] observed cells: train={n_train_obs}/{ytr.size}  test={n_test_obs}/{yte.size}")

    rm_mse = get_resistancemap_test_mse()
    print(f"[ResistanceMap] test_mse = {rm_mse}")

    panel = run_panel(Xtr, ytr, Xva, yva, Xte, yte)
    panel.append({"model": "ResistanceMap (10-agent DAG)", "test_mse": rm_mse,
                  "val_mse": None, "wall_seconds": None,
                  "notes": "from checkpoints/pipeline_validated.pt"})
    panel.sort(key=lambda r: (r["test_mse"] if r["test_mse"] is not None else float("inf")))

    out_json = OUT_DIR / "baseline_comparison.json"
    out_md = OUT_DIR / "baseline_comparison.md"

    out_json.write_text(json.dumps({
        "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]), "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]), "n_drugs": int(ytr.shape[1]),
        "n_train_observed_cells": n_train_obs, "n_test_observed_cells": n_test_obs,
        "drugs": drug_names, "results": panel,
    }, indent=2))

    rows = ["| Rank | Model | test_mse | val_mse | wall (s) | notes |",
            "|---|---|---|---|---|---|"]
    for i, r in enumerate(panel, 1):
        rows.append("| {} | {} | {} | {} | {} | {} |".format(
            i, r["model"],
            f"{r['test_mse']:.4f}" if r["test_mse"] is not None else "n/a",
            f"{r['val_mse']:.4f}" if r["val_mse"] is not None else "n/a",
            r["wall_seconds"] if r["wall_seconds"] is not None else "n/a",
            r["notes"]))
    out_md.write_text(
        "# Real-Data Baseline Comparison\n\n"
        f"- Splits: train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]}\n"
        f"- Features: {Xtr.shape[1]} (proteomics 19177 + epigenomics 42)\n"
        f"- Targets: {ytr.shape[1]}-dim sparse drug-sensitivity ({100*np.isnan(yte).mean():.1f}% NaN in test)\n"
        f"- Drugs: {', '.join(drug_names)}\n"
        f"- Evaluation: NaN-masked MSE pooled across observed (sample, drug) cells (matches resistancemap/main.py:1440-1443)\n\n"
        + "\n".join(rows) + "\n")

    print(f"\n[done] {out_json}\n[done] {out_md}")


if __name__ == "__main__":
    main()
