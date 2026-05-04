"""ElasticNet baseline for ResistanceMap drug-response prediction.

Loads the prepared MultiOmicsDataset checkpoint, builds a per-sample feature
matrix from proteomics + epigenomics + CRISPR (NaN->0 imputed, with a
``has_crispr`` indicator column), and fits a per-drug ``ElasticNetCV`` on the
training split. Reports per-drug n_train / n_test / MSE / Spearman rho on the
test split, plus a predict-mean baseline for context.

Notes on feature handling:
    - Proteomics has ~19k columns. With ~600 train samples ElasticNetCV across
      4 l1_ratios x 5 folds is the dominant cost. To keep total runtime under
      a few minutes on CPU we (a) drop zero-variance columns globally and
      (b) for each drug keep the top-K proteomics columns by absolute Pearson
      correlation with that drug's training labels (default K=2000). The
      epigenomics (42 cols) and CRISPR (18,531 cols + 1 indicator) blocks are
      kept in full -- they are either small or cheap relative to proteomics.
    - The NaN->0 imputation for CRISPR is applied AFTER concatenation, with a
      single binary "has_crispr" indicator appended once (not per-gene).
    - The drug_sensitivity tensor is z-scored globally upstream; this is a
      known leak but we accept it because we only need a relative floor.

Outputs:
    - JSON: paper/v8_artifacts/baselines/elasticnet_results.json
    - Human-readable per-drug summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "data_ready.pt"
DEFAULT_OUT = REPO_ROOT / "paper" / "v8_artifacts" / "baselines" / "elasticnet_results.json"


def _load_blocks(ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    ds = ckpt["dataset"]
    splits = ckpt["splits"]
    proteomics = ds.proteomics.numpy().astype(np.float32, copy=False)
    epigenomics = ds.epigenomics.numpy().astype(np.float32, copy=False)
    crispr = ds.crispr_effect.numpy().astype(np.float32, copy=False)
    drug_y = ds.drug_sensitivity.numpy().astype(np.float32, copy=False)
    drug_names = list(ds.drug_names)
    return {
        "proteomics": proteomics,
        "epigenomics": epigenomics,
        "crispr": crispr,
        "drug_y": drug_y,
        "drug_names": drug_names,
        "splits": {k: np.asarray(v) for k, v in splits.items()},
        "sample_ids": list(ds.sample_ids),
    }


def _build_static_features(blocks: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the parts of the feature matrix that are drug-agnostic.

    Returns:
        epi: (N, 42) epigenomics
        crispr_imputed: (N, n_genes) with NaN->0
        has_crispr: (N, 1) indicator (1.0 if any CRISPR value is non-NaN for
            the row, else 0.0)
    """
    epi = blocks["epigenomics"]
    cr = blocks["crispr"]
    has_crispr = (~np.isnan(cr).all(axis=1)).astype(np.float32).reshape(-1, 1)
    cr_imp = np.where(np.isnan(cr), 0.0, cr).astype(np.float32, copy=False)
    return epi, cr_imp, has_crispr


def _topk_by_abs_corr(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k columns of X_train by |Pearson r| with y_train.

    Uses a fast vectorised computation; columns with zero variance get r=0.
    """
    if X_train.shape[1] <= k:
        return np.arange(X_train.shape[1])
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    yc = y_train - y_train.mean()
    x_std = np.sqrt((Xc ** 2).sum(axis=0))
    y_std = np.sqrt((yc ** 2).sum())
    denom = x_std * y_std
    num = Xc.T @ yc
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    abs_r = np.abs(r)
    # argpartition for speed
    idx = np.argpartition(-abs_r, k - 1)[:k]
    # sort the chosen idx for determinism
    return np.sort(idx)


def run(
    ckpt_path: Path = DEFAULT_CKPT,
    out_path: Path = DEFAULT_OUT,
    proteomics_topk: int = 2000,
    cv: int = 5,
    max_iter: int = 2000,
    tol: float = 1e-3,
    random_state: int = 0,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[elasticnet] Loading {ckpt_path}", flush=True)
    blocks = _load_blocks(ckpt_path)
    drug_names = blocks["drug_names"]
    drug_y = blocks["drug_y"]
    splits = blocks["splits"]

    train_idx = splits["train"].astype(int)
    test_idx = splits["test"].astype(int)

    # Pre-compute static feature blocks
    epi, cr_imp, has_crispr = _build_static_features(blocks)
    prot = blocks["proteomics"]

    # Drop zero-variance proteomics columns globally (computed on TRAIN to
    # avoid using test info even for filtering).
    prot_train = prot[train_idx]
    prot_var = prot_train.var(axis=0)
    keep_mask = prot_var > 0
    prot_kept = prot[:, keep_mask]
    print(
        f"[elasticnet] proteomics cols: {prot.shape[1]} -> {prot_kept.shape[1]} "
        f"after zero-variance filter (train-only var)",
        flush=True,
    )

    l1_ratios = [0.1, 0.5, 0.7, 0.9]
    results: dict[str, Any] = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "proteomics_topk": proteomics_topk,
            "cv": cv,
            "l1_ratios": l1_ratios,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state,
            "feature_blocks": [
                f"proteomics_top{proteomics_topk}_per_drug_by_abs_corr_on_train",
                "epigenomics_full",
                "crispr_full_NaN_to_0",
                "has_crispr_indicator",
            ],
            "n_train_total": int(len(train_idx)),
            "n_test_total": int(len(test_idx)),
            "n_features_pre_topk": int(
                prot_kept.shape[1] + epi.shape[1] + cr_imp.shape[1] + 1
            ),
            "n_features_used_per_drug": int(
                min(proteomics_topk, prot_kept.shape[1])
                + epi.shape[1]
                + cr_imp.shape[1]
                + 1
            ),
        },
        "per_drug": [],
    }

    overall_test_se: list[float] = []
    overall_test_se_meanpred: list[float] = []

    for j, drug in enumerate(drug_names):
        y = drug_y[:, j]

        train_obs = train_idx[~np.isnan(y[train_idx])]
        test_obs = test_idx[~np.isnan(y[test_idx])]
        n_train = int(len(train_obs))
        n_test = int(len(test_obs))

        if n_train < 10 or n_test < 2:
            print(
                f"[elasticnet] {drug}: skipping (n_train={n_train}, n_test={n_test})",
                flush=True,
            )
            results["per_drug"].append({
                "drug": drug,
                "n_train": n_train,
                "n_test": n_test,
                "skipped": True,
            })
            continue

        y_train = y[train_obs].astype(np.float32)
        y_test = y[test_obs].astype(np.float32)

        # Per-drug top-K proteomics filter using TRAIN labels only.
        top_idx = _topk_by_abs_corr(prot_kept[train_obs], y_train, proteomics_topk)
        prot_sel = prot_kept[:, top_idx]

        # Concatenate features. Order: proteomics_topk | epi | crispr | has_crispr
        X_train = np.concatenate(
            [prot_sel[train_obs], epi[train_obs], cr_imp[train_obs], has_crispr[train_obs]],
            axis=1,
        ).astype(np.float32, copy=False)
        X_test = np.concatenate(
            [prot_sel[test_obs], epi[test_obs], cr_imp[test_obs], has_crispr[test_obs]],
            axis=1,
        ).astype(np.float32, copy=False)

        t_drug = time.time()
        model = ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=cv,
            n_jobs=-1,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            selection="random",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = float(np.mean((y_pred - y_test) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_test)))
        if np.std(y_pred) < 1e-12 or np.std(y_test) < 1e-12:
            sp = float("nan")
        else:
            sp = float(spearmanr(y_pred, y_test).statistic)

        # Predict-mean baseline (mean of TRAIN labels).
        mean_pred = float(np.mean(y_train))
        mse_meanpred = float(np.mean((mean_pred - y_test) ** 2))

        overall_test_se.extend(((y_pred - y_test) ** 2).tolist())
        overall_test_se_meanpred.extend(((mean_pred - y_test) ** 2).tolist())

        elapsed = time.time() - t_drug
        print(
            f"[elasticnet] {drug:>16s}  n_tr={n_train:>3d}  n_te={n_test:>3d}  "
            f"mse={mse:.4f}  mae={mae:.4f}  spearman={sp:+.3f}  "
            f"mse_meanpred={mse_meanpred:.4f}  alpha={model.alpha_:.4g}  "
            f"l1={model.l1_ratio_:.2f}  ({elapsed:.1f}s)",
            flush=True,
        )

        results["per_drug"].append({
            "drug": drug,
            "n_train": n_train,
            "n_test": n_test,
            "mse": mse,
            "mae": mae,
            "spearman": sp,
            "mse_predict_mean": mse_meanpred,
            "alpha": float(model.alpha_),
            "l1_ratio": float(model.l1_ratio_),
            "n_nonzero_coefs": int(np.sum(np.abs(model.coef_) > 0)),
            "fit_seconds": elapsed,
            "skipped": False,
        })

    # Pooled ("micro") test MSE across drugs (matches how RM reports test_mse).
    if overall_test_se:
        results["pooled_test_mse"] = float(np.mean(overall_test_se))
        results["pooled_test_mse_predict_mean"] = float(np.mean(overall_test_se_meanpred))
        results["pooled_n_test"] = int(len(overall_test_se))

    results["total_seconds"] = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    print("\n=== ElasticNet baseline summary ===", flush=True)
    print(
        f"feature blocks: {results['config']['feature_blocks']}",
        flush=True,
    )
    print(
        f"n_features used per drug: {results['config']['n_features_used_per_drug']}",
        flush=True,
    )
    rows = [r for r in results["per_drug"] if not r.get("skipped")]
    df = pd.DataFrame(rows)[
        ["drug", "n_train", "n_test", "mse", "mae", "spearman", "mse_predict_mean"]
    ]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    if "pooled_test_mse" in results:
        print(
            f"\npooled test MSE (ElasticNet)    = {results['pooled_test_mse']:.4f}  "
            f"(n={results['pooled_n_test']})",
            flush=True,
        )
        print(
            f"pooled test MSE (predict-mean)  = {results['pooled_test_mse_predict_mean']:.4f}",
            flush=True,
        )
    print(f"\nWrote: {out_path}", flush=True)
    print(f"Total wall time: {results['total_seconds']:.1f}s", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--proteomics-topk", type=int, default=2000)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args(argv)
    run(
        ckpt_path=args.ckpt,
        out_path=args.out,
        proteomics_topk=args.proteomics_topk,
        cv=args.cv,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
