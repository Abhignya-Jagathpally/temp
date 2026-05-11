#!/usr/bin/env python3
"""v12 paired Wilcoxon: ResistanceMap vs Late-Fusion (avg-Ridge) per-drug MSE.

Honest design: at the existing seed=42 split, recompute Late Fusion per-drug
MSE on the same test cells used by ResistanceMap. Run a one-sided paired
Wilcoxon signed-rank test across the 11 drugs and report Holm-Bonferroni-
corrected per-drug paired-sign tests as a secondary breakdown.

H0: median(MSE_RM - MSE_LF) >= 0    (ResistanceMap is not better than LF)
H1: median(MSE_RM - MSE_LF) <  0    (ResistanceMap is better; pre-registered)

Predicted outcome (stated honestly, before running):
    The pooled MSE gap on disk is 0.0078 (RM 2.3731 vs LF 2.3653 — LF wins
    on aggregate). Per-drug paired differences are expected to be small and
    mixed-direction with one Panobinostat outlier whose magnitude Wilcoxon
    ignores. Expected one-sided p in [0.15, 0.85]; the test will likely NOT
    reject the null.

Multi-seed (`--multi-seed`) retrains ResistanceMap at 5 seeds and is not
enabled by default because each retrain costs ~30 min CPU.

Usage:
    python -m scripts.v12.paired_wilcoxon_rm_vs_latefusion \
        --out paper/v8_artifacts/v12_sprint1/paired_wilcoxon_latefusion.json

Files this script reads (must already exist):
    checkpoints/data_ready.pt
    checkpoints/pipeline_validated.pt   (per_drug_metrics block)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "checkpoints" / "data_ready.pt"
RM_VAL = REPO / "checkpoints" / "pipeline_validated.pt"
OUT_DEFAULT = REPO / "paper" / "v8_artifacts" / "v12_sprint1" / "paired_wilcoxon_latefusion.json"

# Multi-seed protocol (only used with --multi-seed); seed=42 is the singleton default.
SEEDS_MULTI = [0, 42, 1337, 2024, 31415]


def per_drug_mse_late_fusion(
    Xp: np.ndarray,
    Xe: np.ndarray,
    y: np.ndarray,
    splits: dict,
    alpha: float = 10.0,
) -> dict[int, dict]:
    """Return per-drug Late-Fusion MSE on the test split.

    Late Fusion = average of one Ridge per modality (proteomics, epigenomics),
    fit independently per drug. This mirrors baseline_late_fusion in
    scripts/data_integration_audit.py but preserves per-drug residuals.
    """
    out: dict[int, dict] = {}
    tr_idx = np.asarray(splits["train"], dtype=np.int64)
    te_idx = np.asarray(splits["test"], dtype=np.int64)
    Xtr_p, Xte_p = Xp[tr_idx], Xp[te_idx]
    Xtr_e, Xte_e = Xe[tr_idx], Xe[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    for d in range(ytr.shape[1]):
        tr_mask = ~np.isnan(ytr[:, d])
        te_mask = ~np.isnan(yte[:, d])
        if tr_mask.sum() < 5 or te_mask.sum() == 0:
            out[d] = {"mse": float("nan"), "n_obs": int(te_mask.sum())}
            continue
        m_p = Ridge(alpha=alpha).fit(Xtr_p[tr_mask], ytr[tr_mask, d])
        m_e = Ridge(alpha=alpha).fit(Xtr_e[tr_mask], ytr[tr_mask, d])
        pred = 0.5 * (m_p.predict(Xte_p[te_mask]) + m_e.predict(Xte_e[te_mask]))
        resid = pred - yte[te_mask, d]
        out[d] = {"mse": float(np.mean(resid ** 2)), "n_obs": int(te_mask.sum())}
    return out


def per_drug_mse_resistancemap(rm_val_path: Path) -> dict[int, dict]:
    """Read per-drug MSE from pipeline_validated.pt[metrics][per_drug_metrics]."""
    ckpt = torch.load(rm_val_path, map_location="cpu", weights_only=False)
    pdm = ckpt.get("metrics", {}).get("per_drug_metrics", [])
    if not pdm:
        raise RuntimeError(
            f"per_drug_metrics absent from {rm_val_path}. Re-run "
            "`python -m resistancemap.main --config configs/default.yaml`."
        )
    return {i: {"mse": float(row["mse"]), "n_obs": int(row["n_obs"]),
                "drug": str(row["drug"])} for i, row in enumerate(pdm)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-ready", type=Path, default=DATA,
                    help=f"Path to data_ready.pt (default: {DATA})")
    ap.add_argument("--rm-val", type=Path, default=RM_VAL,
                    help=f"Path to pipeline_validated.pt (default: {RM_VAL})")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT,
                    help=f"Output JSON path (default: {OUT_DEFAULT})")
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="Ridge regularization alpha (default: 10.0)")
    ap.add_argument("--multi-seed", action="store_true",
                    help=(f"Retrain ResistanceMap at seeds {SEEDS_MULTI}. "
                          "Not implemented in this script — emits "
                          "an instructive RuntimeError. Run "
                          "scripts/v12/retrain_5seeds.py first."))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.multi_seed:
        raise RuntimeError(
            "--multi-seed requires per-seed checkpoints "
            "(pipeline_validated_seed{0,42,1337,2024,31415}.pt). Run "
            "scripts/v12/retrain_5seeds.py to materialize them, then re-run "
            "this script without --multi-seed against each (~30 min CPU/seed)."
        )

    ckpt = torch.load(args.data_ready, map_location="cpu", weights_only=False)
    ds = ckpt["dataset"]
    splits = ckpt["splits"]
    Xp = ds.proteomics.numpy().astype(np.float32)
    Xe = ds.epigenomics.numpy().astype(np.float32)
    y = ds.drug_sensitivity.numpy().astype(np.float32)
    drug_names = list(ds.drug_names)

    logger.info(f"Loaded data_ready.pt: N={Xp.shape[0]}, "
                f"P_proteomics={Xp.shape[1]}, E_epigenomics={Xe.shape[1]}, "
                f"D_drugs={y.shape[1]} ({len(drug_names)} names)")

    lf = per_drug_mse_late_fusion(Xp, Xe, y, splits, alpha=args.alpha)
    rm = per_drug_mse_resistancemap(args.rm_val)

    pairs: list[dict] = []
    for d in range(len(drug_names)):
        rm_d = rm.get(d, {}).get("mse", float("nan"))
        lf_d = lf.get(d, {}).get("mse", float("nan"))
        if np.isnan(rm_d) or np.isnan(lf_d):
            continue
        pairs.append({
            "drug": drug_names[d],
            "rm_mse": float(rm_d),
            "lf_mse": float(lf_d),
            "delta_rm_minus_lf": float(rm_d - lf_d),
            "n_obs": int(rm.get(d, {}).get("n_obs", 0)),
        })

    deltas = np.array([p["delta_rm_minus_lf"] for p in pairs], dtype=float)
    if deltas.size == 0:
        raise RuntimeError("No valid per-drug pairs found.")

    # Primary test: paired one-sided Wilcoxon signed-rank
    w_stat, w_p_less = stats.wilcoxon(deltas, alternative="less",
                                      zero_method="wilcox")
    _, w_p_two = stats.wilcoxon(deltas, alternative="two-sided",
                                zero_method="wilcox")

    # Secondary: per-drug sign test (placeholder; replace with per-sample
    # paired-t once RM per-sample residuals are persisted).
    per_drug_p = []
    for d in deltas:
        # binomtest(observed_negative=int(d<0), n=1, p=0.5) — uninformative at n=1
        # but kept as a structural slot; Holm correction over 11 entries
        # remains conservative.
        per_drug_p.append(float(stats.binomtest(int(d < 0), 1, 0.5).pvalue))
    _, holm_p, _, _ = multipletests(per_drug_p, alpha=0.05, method="holm")

    out = {
        "schema_version": 1,
        "design": (
            "Paired Wilcoxon signed-rank across 11 drugs at the seed=42 split. "
            "H1: median(MSE_RM - MSE_LF) < 0 (one-sided). Predicted outcome: "
            "test likely fails to reject the null; the 0.0078 pooled gap is "
            "dominated by Panobinostat (MSE~22.3) whose magnitude Wilcoxon "
            "ignores."
        ),
        "n_drug_pairs": len(pairs),
        "pairs": pairs,
        "pooled_rm_mse_mean": float(np.mean([p["rm_mse"] for p in pairs])),
        "pooled_lf_mse_mean": float(np.mean([p["lf_mse"] for p in pairs])),
        "median_delta_rm_minus_lf": float(np.median(deltas)),
        "mean_delta_rm_minus_lf": float(np.mean(deltas)),
        "wilcoxon": {
            "statistic": float(w_stat),
            "p_one_sided_less": float(w_p_less),
            "p_two_sided": float(w_p_two),
            "alternative_for_primary": "RM < LF (one-sided)",
        },
        "holm_bonferroni_per_drug": [
            {"drug": pairs[i]["drug"], "raw_p": per_drug_p[i],
             "adj_p": float(holm_p[i])}
            for i in range(len(pairs))
        ],
        "caveats": [
            "Single seed=42 by default; --multi-seed requires per-seed retrains.",
            "Per-drug secondary uses sign test (n=1); upgrade to paired-t over "
            "per-sample residuals once RM persists them.",
            "Wilcoxon exact one-sided p floor at n=11 is ~5e-4; the test cannot "
            "reject below that.",
            "Pooled aggregate gap is dominated by Panobinostat outlier (MSE~22); "
            "Wilcoxon ranks ignore magnitudes, so the direction of the test can "
            "flip relative to the pooled comparison.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info(f"[done] wrote {args.out}")
    logger.info(f"  median Δ(RM-LF) = {np.median(deltas):+.4f}; "
                f"Wilcoxon one-sided p = {w_p_less:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
