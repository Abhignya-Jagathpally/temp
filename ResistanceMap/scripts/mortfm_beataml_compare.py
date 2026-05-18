#!/usr/bin/env python3
"""
scripts/mortfm_beataml_compare.py
=================================
Lane 1 — head-to-head comparison of two Block B variants on identical
patient-disjoint splits:

  * scratch  — same aligned-feature space, no Block A weight port
  * init-A   — same aligned-feature space + 137-param port from Block A

Reads the two per-drug CSVs produced by the finetune script and:
  1. Joins per drug into a paired table (delta Spearman).
  2. Bootstrap 95% CIs for: median Spearman per variant, paired delta.
     Bootstrap uses stdlib :mod:`random` (seeded) — resampling of REAL
     observed Spearman values, no synthetic data generated.
  3. Drug-family breakdown via :mod:`resistancemap.data.drug_ontology`.
  4. Top-k drug-recovery: do high-AUC drugs in truth-rank also surface
     in the model's top-k by predicted score? (Recall@5, @10, @20.)
  5. Honest claim verdict — passes the hematologic_specimen_drug_response
     gate only if BOTH variants achieve median Spearman >= 0.10 AND init-A
     beats scratch by >= 0.02 in paired bootstrap.

Outputs:
  * results/mortfm/beataml_transfer_vs_scratch.csv
  * results/mortfm/beataml_drug_family_metrics.csv
  * results/mortfm/beataml_top_k_recovery.csv
  * logs/mortfm/beataml_compare_summary.json
  * figures/mortfm/beataml_per_drug_spearman.png
  * figures/mortfm/beataml_transfer_gain.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random as _random
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.drug_ontology import lookup_entry_loose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_beataml_compare")


def _read_per_drug(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "spearman": f"spearman_{variant}",
        "spearman_p": f"p_{variant}",
        "mse_against_AUC": f"mse_{variant}",
        "n": f"n_{variant}",
    })
    return df[["drug_name", f"spearman_{variant}", f"p_{variant}",
               f"mse_{variant}", f"n_{variant}"]]


def _median(xs: Sequence[float]) -> float:
    xs = [x for x in xs if not (x is None or math.isnan(x))]
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _bootstrap_median(values: Sequence[float], n_boot: int = 2000, seed: int = 13) -> tuple[float, float, float]:
    """Resample REAL Spearman values with replacement; return (median, p2.5, p97.5)."""
    rng = _random.Random(seed)
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    medians: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        medians.append(_median(sample))
    return _median(vals), _quantile(medians, 0.025), _quantile(medians, 0.975)


def _bootstrap_paired_delta(a: Sequence[float], b: Sequence[float],
                             n_boot: int = 2000, seed: int = 17) -> tuple[float, float, float]:
    """Paired bootstrap of median(a) - median(b) over the REAL observed pairs."""
    rng = _random.Random(seed)
    pairs = [
        (float(x), float(y)) for x, y in zip(a, b)
        if not (math.isnan(float(x)) or math.isnan(float(y)))
    ]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    n = len(pairs)
    deltas: list[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(_median([s[0] for s in sample]) - _median([s[1] for s in sample]))
    obs_delta = _median([p[0] for p in pairs]) - _median([p[1] for p in pairs])
    return obs_delta, _quantile(deltas, 0.025), _quantile(deltas, 0.975)


def _drug_family_of(drug_name: str) -> Optional[str]:
    e = lookup_entry_loose(str(drug_name))
    return e.drug_class if e is not None else None


def _top_k_recovery(truth_order: pd.Series, pred_order: pd.Series, ks=(5, 10, 20)) -> dict:
    """Fraction of the top-k truth drugs that also appear in top-k predictions."""
    out = {}
    for k in ks:
        top_truth = set(truth_order.head(k).index.astype(str))
        top_pred = set(pred_order.head(k).index.astype(str))
        out[f"recall_at_{k}"] = len(top_truth & top_pred) / max(k, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-csv", default="logs/mortfm/beataml_per_drug_scratch.csv")
    ap.add_argument("--init-a-csv", default="logs/mortfm/beataml_per_drug_init_from_a.csv")
    ap.add_argument("--out-merged",
                    default="results/mortfm/beataml_transfer_vs_scratch.csv")
    ap.add_argument("--out-family",
                    default="results/mortfm/beataml_drug_family_metrics.csv")
    ap.add_argument("--out-topk",
                    default="results/mortfm/beataml_top_k_recovery.csv")
    ap.add_argument("--out-summary",
                    default="logs/mortfm/beataml_compare_summary.json")
    ap.add_argument("--fig-per-drug",
                    default="figures/mortfm/beataml_per_drug_spearman.png")
    ap.add_argument("--fig-gain",
                    default="figures/mortfm/beataml_transfer_gain.png")
    args = ap.parse_args()

    scratch = _read_per_drug(Path(args.scratch_csv), "scratch")
    init_a = _read_per_drug(Path(args.init_a_csv), "init_a")
    merged = scratch.merge(init_a, on="drug_name", how="inner")
    merged["delta_init_a_minus_scratch"] = (
        merged["spearman_init_a"] - merged["spearman_scratch"]
    )
    merged["drug_family"] = merged["drug_name"].map(_drug_family_of)
    Path(args.out_merged).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_merged, index=False)
    logger.info("Wrote merged per-drug -> %s (%d drugs)", args.out_merged, len(merged))

    # --- Bootstrap CIs ----------------------------------------------------
    med_scratch, lo_s, hi_s = _bootstrap_median(merged["spearman_scratch"].tolist())
    med_init, lo_i, hi_i = _bootstrap_median(merged["spearman_init_a"].tolist())
    delta_med, lo_d, hi_d = _bootstrap_paired_delta(
        merged["spearman_init_a"].tolist(), merged["spearman_scratch"].tolist(),
    )
    sig_scratch = int((merged["p_scratch"] < 0.05).sum())
    sig_init = int((merged["p_init_a"] < 0.05).sum())

    # --- Drug-family breakdown -------------------------------------------
    fam_rows = []
    for fam, sub in merged.groupby("drug_family", dropna=False):
        if pd.isna(fam):
            fam = "unmapped_in_mm_ontology"
        m_s = _median(sub["spearman_scratch"].tolist())
        m_i = _median(sub["spearman_init_a"].tolist())
        fam_rows.append({
            "drug_family": fam, "n_drugs": int(len(sub)),
            "median_spearman_scratch": m_s,
            "median_spearman_init_a": m_i,
            "delta": m_i - m_s,
        })
    family_df = pd.DataFrame(fam_rows).sort_values("n_drugs", ascending=False)
    Path(args.out_family).parent.mkdir(parents=True, exist_ok=True)
    family_df.to_csv(args.out_family, index=False)
    logger.info("Wrote drug-family metrics -> %s (%d families)", args.out_family, len(family_df))

    # --- Top-k recovery: init-A picks vs scratch picks -------------------
    truth_rank = merged.set_index("drug_name")["spearman_scratch"].sort_values(ascending=False)
    pred_rank = merged.set_index("drug_name")["spearman_init_a"].sort_values(ascending=False)
    topk = _top_k_recovery(truth_rank, pred_rank)
    Path(args.out_topk).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([topk]).to_csv(args.out_topk, index=False)

    # --- Optional matplotlib figures -------------------------------------
    figures_written = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Path(args.fig_per_drug).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(merged["spearman_scratch"], merged["spearman_init_a"], s=18, alpha=0.7)
        lim = max(
            abs(merged["spearman_scratch"]).max(),
            abs(merged["spearman_init_a"]).max(),
            0.1,
        )
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("Scratch Block B per-drug Spearman")
        ax.set_ylabel("Block-A-init per-drug Spearman")
        ax.set_title(
            f"BeatAML per-drug: n={len(merged)}, "
            f"delta_med={delta_med:+.3f} [95% CI {lo_d:+.3f}, {hi_d:+.3f}]"
        )
        fig.tight_layout()
        fig.savefig(args.fig_per_drug, dpi=130)
        plt.close(fig)
        figures_written.append(args.fig_per_drug)

        fig, ax = plt.subplots(figsize=(7, 4))
        gain = merged["delta_init_a_minus_scratch"].dropna().sort_values()
        ax.bar(range(len(gain)), gain.values,
                color=["#1f77b4" if v >= 0 else "#d62728" for v in gain.values])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("drug index (sorted by delta)")
        ax.set_ylabel("delta Spearman (init_a - scratch)")
        ax.set_title(f"Transfer gain per drug (n={len(gain)})")
        fig.tight_layout()
        fig.savefig(args.fig_gain, dpi=130)
        plt.close(fig)
        figures_written.append(args.fig_gain)
    except ImportError:
        logger.warning("matplotlib not available; skipped figure generation")

    # --- Summary + claim verdict -----------------------------------------
    HEME_GATE_THRESHOLD = 0.10
    PAIRED_DELTA_THRESHOLD = 0.02
    heme_specimen_gate_pass = (
        (med_init >= HEME_GATE_THRESHOLD)
        and (med_scratch >= HEME_GATE_THRESHOLD)
        and (delta_med >= PAIRED_DELTA_THRESHOLD)
        and (lo_d > 0)
    )
    summary = {
        "n_drugs_common": int(len(merged)),
        "median_spearman_scratch": med_scratch,
        "median_spearman_scratch_95ci": [lo_s, hi_s],
        "median_spearman_init_a": med_init,
        "median_spearman_init_a_95ci": [lo_i, hi_i],
        "paired_delta_median_init_minus_scratch": delta_med,
        "paired_delta_95ci": [lo_d, hi_d],
        "n_significant_p_lt_0_05_scratch": sig_scratch,
        "n_significant_p_lt_0_05_init_a": sig_init,
        "top_k_recovery_init_vs_scratch": topk,
        "drug_families_with_data": int(len(family_df)),
        "hematologic_specimen_drug_response_gate_thresholds": {
            "min_median_spearman_per_variant": HEME_GATE_THRESHOLD,
            "min_paired_delta_init_minus_scratch": PAIRED_DELTA_THRESHOLD,
            "require_paired_ci_excludes_zero": True,
        },
        "hematologic_specimen_drug_response_gate_pass": bool(heme_specimen_gate_pass),
        "honest_note": (
            "These are 3-epoch debug-tier runs on 213 train / 45 val / 45 "
            "test patients. The gate would be re-evaluated after >=30 "
            "epochs, drug-conditioned head, and per-drug supervision (the "
            "v15 to-do for a real hematologic specimen claim)."
        ),
        "outputs": {
            "merged_csv": args.out_merged,
            "family_csv": args.out_family,
            "topk_csv": args.out_topk,
            "figures": figures_written,
        },
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary: %s", json.dumps({
        k: summary[k] for k in [
            "median_spearman_scratch", "median_spearman_init_a",
            "paired_delta_median_init_minus_scratch", "paired_delta_95ci",
            "n_significant_p_lt_0_05_scratch", "n_significant_p_lt_0_05_init_a",
            "top_k_recovery_init_vs_scratch",
            "hematologic_specimen_drug_response_gate_pass",
        ]
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
