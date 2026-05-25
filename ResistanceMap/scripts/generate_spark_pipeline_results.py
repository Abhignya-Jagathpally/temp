#!/usr/bin/env python
"""
generate_spark_pipeline_results.py -- Generate publication-ready tables and
figures from the Spark evidence-pipeline run artifacts.

Reads real JSON/CSV results only. Never fabricates data. If an input file is
missing, writes "Data not available" in the corresponding table cell and
skips the corresponding figure panel.

Outputs under paper/v8_artifacts/spark_pipeline/:
  Tables:  table1_survival_proxy.{md,csv}
           table2_response_classification.{md,csv}
           table3_trajectory_blocked.md
           table4_pathway_evidence.{md,csv}
           data_quality_summary.md
  Figures: fig_survival_forest_plot.{png,pdf}
           fig_kaplan_meier_risk_groups.{png,pdf}
           fig_claim_gate_dashboard.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nature-style matplotlib configuration
# ---------------------------------------------------------------------------

NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Colorblind-friendly palette (Wong 2011)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red-orange
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def _apply_nature_style() -> None:
    """Apply Nature-style rcParams to matplotlib."""
    import matplotlib

    matplotlib.rcParams.update(NATURE_STYLE)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None if missing or malformed."""
    if not path.is_file():
        logger.warning("Missing file: %s", path)
        return None
    try:
        with open(path) as fh:
            # Handle NaN values which are invalid JSON but common in our outputs
            text = fh.read()
            text = text.replace(": NaN", ": null")
            return json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot parse %s: %s", path, exc)
        return None


def _load_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    """Load CSV into list of dicts, returning None if missing."""
    if not path.is_file():
        logger.warning("Missing file: %s", path)
        return None
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a numeric value for table display."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    return f"{f:.{decimals}f}"


def _safe_float(val: Any) -> float:
    """Convert to float, returning NaN on failure."""
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

# Human-readable model names and input descriptors
MODEL_DISPLAY = {
    "clinical_only_cox": ("Clinical-Only Cox", "ISS + age + gender + bort_1L + n_treatments"),
    "clinical_ridge_cox": ("Clinical Ridge Cox", "ISS + age + gender + bort_1L + n_treatments (L2)"),
    "rna_only_cox": ("RNA-Only Cox", "Top-5000 RNA genes"),
    "rna_plus_clinical_cox": ("RNA + Clinical Cox", "RNA + clinical features"),
    "random_survival_forest": ("Random Survival Forest", "Clinical features (tree ensemble)"),
    "deepsurv_mlp": ("DeepSurv MLP", "Clinical features (neural network)"),
    "mean_time_baseline": ("Mean Time Baseline", "Cohort mean event time"),
    "kaplan_meier_baseline": ("Kaplan-Meier Baseline", "Non-parametric KM estimator"),
    "locf_trajectory_baseline": ("LOCF Trajectory", "Last observation carried forward"),
    "mean_future_state_baseline": ("Mean Future State", "Population mean future state"),
    "mortfm_lens": ("MORT-FM LENS", "RNA + clinical + graph projector (LOO)"),
}


def generate_table1_survival(
    baseline_summary: Optional[Dict],
    lens_summary: Optional[Dict],
    out_dir: Path,
) -> None:
    """Table 1: TT2L survival-proxy comparison."""
    rows: List[Dict[str, str]] = []

    # Baselines
    if baseline_summary and "model_summary" in baseline_summary:
        for model_key, stats in baseline_summary["model_summary"].items():
            display_name, input_desc = MODEL_DISPLAY.get(
                model_key, (model_key, "Unknown")
            )
            mean_c = _safe_float(stats.get("mean_c_index"))
            std_c = _safe_float(stats.get("std_c_index"))
            ci_lo = mean_c - 1.96 * std_c if not math.isnan(std_c) else float("nan")
            ci_hi = mean_c + 1.96 * std_c if not math.isnan(std_c) else float("nan")
            rows.append({
                "Model": display_name,
                "Input": input_desc,
                "C-index": _fmt(mean_c),
                "IBS": _fmt(stats.get("mean_ibs")),
                "CI_low": _fmt(ci_lo),
                "CI_high": _fmt(ci_hi),
            })

    # MORT-FM LENS
    if lens_summary:
        c_idx = _safe_float(lens_summary.get("loo_cindex"))
        ci = lens_summary.get("loo_cindex_95ci", [None, None])
        rows.append({
            "Model": "MORT-FM LENS",
            "Input": "RNA + clinical + graph projector (LOO)",
            "C-index": _fmt(c_idx),
            "IBS": _fmt(lens_summary.get("integrated_brier_score")),
            "CI_low": _fmt(ci[0] if ci else None),
            "CI_high": _fmt(ci[1] if ci else None),
        })

    if not rows:
        rows.append({
            "Model": "Data not available",
            "Input": "Data not available",
            "C-index": "Data not available",
            "IBS": "Data not available",
            "CI_low": "Data not available",
            "CI_high": "Data not available",
        })

    # Write CSV
    csv_path = out_dir / "table1_survival_proxy.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write markdown
    md_path = out_dir / "table1_survival_proxy.md"
    with open(md_path, "w") as fh:
        fh.write("# Table 1: TT2L Survival-Proxy Comparison\n\n")
        fh.write("Endpoint: Time-to-next-treatment (TT2L) as a resistance proxy.\n\n")
        header = list(rows[0].keys())
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(row[k] for k in header) + " |\n")
        fh.write("\n")
        fh.write("CI = 95% confidence interval (bootstrap or LOO).\n")
        fh.write("IBS = Integrated Brier Score (lower is better).\n")

    logger.info("Wrote %s and %s", csv_path, md_path)


def generate_table2_response(out_dir: Path) -> None:
    """Table 2: Response classification placeholder."""
    rows = [
        {
            "Model": "Xiong et al. 2024 (clinical ceiling)",
            "Endpoint": "Short-term response (CR/VGPR vs PR/SD/PD)",
            "F1": "0.75",
            "Note": "Published clinical ceiling; no internal model trained yet",
        },
        {
            "Model": "MORT-FM",
            "Endpoint": "Short-term response (CR/VGPR vs PR/SD/PD)",
            "F1": "Data not available",
            "Note": "Response labels not yet wired into Spark DAG",
        },
    ]

    csv_path = out_dir / "table2_response_classification.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "table2_response_classification.md"
    with open(md_path, "w") as fh:
        fh.write("# Table 2: Short-Term Response Classification\n\n")
        fh.write("Clinical ceiling from Xiong et al. 2024 (F1 = 0.75).\n\n")
        header = list(rows[0].keys())
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(row[k] for k in header) + " |\n")

    logger.info("Wrote %s and %s", csv_path, md_path)


def generate_table3_trajectory(out_dir: Path) -> None:
    """Table 3: Trajectory blocked statement."""
    md_path = out_dir / "table3_trajectory_blocked.md"
    with open(md_path, "w") as fh:
        fh.write("# Table 3: Trajectory Forecasting -- BLOCKED\n\n")
        fh.write("**Status**: Trajectory baselines are BLOCKED.\n\n")
        fh.write("**Reason**: No real molecular visit timestamps are available in the\n")
        fh.write("GDC open-access MMRF CoMMpass download. Trajectory forecasting\n")
        fh.write("requires paired multi-omic snapshots at known calendar-time\n")
        fh.write("intervals (e.g., baseline + 6-month follow-up).\n\n")
        fh.write("**Required data**: dbGaP phs000748 controlled-access visit-level\n")
        fh.write("aliquot tables, or EGA longitudinal scRNA/multiome.\n\n")
        fh.write("**Blocked methods**: PRESCIENT, scNODE, TrajectoryNet, CellRank 2\n")
        fh.write("cannot run on fabricated visit_time_days.\n")

    logger.info("Wrote %s", md_path)


def generate_table4_pathway(causal_csv_rows: Optional[List[Dict]], out_dir: Path) -> None:
    """Table 4: Pathway evidence summary from causal edge CSV."""
    if causal_csv_rows is None:
        md_path = out_dir / "table4_pathway_evidence.md"
        with open(md_path, "w") as fh:
            fh.write("# Table 4: Pathway Evidence Summary\n\n")
            fh.write("Data not available: causal_edge_evidence_v2.csv not found.\n")
        logger.info("Wrote %s (empty)", md_path)
        return

    # Summarise top-20 edges by delta
    sorted_rows = sorted(
        causal_csv_rows,
        key=lambda r: _safe_float(r.get("delta", 0)),
        reverse=True,
    )
    top_k = sorted_rows[:20]

    summary_rows = []
    for r in top_k:
        summary_rows.append({
            "Source": r.get("source_hgnc", r.get("source_raw", "?")),
            "Target": r.get("target_hgnc", r.get("target_raw", "?")),
            "Delta": _fmt(r.get("delta"), 4),
            "CRISPR_Essential": r.get("target_is_common_essential", "?"),
            "Drug_Target": r.get("drug_target_either_endpoint", "?"),
            "Pathway": r.get("target_top_pathway", "?"),
            "Pathway_Count": r.get("target_pathway_count", "?"),
        })

    csv_path = out_dir / "table4_pathway_evidence.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = out_dir / "table4_pathway_evidence.md"
    with open(md_path, "w") as fh:
        fh.write("# Table 4: Pathway Evidence Summary (Top 20 Edges by Delta)\n\n")
        fh.write("Source: causal_edge_evidence_v2.csv (three-way join: CRISPR + drug-target + Reactome).\n\n")
        header = list(summary_rows[0].keys())
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in summary_rows:
            fh.write("| " + " | ".join(str(row[k]) for k in header) + " |\n")

        # Summary statistics
        n_essential = sum(
            1 for r in causal_csv_rows
            if str(r.get("target_is_common_essential", "")).lower() == "true"
        )
        n_drug_target = sum(
            1 for r in causal_csv_rows
            if str(r.get("drug_target_either_endpoint", "")).lower() == "true"
        )
        fh.write(f"\n**Total edges scored**: {len(causal_csv_rows)}\n")
        fh.write(f"**Common-essential targets**: {n_essential}\n")
        fh.write(f"**Drug-target supported edges**: {n_drug_target}\n")

    logger.info("Wrote %s and %s", csv_path, md_path)


def generate_data_quality_summary(
    dq_report: Optional[Dict],
    split_manifest: Optional[Dict],
    out_dir: Path,
) -> None:
    """Data quality and audit summary."""
    md_path = out_dir / "data_quality_summary.md"
    with open(md_path, "w") as fh:
        fh.write("# Data Quality and Audit Summary\n\n")

        if dq_report:
            fh.write("## Cohort\n\n")
            fh.write(f"- **Total patients**: {dq_report.get('n_patients', 'N/A')}\n")
            fh.write(f"- **Total rows**: {dq_report.get('total_rows', 'N/A')}\n")
            fh.write(f"- **Survival rows**: {dq_report.get('n_survival_rows', 'N/A')}\n")
            fh.write(f"- **Trajectory pairs**: {dq_report.get('n_trajectory_rows', 'N/A')}\n")
            fh.write(f"- **Overall audit passed**: {dq_report.get('overall_passed', 'N/A')}\n\n")

            fh.write("## Gates\n\n")
            gates = dq_report.get("gates", {})
            fh.write("| Gate | Threshold | Actual | Passed |\n")
            fh.write("| --- | --- | --- | --- |\n")
            for gate_name, gate_info in gates.items():
                fh.write(
                    f"| {gate_name} | {gate_info.get('threshold', '?')} "
                    f"| {gate_info.get('actual', '?')} "
                    f"| {'PASS' if gate_info.get('passed') else 'FAIL'} |\n"
                )

            fh.write("\n## Allowed Claims\n\n")
            for claim in dq_report.get("allowed_claims", []):
                fh.write(f"- {claim}\n")

            fh.write("\n## Blocked Claims\n\n")
            blocked = dq_report.get("blocked_claims", {})
            if blocked:
                for claim, reason in blocked.items():
                    fh.write(f"- **{claim}**: {reason}\n")
            else:
                fh.write("- None\n")
        else:
            fh.write("Data quality report not available.\n\n")

        if split_manifest:
            fh.write("\n## Patient-Disjoint Splits\n\n")
            splits = split_manifest.get("splits", {})
            fh.write("| Split | Patients | Rows | Survival Rows | Trajectory Rows |\n")
            fh.write("| --- | --- | --- | --- | --- |\n")
            for split_name, info in splits.items():
                fh.write(
                    f"| {split_name} | {info.get('n_patients', '?')} "
                    f"| {info.get('n_rows', '?')} "
                    f"| {info.get('n_survival_rows', '?')} "
                    f"| {info.get('n_trajectory_rows', '?')} |\n"
                )
            fh.write(f"\nSplit seed: {split_manifest.get('seed', '?')}\n")
            fh.write(f"Method: {split_manifest.get('method', '?')}\n")
        else:
            fh.write("\nSplit manifest not available.\n")

    logger.info("Wrote %s", md_path)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def _collect_forest_plot_data(
    baseline_summary: Optional[Dict],
    lens_summary: Optional[Dict],
) -> List[Tuple[str, float, float, float]]:
    """Collect (label, c_index, ci_lo, ci_hi) for forest plot."""
    entries: List[Tuple[str, float, float, float]] = []

    if baseline_summary and "model_summary" in baseline_summary:
        for model_key, stats in baseline_summary["model_summary"].items():
            display_name = MODEL_DISPLAY.get(model_key, (model_key,))[0]
            mean_c = _safe_float(stats.get("mean_c_index"))
            std_c = _safe_float(stats.get("std_c_index"))
            if math.isnan(mean_c):
                continue
            ci_lo = mean_c - 1.96 * std_c if not math.isnan(std_c) else mean_c
            ci_hi = mean_c + 1.96 * std_c if not math.isnan(std_c) else mean_c
            entries.append((display_name, mean_c, ci_lo, ci_hi))

    if lens_summary:
        c_idx = _safe_float(lens_summary.get("loo_cindex"))
        ci = lens_summary.get("loo_cindex_95ci", [None, None])
        ci_lo = _safe_float(ci[0]) if ci and ci[0] is not None else c_idx
        ci_hi = _safe_float(ci[1]) if ci and ci[1] is not None else c_idx
        if not math.isnan(c_idx):
            entries.append(("MORT-FM LENS", c_idx, ci_lo, ci_hi))

    return entries


def generate_fig_forest_plot(
    baseline_summary: Optional[Dict],
    lens_summary: Optional[Dict],
    out_dir: Path,
) -> None:
    """Forest plot of C-index with 95% CIs."""
    import matplotlib.pyplot as plt

    entries = _collect_forest_plot_data(baseline_summary, lens_summary)
    if not entries:
        logger.warning("No data for forest plot -- skipping.")
        return

    _apply_nature_style()

    # Sort by C-index descending for readability
    entries.sort(key=lambda x: x[1], reverse=True)
    labels = [e[0] for e in entries]
    c_vals = [e[1] for e in entries]
    ci_lo = [e[2] for e in entries]
    ci_hi = [e[3] for e in entries]

    n = len(entries)
    fig, ax = plt.subplots(figsize=(3.5, max(2.0, 0.35 * n + 0.5)))

    y_positions = list(range(n))
    colors = [PALETTE[7] if lab != "MORT-FM LENS" else PALETTE[0] for lab in labels]

    for i in range(n):
        ax.plot(
            [ci_lo[i], ci_hi[i]],
            [y_positions[i], y_positions[i]],
            color=colors[i],
            linewidth=1.2,
            solid_capstyle="round",
        )
        ax.plot(
            c_vals[i],
            y_positions[i],
            "o",
            color=colors[i],
            markersize=4,
            zorder=5,
        )

    ax.axvline(x=0.5, color="#999999", linestyle="--", linewidth=0.5, label="Chance (0.5)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("C-index")
    ax.set_title("TT2L Survival-Proxy: C-index Forest Plot")
    ax.set_xlim(0.35, 1.05)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_survival_forest_plot.{fmt}", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote fig_survival_forest_plot.{png,pdf}")


def generate_fig_kaplan_meier(lens_summary: Optional[Dict], out_dir: Path) -> None:
    """KM curves for high/mid/low risk tertiles from LOO predictions."""
    import matplotlib.pyplot as plt

    if lens_summary is None:
        logger.warning("No LENS summary -- skipping KM figure.")
        return

    km_data = lens_summary.get("km_strata_separation")
    if km_data is None:
        logger.warning("No km_strata_separation in LENS summary -- skipping KM figure.")
        return

    strata = km_data.get("strata", {})
    if not strata:
        logger.warning("Empty strata in KM data -- skipping.")
        return

    _apply_nature_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    stratum_colors = {"low": PALETTE[2], "mid": PALETTE[0], "high": PALETTE[3]}
    stratum_labels = {"low": "Low risk", "mid": "Mid risk", "high": "High risk"}

    # Time landmarks in days
    time_points = [0, 180, 365]
    time_labels_months = [0, 6, 12]

    for stratum_name in ["low", "mid", "high"]:
        info = strata.get(stratum_name)
        if info is None:
            continue

        s_vals = [
            1.0,
            _safe_float(info.get("median_S_at_180d", 1.0)),
            _safe_float(info.get("median_S_at_365d", 1.0)),
        ]
        # Filter out NaN
        valid = [(t, s) for t, s in zip(time_labels_months, s_vals) if not math.isnan(s)]
        if not valid:
            continue

        t_plot = [v[0] for v in valid]
        s_plot = [v[1] for v in valid]
        color = stratum_colors.get(stratum_name, PALETTE[7])
        label = stratum_labels.get(stratum_name, stratum_name)
        n = info.get("n", "?")
        n_ev = info.get("n_events", "?")
        full_label = f"{label} (n={n}, events={n_ev})"

        # Step function style
        ax.step(t_plot, s_plot, where="post", color=color, label=full_label, linewidth=1.0)

    chi2 = km_data.get("log_rank_chi2_high_vs_low")
    p_val = km_data.get("log_rank_p_lower_bound")
    if chi2 is not None and p_val is not None:
        ax.text(
            0.98, 0.02,
            f"Log-rank (high vs low):\nchi2={chi2:.1f}, p<{p_val:.1e}",
            transform=ax.transAxes,
            fontsize=5,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("KM Curves by LOO Risk Tertile (TT2L)")
    ax.set_ylim(0.7, 1.02)
    ax.set_xlim(-0.5, 13)
    ax.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_kaplan_meier_risk_groups.{fmt}", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote fig_kaplan_meier_risk_groups.{png,pdf}")


def generate_fig_claim_dashboard(dq_report: Optional[Dict], out_dir: Path) -> None:
    """Visual dashboard showing allowed vs blocked claims."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if dq_report is None:
        logger.warning("No data quality report -- skipping claim dashboard.")
        return

    _apply_nature_style()

    allowed = dq_report.get("allowed_claims", [])
    blocked = dq_report.get("blocked_claims", {})

    # Build claim list with status
    claims = []
    for c in allowed:
        claims.append((c, True, ""))
    for c, reason in blocked.items():
        claims.append((c, False, reason))

    if not claims:
        logger.warning("No claims to display -- skipping claim dashboard.")
        return

    n = len(claims)
    fig, ax = plt.subplots(figsize=(5.5, max(1.5, 0.4 * n + 0.5)))

    y_positions = list(range(n))
    bar_colors = []
    labels = []
    for claim_name, is_allowed, reason in claims:
        bar_colors.append(PALETTE[2] if is_allowed else PALETTE[3])
        status = "ALLOWED" if is_allowed else "BLOCKED"
        labels.append(f"{claim_name} [{status}]")

    ax.barh(y_positions, [1] * n, color=bar_colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks([])
    ax.set_xlim(0, 1.5)
    ax.invert_yaxis()
    ax.set_title("Claim Gate Dashboard")

    # Add reason annotations for blocked claims
    for i, (claim_name, is_allowed, reason) in enumerate(claims):
        if not is_allowed and reason:
            # Truncate long reasons
            short_reason = reason[:80] + "..." if len(reason) > 80 else reason
            ax.text(
                1.02, i, short_reason, va="center", ha="left", fontsize=4.5,
                color="#555555",
            )

    # Legend
    allowed_patch = mpatches.Patch(color=PALETTE[2], label="Allowed")
    blocked_patch = mpatches.Patch(color=PALETTE[3], label="Blocked")
    ax.legend(handles=[allowed_patch, blocked_patch], loc="upper right", frameon=False)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_claim_gate_dashboard.{fmt}", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote fig_claim_gate_dashboard.{png,pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication tables and figures from Spark pipeline artifacts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root of the results/ directory.",
    )
    parser.add_argument(
        "--lake-dir",
        type=Path,
        default=Path("data/lake"),
        help="Root of the data/lake/ directory.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root of the logs/ directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/v8_artifacts/spark_pipeline"),
        help="Output directory for tables and figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load all artifacts
    # ---------------------------------------------------------------
    baseline_results = _load_json(
        args.results_dir / "baselines" / "patient_longitudinal" / "patient_longitudinal_results.json"
    )
    baseline_comparison = _load_json(
        args.results_dir / "baselines" / "patient_longitudinal" / "patient_longitudinal_claim_comparison.json"
    )
    dq_report = _load_json(args.lake_dir / "confirmed" / "data_quality_report.json")
    split_manifest = _load_json(args.lake_dir / "confirmed" / "split_manifest.json")
    lens_summary = _load_json(args.logs_dir / "mortfm" / "lens_resistance_summary.json")
    causal_csv = _load_csv_rows(
        args.results_dir / "mortfm" / "causal_edge_evidence_v2.csv"
    )

    # Use the claim-comparison summary if available, else fall back to raw results
    summary_source = baseline_comparison if baseline_comparison else baseline_results

    # ---------------------------------------------------------------
    # Tables
    # ---------------------------------------------------------------
    logger.info("Generating tables...")
    generate_table1_survival(summary_source, lens_summary, out_dir)
    generate_table2_response(out_dir)
    generate_table3_trajectory(out_dir)
    generate_table4_pathway(causal_csv, out_dir)
    generate_data_quality_summary(dq_report, split_manifest, out_dir)

    # ---------------------------------------------------------------
    # Figures (require matplotlib)
    # ---------------------------------------------------------------
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not installed -- skipping figure generation.")
        return 0

    # Use non-interactive backend for headless environments
    import matplotlib

    matplotlib.use("Agg")

    logger.info("Generating figures...")
    generate_fig_forest_plot(summary_source, lens_summary, out_dir)
    generate_fig_kaplan_meier(lens_summary, out_dir)
    generate_fig_claim_dashboard(dq_report, out_dir)

    logger.info("All outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
