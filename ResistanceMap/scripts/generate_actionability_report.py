#!/usr/bin/env python3
"""v8 actionability decision matrix.

Reads per-drug metrics + actionability flags from checkpoints/pipeline_validated.pt
and writes a clinically-interpretable decision matrix to:

  paper/v8_artifacts/actionability_matrix.md

For each of the 11 GDSC drugs, this answers in plain language:
  - Is ResistanceMap useful for ranking this drug's IC50? (Spearman ≥ 0.25 AND n ≥ 30)
  - Is it also well-calibrated for absolute prediction? (additionally MSE < 1.0)
  - Or is it a known failure mode?

Plus drug-class aggregates and a "use this for / don't use this for" summary
that the README points at.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints" / "pipeline_validated.pt"
OUT = REPO / "paper" / "v8_artifacts" / "actionability_matrix.md"


def main():
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = c.get("metrics") or {}
    rows = m.get("per_drug_metrics") or []
    summary = m.get("actionability_summary") or {}
    if not rows:
        raise SystemExit("pipeline_validated.pt has no per_drug_metrics; rerun the validate stage.")
    if not summary:
        raise SystemExit("pipeline_validated.pt has no actionability_summary; rerun the validate stage.")

    rows_sorted = sorted(rows, key=lambda r: (-int(r.get("actionable_for_screening", False)),
                                              r.get("mse", float("inf"))))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Actionability Decision Matrix — ResistanceMap v6")
    md.append("")
    md.append(f"_Generated from `checkpoints/pipeline_validated.pt`. test_mse={m.get('test_mse')} "
              f"on n_test={m.get('n_test_samples')} cell lines × {m.get('n_drugs')} drugs._")
    md.append("")
    md.append("**Thresholds (from `validate_pipeline`):**")
    th = summary.get("thresholds", {})
    md.append(f"- Actionable for compound-prioritization screening: Spearman ≥ {th.get('actionable_min_spearman')} "
              f"AND n_test ≥ {th.get('actionable_min_n_obs')}")
    md.append(f"- Well-calibrated for absolute IC50 prediction: additionally MSE < {th.get('well_calibrated_max_mse')}")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"> {summary.get('interpretation', '')}")
    md.append("")

    md.append("## Per-drug decision matrix")
    md.append("")
    md.append("| Drug | Class | n_test | MSE | MAE | Spearman | Screening? | Calibrated? | Verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows_sorted:
        screen = "✅" if r.get("actionable_for_screening") else "❌"
        cal = "✅" if r.get("well_calibrated") else "—"
        if r.get("well_calibrated"):
            verdict = "**Use for screening + IC50 estimation**"
        elif r.get("actionable_for_screening"):
            verdict = "Use for screening only (rank-meaningful, absolute IC50 unreliable)"
        elif r.get("failure_mode"):
            verdict = "**❌ DO NOT USE — failure mode documented**"
        else:
            verdict = "n/a"
        md.append(
            f"| {r['drug']} | {r.get('drug_class', '?')} | {r['n_obs']} | "
            f"{r['mse']:.3f} | {r['mae']:.3f} | "
            f"{r.get('spearman', float('nan')):.3f} | "
            f"{screen} | {cal} | {verdict} |"
        )
    md.append("")

    md.append("## By drug class")
    md.append("")
    md.append("| Class | n_drugs | Mean MSE | Mean Spearman | n_actionable | Drugs |")
    md.append("|---|---|---|---|---|---|")
    for cls, c in (summary.get("by_class") or {}).items():
        md.append(
            f"| {cls} | {c['n_drugs']} | {c['mean_mse']:.3f} | {c['mean_spearman']:.3f} | "
            f"{c['n_actionable']}/{c['n_drugs']} | {', '.join(c['drugs'])} |"
        )
    md.append("")

    md.append("## What this means in practice")
    md.append("")
    actionable = summary.get("actionable_drugs") or []
    well_cal = summary.get("well_calibrated_drugs") or []
    failures = summary.get("failure_drugs") or []
    md.append("### ✅ Use ResistanceMap for")
    md.append("")
    md.append("**Compound-prioritization screening** (rank candidate cell lines by predicted resistance) for these drugs:")
    md.append("")
    for d in actionable:
        cal_note = " — also well-calibrated for absolute IC50" if d in well_cal else ""
        md.append(f"- **{d}**{cal_note}")
    md.append("")
    md.append("### ❌ Do NOT use ResistanceMap for")
    md.append("")
    if failures:
        md.append(f"**{', '.join(failures)}** — Spearman correlation is at chance level. "
                  "These are documented failure modes. The omics features available do not contain "
                  "sufficient signal for this drug class given the current training data scale.")
    else:
        md.append("(no documented failure modes at current thresholds)")
    md.append("")
    md.append("### ⚠️ Use with caution")
    md.append("")
    actionable_only = [d for d in actionable if d not in well_cal]
    if actionable_only:
        md.append(f"**{', '.join(actionable_only)}** — rank-correlation is meaningful but absolute IC50 "
                  "estimates have MSE > 1.0. Use the predictions to *rank* compounds, not to "
                  "estimate dose-response curves.")
    md.append("")

    md.append("## Always disqualified (out of scope)")
    md.append("")
    md.append("- Patient-level IC50 prediction (no patient training data; cell-line cohort only)")
    md.append("- Time-to-resistance forecasting (no longitudinal training pairs)")
    md.append("- Pathway-level transition prediction (no perturbation grounding)")
    md.append("- HDAC-class compounds (3 of 3 HDAC drugs underperform — class-level limitation)")
    md.append("")

    md.append("## Reproducibility")
    md.append("")
    md.append("```bash")
    md.append("python main.py --config configs/default.yaml")
    md.append("python scripts/run_baselines_real.py")
    md.append("python scripts/generate_actionability_report.py")
    md.append("```")
    md.append("")
    md.append("All numbers above derive from `checkpoints/pipeline_validated.pt`. ")
    md.append("Source of truth: `logs/per_drug_metrics.csv` + `paper/tables/baseline_comparison.json`.")
    md.append("")

    OUT.write_text("\n".join(md))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
