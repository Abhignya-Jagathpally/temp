#!/usr/bin/env python3
"""Runner that generates only paper figures we can produce from REAL data.

We deliberately skip any figure whose existing implementation falls back to
synthetic / RNG-generated numbers when real inputs are missing. The rule is:
no fabricated baseline or ablation numbers ever leave this run.

Real inputs available (from checkpoints/pipeline_validated.pt):
    - test_mse = 2.327
    - n_test_samples = 132
    - n_drugs = 11
    - target_drugs = [Bortezomib, Lenalidomide, ...]

Figures attempted:
    - figure1_architecture: structural diagram, no numeric inputs needed.
    - real_metrics_summary: a NEW honest figure showing only the real
      test_mse number on top of a per-drug placeholder note. We render
      ONLY the real scalar; no synthetic per-drug values are invented.

Figures explicitly skipped (and why):
    - figure2_main_results: requires per-baseline AUROC/AUPRC + calibration
      curves we do not have; current impl fabricates them.
    - figure3_interpretability: Waddington / pathway / counterfactual /
      gene-network panels are entirely RNG-driven in current impl.
    - figure4_temporal: trajectory + uncertainty + intervention windows
      are RNG-synthesized; we have no per-patient trajectory eval.
    - figure5_ablation: no ablation results were run.
    - figure_s1_disentanglement..s5_hyperparameter: all RNG-driven.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap")
sys.path.insert(0, str(REPO / "scripts"))

from generate_paper_figures import PaperFigureGenerator, set_nature_style


def load_real_metrics(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    real = dict(ckpt["metrics"])
    real["target_drugs"] = list(ckpt["config"]["data"]["target_drugs"])
    return real


def load_stage_val_losses(ckpt_dir: Path) -> dict:
    """Pull stage-level val_loss values from checkpoint files.

    vae_pretrained.pt stores best val_loss in `best_metric`. The fusion and
    landscape val_loss values for this run are passed in by the caller so
    that we record exactly what the orchestrator observed; we do not
    re-derive them from any model file to avoid confusion if checkpoints
    are overwritten.
    """
    out: dict = {}
    vae = torch.load(
        ckpt_dir / "vae_pretrained.pt", map_location="cpu", weights_only=False
    )
    out["vae_pretrained_val_loss"] = float(vae["best_metric"])
    return out


def figure_real_metrics_summary(out_dir: Path, real: dict) -> str:
    """Honest figure: shows ONLY the real scalar test_mse + sample counts.

    No fabricated per-drug breakdown. If you want a per-drug bar chart,
    you must compute and pass per-drug MSEs first.
    """
    set_nature_style({"font_size": 8})
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.axis("off")

    drugs_str = ", ".join(real["target_drugs"])
    text = (
        f"ResistanceMap v6 -- end-to-end pipeline validation (real metrics)\n"
        f"\n"
        f"Test MSE: {real['test_mse']:.3f}\n"
        f"Test samples: {real['n_test_samples']}\n"
        f"Drugs evaluated: {real['n_drugs']}\n"
        f"Split: {real['split']}\n"
        f"\n"
        f"Drug panel: {drugs_str}\n"
        f"\n"
        f"Note: per-drug AUROC/AUPRC and baseline comparisons require\n"
        f"additional eval runs and are intentionally not shown."
    )
    ax.text(0.02, 0.98, text, ha="left", va="top", family="monospace", fontsize=7)
    out = out_dir / "figure_real_metrics_summary.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> int:
    out_dir = REPO / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = REPO / "checkpoints"
    ckpt = ckpt_dir / "pipeline_validated.pt"
    real = load_real_metrics(ckpt)

    # Stage-level val_loss values from this run's checkpoints + orchestrator.
    stage = load_stage_val_losses(ckpt_dir)
    # These two are the orchestrator-reported best val_losses for this run.
    # Hard-coded here because the fusion/landscape checkpoints don't carry
    # them in the metrics dict; they were observed at training time.
    stage["fusion_val_loss"] = 1.2420
    stage["landscape_val_loss"] = 1.2434

    # Persist the real metrics blob alongside the figures for traceability.
    metrics_blob = (
        {k: v for k, v in real.items() if k != "target_drugs"}
        | stage
        | {"target_drugs": real["target_drugs"]}
    )
    (out_dir / "real_metrics.json").write_text(json.dumps(metrics_blob, indent=2))

    gen = PaperFigureGenerator(output_dir=str(out_dir), config={"format": "pdf"})

    status: list[tuple[str, str, str]] = []

    # ---- Attempted figures ----
    try:
        path = gen.figure1_architecture()
        status.append(("figure1_architecture", "generated", path))
    except Exception as e:  # noqa: BLE001
        status.append(("figure1_architecture", f"errored: {e}", ""))

    try:
        path = figure_real_metrics_summary(out_dir, real)
        status.append(("figure_real_metrics_summary", "generated", path))
    except Exception as e:  # noqa: BLE001
        status.append(("figure_real_metrics_summary", f"errored: {e}", ""))

    # ---- Skipped figures (no fabrication policy) ----
    skipped = [
        ("figure2_main_results",
         "skipped: requires per-baseline AUROC/AUPRC + calibration we do not have"),
        ("figure3_interpretability",
         "skipped: Waddington/pathway/counterfactual/gene-net panels are RNG-driven"),
        ("figure4_temporal",
         "skipped: trajectory + uncertainty + intervention panels are RNG-driven"),
        ("figure5_ablation",
         "skipped: no ablation runs were executed"),
        ("figure_s1_disentanglement", "skipped: RNG-driven"),
        ("figure_s2_identifiability", "skipped: RNG-driven"),
        ("figure_s3_stability", "skipped: RNG-driven"),
        ("figure_s4_per_drug", "skipped: per-drug ROC requires per-drug eval we do not have"),
        ("figure_s5_hyperparameter", "skipped: no hyperparam sweep was run"),
    ]
    for name, why in skipped:
        status.append((name, why, ""))

    print("\n=== Per-figure status ===")
    for name, st, path in status:
        line = f"  {name:35s}  {st}"
        if path:
            line += f"  -> {path}"
        print(line)

    # Final listing
    print("\n=== Files in paper/figures ===")
    for p in sorted(out_dir.iterdir()):
        sz = p.stat().st_size
        print(f"  {p.name:40s}  {sz:>10d} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
