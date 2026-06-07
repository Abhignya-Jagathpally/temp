#!/usr/bin/env python3
"""
ResistanceMap v20 — Combined Results Report Generator
=====================================================
Reads all outputs and produces a single markdown report.
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")


def main():
    report = []
    report.append("# ResistanceMap v20 — Combined Results Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Branch:** v20/lab-first-pivot")
    report.append(f"**Data:** Open-access only (GDC open tier + GEO + Zenodo)")
    report.append("")

    # --- Section 1: Baselines ---
    baseline_path = RESULTS_DIR / "v20_first_baselines" / "v20_first_baselines.md"
    if baseline_path.exists():
        report.append("---")
        report.append(baseline_path.read_text())
    else:
        report.append("\n## Baselines\n\n*Not yet run. Execute `python scripts/run_first_results.py`*\n")

    # --- Section 2: scVI ---
    scvi_latent = Path("data/processed/scvi_latent_z.npy")
    if scvi_latent.exists():
        import numpy as np
        z = np.load(scvi_latent)
        report.append("\n---")
        report.append("\n## scVI Integration\n")
        report.append(f"- Latent space: {z.shape[0]} cells × {z.shape[1]} dimensions")
        report.append(f"- Model: checkpoints/scvi/scvi_model/")

        chromatin_path = Path("data/processed/chromatin_expression.csv")
        if chromatin_path.exists():
            import pandas as pd
            ch = pd.read_csv(chromatin_path)
            report.append(f"- Chromatin genes extracted: {ch.shape[1]} genes × {ch.shape[0]} cells")

        proxy_path = Path("data/processed/biomarker_proxy_expression.csv")
        if proxy_path.exists():
            import pandas as pd
            pr = pd.read_csv(proxy_path)
            report.append(f"- Biomarker proxies extracted: {pr.shape[1]} genes × {pr.shape[0]} cells")
    else:
        report.append("\n## scVI Integration\n\n*Not yet run. Execute `python scripts/run_scvi_integration.py`*\n")

    # --- Section 3: PK-SSM ---
    pkssm_path = RESULTS_DIR / "v20_pkssm" / "forward_pass_results.json"
    if pkssm_path.exists():
        with open(pkssm_path) as f:
            pkssm = json.load(f)
        report.append("\n---")
        report.append("\n## PK-SSM Forward Pass (Untrained)\n")
        report.append(f"- Cells processed: {pkssm['n_cells']}")
        report.append(f"- Model parameters: {pkssm['pk_ssm_params']:,}")
        report.append(f"- PK priors: FLC half-life {pkssm['model_config']['pk_priors']['flc_halflife_hours']}h, "
                      f"IgG half-life {pkssm['model_config']['pk_priors']['igg_halflife_days']}d, "
                      f"B2M half-life {pkssm['model_config']['pk_priors']['b2m_halflife_hours']}h")
        report.append(f"\n**Mechanism distribution (random init — NOT trained):**\n")
        for mech, count in pkssm['mechanism_distribution'].items():
            pct = count / pkssm['n_cells'] * 100
            report.append(f"- {mech}: {count} ({pct:.1f}%)")
        report.append(f"\n> ⚠ **These are untrained outputs.** The architecture runs end-to-end ")
        report.append(f"> on real scRNA-seq data, but mechanism assignments and hazard estimates ")
        report.append(f"> are from random initialization. Training requires MMRF Virtual Lab data.")
    else:
        report.append("\n## PK-SSM\n\n*Not yet run. Execute `python scripts/run_pkssm_forward.py`*\n")

    # --- Section 4: What This Proves ---
    # Each claim is gated on the artifact that would substantiate it — a step
    # that did not run this session is reported as NOT yet demonstrated, never
    # asserted as proven (no claim inflation).
    report.append("\n---")
    report.append("\n## What This Report Proves (gated on artifacts produced this run)\n")
    proven, not_yet = [], []
    (proven if baseline_path.exists() else not_yet).append(
        "The corrected baseline cascade runs with proper censoring (sksurv)")
    (proven if scvi_latent.exists() else not_yet).append(
        "scVI integrates MM scRNA-seq into batch-corrected latents + extracts "
        "chromatin / biomarker-proxy expression")
    (proven if pkssm_path.exists() else not_yet).append(
        "The PK-SSM architecture runs end-to-end on real biological data")
    proven.append("All open-access — no IRB, no dbGaP, no gateway registration required")
    for i, c in enumerate(proven, 1):
        report.append(f"{i}. {c}")
    if not_yet:
        report.append("\n**NOT demonstrated this run (step did not complete):**")
        for c in not_yet:
            report.append(f"- {c}")

    report.append("\n## What Remains Data-Blocked\n")
    report.append("- Training the PK-SSM survival head (requires MMRF Virtual Lab visit-level labs + outcomes)")
    report.append("- Producing actual C-index for PK-SSM vs baselines")
    report.append("- Validating mechanism classifier assignments against molecular subtypes")
    report.append("- External validation on GSE136337 (B2M/albumin/LDH) or GMMG-MM5")

    # Write
    output_path = RESULTS_DIR / "v20_combined_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to: {output_path}")
    print("\n".join(report))


if __name__ == "__main__":
    main()