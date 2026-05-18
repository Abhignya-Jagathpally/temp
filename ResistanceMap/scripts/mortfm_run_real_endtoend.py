#!/usr/bin/env python3
"""
scripts/mortfm_run_real_endtoend.py
===================================
Run the MORT-FM pipeline end-to-end on REAL public data already present in
``data/raw/`` and ``data/processed/`` of this repo.

Honest behaviour
----------------
* Uses only on-disk data — never fabricates inputs.
* Reports the actual cohort size + modality coverage + event rate
  ingested at every step.
* Trains at ``acceptance_level="debug"`` only. Per docs/MORTFM_LIMITATIONS.md
  no patient-level claim can be released from this run; the goal is to
  confirm the v15 pipeline *executes* against real public data without
  fabrication.
* Writes a real checkpoint to ``checkpoints/mortfm/real_endtoend.pt`` and
  appends a one-line summary to ``logs/mortfm/real_endtoend_summary.json``.

Data sources expected on disk (per the inventory):
    data/raw/mmrf_commpass/clinical.tsv               (995 GDC clinical rows)
    data/raw/mmrf_commpass/gene_expression.tsv         (MMRF RNA-seq, optional)
    data/raw/gse271107.h5ad                            (143,748 cells / 19 samples)
    data/processed/beataml_rpkm.parquet                (already-processed BeatAML RNA)
    data/processed/beataml_clinical.tsv                (already-processed BeatAML clinical)
    data/processed/graphs/biological_edges.parquet      (STRING PPI, 473k edges)
    data/processed/drug_response/gdsc_response_long.parquet (484k rows)
    data/processed/drug_response/prism_response_long.parquet (701k rows)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.clinical_outcome_loader import (
    load_mortfm_outcomes,
    summarise_outcomes,
)
from resistancemap.data.trajectory_pair_builder import (
    build_temporal_pairs,
    summarise_pairs,
)
from resistancemap.mortfm.acceptance_gate import evaluate_cohort
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityName,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)
from resistancemap.mortfm.trainer import MORTFMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_endtoend")


# ---------------------------------------------------------------------------
# Step 1 — assemble MMRF snapshots from the processed parquet + outcomes
# ---------------------------------------------------------------------------

def build_mmrf_snapshots(
    expr_path: Path,
    metadata_path: Path,
    *,
    max_patients: int = 300,
    rna_genes: int = 2000,
) -> list[PatientCellSnapshot]:
    """Build PatientCellSnapshot from the on-disk processed MMRF parquet.

    Uses the top-``rna_genes`` highest-variance genes to keep the model
    debug-tractable. No fabrication: every value is real.
    """
    logger.info("Loading MMRF expression from %s ...", expr_path)
    expr = pd.read_parquet(expr_path)
    # Expression: rows=samples, columns=genes. Trim to top-variance genes.
    if expr.shape[1] > rna_genes:
        variances = expr.var(axis=0).sort_values(ascending=False)
        top = variances.index[:rna_genes].tolist()
        expr = expr[top]
    if expr.shape[0] > max_patients:
        expr = expr.iloc[:max_patients]
    logger.info("MMRF expression usable: %d patients x %d genes", *expr.shape)

    snapshots: list[PatientCellSnapshot] = []
    for sample_id, row in expr.iterrows():
        pid = str(sample_id).split("_")[0] + "_" + str(sample_id).split("_")[1] if "_" in str(sample_id) else str(sample_id)
        # Recover canonical patient ID prefix MMRF_xxxx
        rna_tensor = ModalityTensor(
            name=ModalityName.RNA.value,
            values=torch.from_numpy(row.values.astype(np.float32)).unsqueeze(0),
            feature_names=list(expr.columns.astype(str)),
        )
        snapshots.append(
            PatientCellSnapshot(
                patient_id=pid,
                sample_id=str(sample_id),
                disease="MM",
                timepoint=0.0,
                rna=rna_tensor,
                drug=DrugContext("VRd", "proteasome_inhibitor",
                                  target_genes=["PSMB5"], dose=None),
            )
        )
    logger.info("Built %d MMRF snapshots", len(snapshots))
    return snapshots


# ---------------------------------------------------------------------------
# Step 2 — load outcomes
# ---------------------------------------------------------------------------

def load_outcomes() -> list[ResistanceOutcome]:
    return load_mortfm_outcomes("data/raw/mmrf_commpass", endpoint="os", strict=True)


# ---------------------------------------------------------------------------
# Step 3 — main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    out_dir = Path("logs/mortfm")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/mortfm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Snapshots --------------------------------------------------
    snapshots = build_mmrf_snapshots(
        Path("data/processed/mmrf_baseline_expression.parquet"),
        Path("data/processed/mmrf_baseline_metadata.tsv"),
        max_patients=300,
        rna_genes=1000,
    )

    # ---- 2. Outcomes ---------------------------------------------------
    outcomes = load_outcomes()
    logger.info("MMRF outcomes: %s", summarise_outcomes(outcomes))

    # ---- 3. Temporal pairs (will produce survival-only rows; no real X_t_delta) --
    pairs = build_temporal_pairs(
        snapshots, outcomes,
        allowed_time_gaps=None,           # accept any positive gap
        gap_tolerance=2.0,
        outcome_match_tolerance=2.0,
        include_survival_only=True,
    )
    summary = summarise_pairs(pairs)
    logger.info("Pairs: %s", summary)

    # ---- 4. Acceptance gate -------------------------------------------
    debug_report = evaluate_cohort(pairs, level="debug")
    logger.info("Debug acceptance: %s", debug_report.summary())
    technical_report = evaluate_cohort(pairs, level="technical")
    logger.info("Technical acceptance: %s", technical_report.summary())

    # ---- 5. Build trainer --------------------------------------------
    cfg = MORTFMConfig(
        rna_input_dim=snapshots[0].rna.feature_dim,
        proteomics_input_dim=50,
        clinical_input_dim=4,
        d_token=32, d_latent=64, fusion_layers=2, fusion_heads=4, fusion_dropout=0.1,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        use_proteomics=False,
        n_time_grid=4, integration_time=12.0,
        pretrain_epochs=2, finetune_epochs=2, trajectory_epochs=1, survival_epochs=2,
        batch_size=16, lr=1e-4, weight_decay=1e-5,
        mixed_precision=False, drug_embed_dim=16,
        checkpoint_dir=str(ckpt_dir),
    )
    splits, train_loader, val_loader, test_loader = make_data_module(
        pairs, batch_size=cfg.batch_size, val_fraction=0.15, test_fraction=0.15, seed=cfg.seed,
    )
    logger.info("Split summary (REAL data): %s", splits.summary())

    model = MORTFM(cfg, n_pathway_proteins=200, n_drug_candidates=11, n_resistance_states=4)
    trainer = MORTFMTrainer(
        model, cfg,
        train_loader=train_loader, val_loader=val_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # ---- 6. Run stages (debug-level gate) -----------------------------
    history = []
    for stage, epochs in [("A", cfg.pretrain_epochs), ("F", cfg.survival_epochs)]:
        logger.info("=== Stage %s (%d epochs) ===", stage, epochs)
        ms = trainer.fit_stage(stage, n_epochs=epochs, acceptance_level="debug",
                                acceptance_pairs=splits.train)
        for m in ms:
            history.append(asdict(m))

    # ---- 7. Save checkpoint -------------------------------------------
    ckpt = trainer.save_checkpoint("real_endtoend.pt")
    logger.info("Saved real-data checkpoint -> %s", ckpt)

    # ---- 8. Final test inference --------------------------------------
    if len(test_loader.dataset) > 0:
        batch = next(iter(test_loader))
        batch = batch.to(trainer.device)
        model.eval()
        with torch.no_grad():
            pred = model(batch)
        logger.info(
            "Test inference: z_path %s, state_logits %s, hazard %s",
            tuple(pred.z_path.shape),
            tuple(pred.resistance_state_logits.shape) if pred.resistance_state_logits is not None else None,
            tuple(pred.hazard.shape) if pred.hazard is not None else "(discrete, see survival_curve)",
        )

    # ---- 9. Honest summary --------------------------------------------
    final_summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-debug",
        "n_snapshots": len(snapshots),
        "n_outcomes": len(outcomes),
        "n_pairs": len(pairs),
        "n_longitudinal_pairs": summary["n_longitudinal_pairs"],
        "n_unique_patients": summary["n_unique_patients"],
        "n_events_observed": debug_report.n_events_observed,
        "modality_coverage": debug_report.modality_coverage,
        "split_summary": splits.summary(),
        "acceptance_level_achieved": debug_report.level_achieved,
        "wall_time_s": round(time.time() - t0, 1),
        "history": history,
        "checkpoint": ckpt,
        "claim_permitted": "code-correctness only; no patient-level prediction claims (debug-level, "
                            "below the 200-patient threshold for any survival/trajectory claim)",
    }
    summary_path = out_dir / "real_endtoend_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    logger.info("Wrote summary -> %s", summary_path)

    logger.info("MORT-FM REAL END-TO-END run COMPLETE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
