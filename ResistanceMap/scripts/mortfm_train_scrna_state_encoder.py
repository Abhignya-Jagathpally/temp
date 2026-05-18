#!/usr/bin/env python3
"""
scripts/mortfm_train_scrna_state_encoder.py
===========================================
Block C — pre-train the cell-state RNA encoder on processed single-cell
AnnDatas (Scanpy QC + HVG already applied by
``scripts/mortfm_preprocess_scrna.py``).

Pipeline (all delegated, no inline ML):
  1. Read the scRNA manifest produced by mortfm_build_scrna_manifest.py.
  2. For each h5ad, pseudobulk per (patient_id, pseudotime_disease_stage)
     using :func:`resistancemap.data.scrna_loader.build_snapshots_from_h5ad`.
  3. Align every snapshot to a shared feature space (HVG union) using
     :func:`resistancemap.data.feature_alignment.align_to_reference_features`.
  4. Train MORT-FM with ``use_rna=True`` and every other modality off, stage A
     reconstruction only.

Honest constraints encoded by the run:
  * No calendar-time labels in either dataset (manifest's
    has_pseudotime_only=True for both). We therefore use only
    `pseudotime_disease_stage` as ordinal label and never claim wall-clock
    trajectories.
  * The output checkpoint contains *encoder + fusion* weights only — they
    are exported so Block E can splice them into the integrated model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.feature_alignment import align_to_reference_features
from resistancemap.data.scrna_loader import build_snapshots_from_h5ad
from resistancemap.data.trajectory_pair_builder import build_temporal_pairs, summarise_pairs
from resistancemap.mortfm.acceptance_gate import evaluate_cohort
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    ModalityName,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)
from resistancemap.mortfm.trainer import MORTFMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_train_scrna_state_encoder")


def _shared_hvg_union(h5ad_paths: Iterable[Path], top_per_dataset: int) -> list[str]:
    """Return the union of the top-``top_per_dataset`` HVGs across all h5ads."""
    import anndata as ad
    union: List[str] = []
    seen = set()
    for p in h5ad_paths:
        a = ad.read_h5ad(str(p), backed="r")
        if "highly_variable" in a.var.columns:
            ranking = a.var["means"] if "means" in a.var.columns else None
            hvg = a.var.index[a.var["highly_variable"].astype(bool)]
        else:
            hvg = a.var.index[: top_per_dataset]
        for g in list(hvg.astype(str))[: top_per_dataset]:
            if g not in seen:
                seen.add(g)
                union.append(g)
    return union


def _snapshots_aligned(
    h5ad_path: Path,
    *,
    reference_features: List[str],
    disease_tag: str,
) -> tuple[list[PatientCellSnapshot], list[ResistanceOutcome]]:
    snapshots = build_snapshots_from_h5ad(
        str(h5ad_path),
        patient_obs_col="patient_id",
        timepoint_obs_col="pseudotime_disease_stage",
        disease=disease_tag,
    )
    aligned: list[PatientCellSnapshot] = []
    outcomes: list[ResistanceOutcome] = []
    for s in snapshots:
        if s.rna is None:
            continue
        # ModalityTensor.values has shape (1, n_genes). Make a 1-row DataFrame
        # so we can reuse the same alignment primitive Block B uses.
        row = pd.DataFrame(
            s.rna.values.cpu().numpy(),
            index=[s.sample_id],
            columns=s.rna.feature_names,
        )
        a, _, _ = align_to_reference_features(row, reference_features)
        new_rna = ModalityTensor(
            name=ModalityName.RNA.value,
            values=torch.from_numpy(a.values.astype(np.float32)),
            feature_names=list(reference_features),
        )
        aligned.append(PatientCellSnapshot(
            patient_id=s.patient_id, sample_id=s.sample_id,
            disease=s.disease, timepoint=s.timepoint, rna=new_rna,
        ))
        # Pseudotime is *ordinal* — we encode it as drug_response surrogate
        # only when paired across timepoints in Block E. Block C is encoder
        # pre-training only, so we emit an unlabelled outcome.
        outcomes.append(ResistanceOutcome(
            patient_id=s.patient_id,
            baseline_time=0.0, event_time=None, censored=True,
            resistance_label=None, drug_response=None,
        ))
    return aligned, outcomes


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/single_cell/scrna_manifest.csv")
    ap.add_argument("--hvg-per-dataset", type=int, default=2000)
    ap.add_argument("--checkpoint-name", default="block_c_state_encoder.pt")
    ap.add_argument("--summary-out", default="logs/mortfm/block_c_summary.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-snapshots", type=int, default=None,
                    help="Optional: cap the number of pseudobulked snapshots for "
                         "quick smoke tests. None = use all.")
    args = ap.parse_args()

    Path("logs/mortfm").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/mortfm").mkdir(parents=True, exist_ok=True)

    # ---- 1. Load manifest --------------------------------------------
    manifest = pd.read_csv(args.manifest)
    if len(manifest) == 0:
        logger.error("Empty manifest at %s", args.manifest)
        return 1
    h5ad_paths = [Path(p) for p in manifest["h5ad_path"].tolist()]
    logger.info("Manifest: %d datasets, total cells %d",
                len(manifest), int(manifest["n_cells"].sum()))

    # ---- 2. Build the union-HVG reference --------------------------------
    ref_features = _shared_hvg_union(h5ad_paths, top_per_dataset=args.hvg_per_dataset)
    logger.info("Shared HVG union: %d genes (top-%d per dataset)",
                len(ref_features), args.hvg_per_dataset)

    # ---- 3. Build pseudobulked snapshots from every h5ad -----------------
    all_snaps, all_outs = [], []
    per_dataset_n = {}
    for _, row in manifest.iterrows():
        path = Path(row["h5ad_path"])
        disease_tag = str(row.get("disease_stages", "")).split(",")[0] or "MM"
        snaps, outs = _snapshots_aligned(
            path, reference_features=ref_features, disease_tag=disease_tag,
        )
        per_dataset_n[row["dataset_id"]] = len(snaps)
        all_snaps.extend(snaps); all_outs.extend(outs)
        logger.info("  %s -> %d pseudobulked snapshots", row["dataset_id"], len(snaps))
    if args.max_snapshots and len(all_snaps) > args.max_snapshots:
        all_snaps = all_snaps[: args.max_snapshots]
        all_outs = all_outs[: args.max_snapshots]

    pairs = build_temporal_pairs(
        all_snaps, all_outs,
        include_survival_only=False,
        include_drug_response_only=False,
        include_unlabelled=True,
    )
    summary = summarise_pairs(pairs)
    logger.info("scRNA pairs: %s", summary)

    debug = evaluate_cohort(pairs, level="debug")
    logger.info("Acceptance debug=%s", debug.passed)

    # ---- 4. Build model + trainer ----------------------------------------
    cfg = MORTFMConfig(
        rna_input_dim=len(ref_features),
        proteomics_input_dim=50, clinical_input_dim=4,
        d_token=32, d_latent=64, fusion_layers=2, fusion_heads=4, fusion_dropout=0.1,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=False,
        use_proteomics=False,
        n_time_grid=4, integration_time=6.0,
        pretrain_epochs=args.epochs, finetune_epochs=0,
        trajectory_epochs=0, survival_epochs=0,
        batch_size=32, lr=1e-4, weight_decay=1e-5,
        mixed_precision=False, drug_embed_dim=16,
        checkpoint_dir="checkpoints/mortfm",
        survival_n_bins=4,
    )
    splits, train_loader, val_loader, test_loader = make_data_module(
        pairs, batch_size=cfg.batch_size, val_fraction=0.15, test_fraction=0.15,
        seed=cfg.seed,
    )
    logger.info("Split summary: %s", splits.summary())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MORTFM(cfg, n_pathway_proteins=200, n_drug_candidates=11, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=device)

    # Block C is *self-supervised encoder pre-training*; it carries no clinical
    # claim, so we deliberately do NOT request an acceptance level here. The
    # cohort acceptance gate exists to bound *claim* runs, not unsupervised
    # representation learning. The encoder it produces is consumed by Block E
    # which is what actually triggers the cohort gates.
    history = []
    for h in trainer.fit_stage("A", n_epochs=cfg.pretrain_epochs):
        history.append(asdict(h))

    ckpt = trainer.save_checkpoint(args.checkpoint_name)
    logger.info("Saved Block-C state encoder -> %s", ckpt)

    out = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-block-c-state-encoder",
        "checkpoint": ckpt,
        "n_datasets": int(len(manifest)),
        "n_total_cells_in_manifest": int(manifest["n_cells"].sum()),
        "n_snapshots_per_dataset": per_dataset_n,
        "n_pseudobulked_snapshots_total": int(summary["n_pairs_total"]),
        "n_unique_patients": int(summary["n_unique_patients"]),
        "n_reference_hvgs": int(len(ref_features)),
        "all_datasets_pseudotime_only": bool(manifest["has_pseudotime_only"].all()),
        "calendar_time_available": bool(manifest["has_real_time_units"].any()),
        "trajectory_claim_allowed": False,
        "trajectory_claim_block_reason": "All input scRNA datasets carry ordinal pseudotime "
                                          "(disease stage) only; no calendar-time-aware "
                                          "trajectory claim is permitted.",
        "history": history,
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Wrote Block-C summary -> %s", args.summary_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
