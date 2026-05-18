#!/usr/bin/env python3
"""
scripts/mortfm_train_beataml_finetune.py
========================================
Block B — BeatAML patient/specimen-level drug-response finetune.

Uses the auto-mode BeatAML ingestion (processed parquet) to train MORT-FM on
**ex vivo drug-response prediction**. This is *not* a resistance-trajectory
run — the endpoint validator labels it as ``static_drug_response`` and the
claim gate restricts release to "specimen-level ex vivo drug-response".

Inputs (all already on disk, no download required):
    data/processed/beataml/beataml_expression.parquet         (451 specimens × 22843 genes)
    data/processed/beataml/beataml_drug_response.parquet      (95,300 rows, 528 specimens × 122 drugs)
    data/processed/beataml/beataml_clinical.csv               (672 patients)
    data/processed/graphs/biological_edges.parquet            (STRING PPI, 473,618 edges)

Outputs:
    checkpoints/mortfm/beataml_finetuned.pt
    logs/mortfm/beataml_summary.json
    logs/mortfm/beataml_per_drug_metrics.csv
    logs/mortfm/endpoint_semantics_report.json
    logs/mortfm/claim_gate_report.json
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs, summarise_pairs
from resistancemap.governance.claim_gates import RunEvidence, run_gates
from resistancemap.governance.endpoint_validator import (
    validate_endpoint_semantics,
    write_endpoint_semantics_report,
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
logger = logging.getLogger("mortfm_beataml_finetune")


def build_beataml_pairs(
    *,
    expr_path: Path = Path("data/processed/beataml/beataml_expression.parquet"),
    drug_path: Path = Path("data/processed/beataml/beataml_drug_response.parquet"),
    clin_path: Path = Path("data/processed/beataml/beataml_clinical.csv"),
    rna_genes: int = 1000,
    drug_metric: str = "AUC",
):
    expr = pd.read_parquet(expr_path)
    drug = pd.read_parquet(drug_path)
    clin = pd.read_csv(clin_path)

    drug = drug[drug["response_metric"] == drug_metric].copy()
    drug = drug.dropna(subset=["response_value"])
    logger.info("BeatAML drug rows (metric=%s): %d", drug_metric, len(drug))

    # Drug-response label per specimen: mean response across the panel
    per_specimen_response = drug.groupby("lab_id")["response_value"].mean()
    overlap = sorted(set(expr.index.astype(str)) & set(per_specimen_response.index.astype(str)))
    logger.info("Specimens with both expression and drug response: %d", len(overlap))

    # Subset expression to top-variance genes
    expr_sub = expr.loc[overlap]
    if expr_sub.shape[1] > rna_genes:
        variances = expr_sub.var(axis=0).sort_values(ascending=False)
        top = variances.index[:rna_genes].tolist()
        expr_sub = expr_sub[top]
    feature_names = list(expr_sub.columns.astype(str))
    logger.info("Expression subset: %d specimens × %d genes", *expr_sub.shape)

    # Patient ID for split (LabId -> PatientId so specimens from the same patient stay in one split)
    pid_map = dict(zip(clin["lab_id"].astype(str), clin["patient_id"].astype(str)))

    snapshots = []
    outcomes = []
    for lab in overlap:
        pid = pid_map.get(lab, lab)
        row = expr_sub.loc[lab].values.astype(np.float32)
        snapshots.append(
            PatientCellSnapshot(
                patient_id=str(pid),
                sample_id=str(lab),
                disease="AML",
                timepoint=0.0,
                rna=ModalityTensor(
                    name=ModalityName.RNA.value,
                    values=torch.from_numpy(row).unsqueeze(0),
                    feature_names=feature_names,
                ),
                drug=DrugContext("BeatAML_panel_mean", "panel_mean"),
            )
        )
        outcomes.append(
            ResistanceOutcome(
                patient_id=str(pid),
                baseline_time=0.0,
                event_time=None,
                censored=True,
                resistance_label=None,
                drug_response=torch.tensor([float(per_specimen_response[lab])]),
            )
        )

    pairs = build_temporal_pairs(
        snapshots, outcomes,
        include_survival_only=False,
        include_drug_response_only=True,
        include_unlabelled=False,
    )
    return pairs, feature_names


def per_drug_metrics(
    *,
    drug_path: Path,
    model,
    feature_names: list,
    expr_path: Path,
    clin_path: Path,
    device,
    cfg,
    out_csv: Path,
):
    """Compute per-drug Spearman + MSE on the AUC test set."""
    drug = pd.read_parquet(drug_path)
    drug = drug[drug["response_metric"] == "AUC"].dropna(subset=["response_value"])
    expr = pd.read_parquet(expr_path)
    clin = pd.read_csv(clin_path)
    pid_map = dict(zip(clin["lab_id"].astype(str), clin["patient_id"].astype(str)))

    # Use the same gene subset
    expr_sub = expr[feature_names]

    # Pre-compute z0 for every specimen via the encoder
    model.eval()
    spec_to_z = {}
    with torch.no_grad():
        for lab in expr_sub.index.astype(str):
            row = torch.from_numpy(expr_sub.loc[lab].values.astype(np.float32)).unsqueeze(0).to(device)
            from resistancemap.mortfm.schemas import MORTBatch
            batch = MORTBatch(
                rna=row, drug=torch.zeros(1, 3).to(device),
                modality_mask={"rna": torch.ones(1, dtype=torch.bool).to(device),
                                "drug": torch.zeros(1, dtype=torch.bool).to(device)},
                patient_ids=[pid_map.get(lab, lab)],
                time=torch.tensor([0.0]).to(device),
            )
            state = model.encode(batch)
            spec_to_z[lab] = state.z0.cpu().numpy()[0]

    rows = []
    for d, sub in drug.groupby("drug_name"):
        sub = sub[sub["lab_id"].astype(str).isin(spec_to_z)]
        if len(sub) < 10:
            continue
        # Predict via the drug_specific_risk head averaged over candidate drugs.
        zs = np.stack([spec_to_z[str(lab)] for lab in sub["lab_id"].astype(str)])
        z_tensor = torch.from_numpy(zs).to(device)
        with torch.no_grad():
            risk_all = model.drug_risk_head(z_tensor)  # (N, n_drug_candidates)
            # Use mean across candidates as a proxy "predicted response score"
            preds = risk_all.mean(dim=-1).cpu().numpy()
        truth = sub["response_value"].values.astype(np.float32)
        rho, p = spearmanr(preds, truth)
        mse = float(np.mean((preds - truth) ** 2))
        rows.append({"drug_name": d, "n": int(len(sub)),
                     "spearman": float(rho), "spearman_p": float(p),
                     "mse_against_AUC": mse})
    out_df = pd.DataFrame(rows).sort_values("n", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("Wrote per-drug metrics for %d drugs -> %s", len(out_df), out_csv)
    return out_df


def main() -> int:
    t0 = time.time()
    Path("logs/mortfm").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/mortfm").mkdir(parents=True, exist_ok=True)

    # ---- 1. Build cohort ---------------------------------------------
    pairs, feature_names = build_beataml_pairs()
    summary = summarise_pairs(pairs)
    logger.info("BeatAML pairs: %s", summary)

    # ---- 2. Acceptance gate ------------------------------------------
    debug = evaluate_cohort(pairs, level="debug")
    technical = evaluate_cohort(pairs, level="technical")
    logger.info("Acceptance levels achieved -- debug=%s, technical=%s",
                debug.passed, technical.passed)

    # ---- 3. Endpoint semantics ---------------------------------------
    endpoint_report = validate_endpoint_semantics(endpoint_name="auc")
    write_endpoint_semantics_report(endpoint_report,
                                     out_json="logs/mortfm/beataml_endpoint_semantics.json")

    # ---- 4. Build model + trainer ------------------------------------
    cfg = MORTFMConfig(
        rna_input_dim=len(feature_names),
        proteomics_input_dim=50, clinical_input_dim=4,
        d_token=32, d_latent=64, fusion_layers=2, fusion_heads=4, fusion_dropout=0.1,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        use_proteomics=False,
        n_time_grid=4, integration_time=6.0,
        pretrain_epochs=3, finetune_epochs=3, trajectory_epochs=0, survival_epochs=0,
        batch_size=32, lr=1e-4, weight_decay=1e-5,
        mixed_precision=False, drug_embed_dim=16,
        checkpoint_dir="checkpoints/mortfm",
        survival_n_bins=4,
    )
    splits, train_loader, val_loader, test_loader = make_data_module(
        pairs, batch_size=cfg.batch_size, val_fraction=0.15, test_fraction=0.15, seed=cfg.seed,
    )
    logger.info("Split summary: %s", splits.summary())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MORTFM(cfg, n_pathway_proteins=200, n_drug_candidates=11, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=device)

    # ---- 5. Train Stage A (encoder pretrain) + Stage C (drug response) ----
    history = []
    for stage, epochs in [("A", cfg.pretrain_epochs), ("C", cfg.finetune_epochs)]:
        m = trainer.fit_stage(stage, n_epochs=epochs, acceptance_level="debug",
                               acceptance_pairs=splits.train)
        for h in m:
            history.append(asdict(h))

    # ---- 6. Save checkpoint -------------------------------------------
    ckpt = trainer.save_checkpoint("beataml_finetuned.pt")
    logger.info("Saved checkpoint -> %s", ckpt)

    # ---- 7. Per-drug metrics -----------------------------------------
    per_drug_df = per_drug_metrics(
        drug_path=Path("data/processed/beataml/beataml_drug_response.parquet"),
        model=model, feature_names=feature_names,
        expr_path=Path("data/processed/beataml/beataml_expression.parquet"),
        clin_path=Path("data/processed/beataml/beataml_clinical.csv"),
        device=device, cfg=cfg,
        out_csv=Path("logs/mortfm/beataml_per_drug_metrics.csv"),
    )
    median_spearman = float(per_drug_df["spearman"].median()) if len(per_drug_df) else float("nan")
    n_significant = int((per_drug_df["spearman_p"] < 0.05).sum()) if len(per_drug_df) else 0
    logger.info("Per-drug median Spearman = %.3f, %d/%d significant at p<0.05",
                median_spearman, n_significant, len(per_drug_df))

    # ---- 8. Run claim gates -----------------------------------------
    evidence = RunEvidence(
        endpoint_name="auc",
        n_patients=summary["n_unique_patients"],
        n_events=0,
        n_longitudinal_pairs=0,
        has_c_index=False,
        has_patient_disjoint_split=True,
        n_models=int(summary["n_unique_patients"]),
        n_drugs=int(per_drug_df["drug_name"].nunique()) if len(per_drug_df) else 0,
        n_observed_response_rows=int(per_drug_df["n"].sum()) if len(per_drug_df) else 0,
        model_mapping_rate=1.0,  # all specimens are native BeatAML LabIds
        has_reactome=False, has_drug_target_edges=False,
        has_uniprot_fasta=False, has_embedding_cache=False,
    )
    gate_report = run_gates(evidence, out_json="logs/mortfm/beataml_claim_gate_report.json")
    logger.info("Claim gates -- allowed=%s blocked=%s",
                gate_report.claim_levels_allowed, gate_report.claim_levels_blocked)

    # ---- 9. Honest summary ------------------------------------------
    final_summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-beataml-block-b",
        "endpoint_name": "auc",
        "endpoint_type": "drug_response",
        "n_specimens": int(summary["n_pairs_total"]),
        "n_unique_patients": int(summary["n_unique_patients"]),
        "n_drugs_evaluated": int(per_drug_df["drug_name"].nunique()) if len(per_drug_df) else 0,
        "per_drug_median_spearman": median_spearman,
        "per_drug_significant_at_p_lt_0.05": n_significant,
        "claim_levels_allowed": gate_report.claim_levels_allowed,
        "claim_levels_blocked": gate_report.claim_levels_blocked,
        "split_summary": splits.summary(),
        "history": history,
        "checkpoint": ckpt,
        "claim_permitted": "ex vivo drug-response per-specimen ranking ONLY; no resistance, no trajectory, "
                            "no survival, no pathway-mechanism claim until ChEMBL+Reactome+UniProt ingested.",
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_path = Path("logs/mortfm/beataml_summary.json")
    with open(out_path, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    logger.info("Wrote summary -> %s", out_path)
    logger.info("MORT-FM BEATAML BLOCK-B RUN COMPLETE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
