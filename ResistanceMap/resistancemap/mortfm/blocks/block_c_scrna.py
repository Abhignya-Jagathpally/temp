"""
resistancemap/mortfm/blocks/block_c_scrna.py
============================================
Block C — single-cell state encoder adapter.

Block C operates on a different RNA feature space than Blocks A/B
(scrna_block_c_hvg_union_3535 vs rna_block_a_top2000). The integrator
keeps it as a separate pointer; this adapter advertises that fact.
"""

from __future__ import annotations

import logging
from pathlib import Path

from resistancemap.mortfm.blocks._common import (
    git_head_short, load_torch_payload, read_json, utcnow_iso,
)
from resistancemap.mortfm.registries.artifact_registry import ArtifactRecord

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_C_CHECKPOINT = "checkpoints/mortfm/block_c_state_encoder.pt"
DEFAULT_BLOCK_C_REPORT = "logs/mortfm/block_c_summary.json"


def load_block_c(
    checkpoint_path: str = DEFAULT_BLOCK_C_CHECKPOINT,
):
    payload = load_torch_payload(checkpoint_path)
    if payload is None:
        logger.warning("Block C checkpoint missing at %s", checkpoint_path)
        return None, None, None
    from resistancemap.mortfm.model import MORTFM
    from resistancemap.mortfm.schemas import MORTFMConfig

    cfg = MORTFMConfig(**payload["config"]) if isinstance(payload.get("config"), dict) else MORTFMConfig()
    model = MORTFM(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, cfg, payload


def block_c_artifact_record(
    checkpoint_path: str = DEFAULT_BLOCK_C_CHECKPOINT,
    report_path: str = DEFAULT_BLOCK_C_REPORT,
) -> ArtifactRecord:
    rep = read_json(report_path) or {}
    n_pseudobulks = int(rep.get("n_pseudobulked_snapshots_total", 0)) or 51
    n_ref = int(rep.get("n_reference_hvgs", 0)) or 3535
    allowed = ["technical", "single_cell_state"]
    blocked = [
        "longitudinal_trajectory", "resistance_emergence",
        "patient_level_clinical_prediction", "causal_mechanism",
        "static_drug_response", "hematologic_specimen_drug_response",
    ]
    return ArtifactRecord(
        block_id="C_scrna_state_encoder",
        artifact_type="checkpoint",
        path=str(Path(checkpoint_path).resolve() if Path(checkpoint_path).exists() else checkpoint_path),
        data_source=[
            "GSE124310 (26,321 cells × 20,805 genes)",
            "GSE271107 (125,676 cells × 29,511 genes)",
            "Total: 151,997 cells, 51 pseudobulked patient-stage snapshots",
        ],
        n_samples=n_pseudobulks,
        n_features=n_ref,
        feature_space_id="scrna_block_c_hvg_union_3535",
        endpoint_name="disease_stage_pseudotime",
        allowed_claims=allowed,
        blocked_claims=blocked,
        checkpoint_compatible=True,
        created_at=utcnow_iso(),
        git_commit=git_head_short(),
        gate_report_path=report_path if Path(report_path).exists() else None,
        notes=(
            "Both source datasets are pseudotime-only (has_real_time_units=False); "
            "trajectory claim is explicitly blocked. Encoder is self-supervised "
            "reconstruction; produces latents on a DIFFERENT feature axis than "
            "Block A/B — do not feed Block C inputs into Block A's encoder."
        ),
    )
