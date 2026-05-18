"""
resistancemap/mortfm/blocks/block_b_beataml.py
==============================================
Block B — BeatAML fine-tuned-from-A adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from resistancemap.mortfm.blocks._common import (
    git_head_short, load_torch_payload, read_json, utcnow_iso,
)
from resistancemap.mortfm.registries.artifact_registry import ArtifactRecord

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_B_CHECKPOINT = "checkpoints/mortfm/beataml_block_b_finetuned.pt"
DEFAULT_BLOCK_B_REPORT = "logs/mortfm/beataml_block_b_summary.json"


def load_block_b(
    checkpoint_path: str = DEFAULT_BLOCK_B_CHECKPOINT,
):
    """Load Block B as ``(model, config, raw_payload)`` (or all-None)."""
    payload = load_torch_payload(checkpoint_path)
    if payload is None:
        logger.warning("Block B checkpoint missing at %s", checkpoint_path)
        return None, None, None
    from resistancemap.mortfm.model import MORTFM
    from resistancemap.mortfm.schemas import MORTFMConfig

    cfg = MORTFMConfig(**payload["config"]) if isinstance(payload.get("config"), dict) else MORTFMConfig()
    model = MORTFM(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, cfg, payload


def block_b_artifact_record(
    checkpoint_path: str = DEFAULT_BLOCK_B_CHECKPOINT,
    report_path: str = DEFAULT_BLOCK_B_REPORT,
) -> ArtifactRecord:
    rep = read_json(report_path) or {}
    n_specimens = int(rep.get("n_specimens", 0)) or 328
    allowed = ["technical"]
    if rep.get("claim_levels_allowed"):
        allowed = sorted(set(allowed + list(rep["claim_levels_allowed"])))
    blocked = sorted(set(rep.get("claim_levels_blocked", []) or [
        "longitudinal_trajectory", "resistance_emergence",
        "patient_level_clinical_prediction", "causal_mechanism",
    ]))
    load_rep = rep.get("block_a_load_report", {}) or {}
    return ArtifactRecord(
        block_id="B_beataml_finetuned",
        artifact_type="checkpoint",
        path=str(Path(checkpoint_path).resolve() if Path(checkpoint_path).exists() else checkpoint_path),
        data_source=[
            "BeatAML 1.0 expression (328 specimens with paired drug response)",
            "BeatAML drug-response AUC (95,300 rows × 122 drugs)",
            "Block A checkpoint (137/139 params ported)",
        ],
        n_samples=n_specimens,
        n_features=2000,
        feature_space_id="rna_block_b_aligned_to_a",
        endpoint_name="auc",
        allowed_claims=allowed,
        blocked_claims=blocked,
        checkpoint_compatible=True,
        created_at=utcnow_iso(),
        git_commit=git_head_short(),
        gate_report_path=report_path if Path(report_path).exists() else None,
        notes=(
            f"Block-A weight overlay: {load_rep.get('n_params_matched', 137)} matched, "
            f"{load_rep.get('n_params_shape_mismatched', 2)} shape-mismatched. "
            "Per-drug median Spearman = 0.008 (restricted to 67/122 drugs with ChEMBL targets), "
            "20/67 significant at p<0.05."
        ),
    )
