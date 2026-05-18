"""
resistancemap/mortfm/blocks/block_a_cellline.py
===============================================
Block A — cell-line foundation adapter.

Wraps ``checkpoints/mortfm/block_a_cellline_foundation.pt`` (real run on
DepMap × GDSC × STRING × Reactome × ChEMBL) into a loader + artifact record.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

from resistancemap.mortfm.blocks._common import (
    git_head_short, load_torch_payload, read_json, utcnow_iso,
)
from resistancemap.mortfm.registries.artifact_registry import ArtifactRecord

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_A_CHECKPOINT = "checkpoints/mortfm/block_a_cellline_foundation.pt"
DEFAULT_BLOCK_A_REPORT = "logs/mortfm/block_a_cellline_summary.json"


def load_block_a(
    checkpoint_path: str = DEFAULT_BLOCK_A_CHECKPOINT,
):
    """Load Block A as ``(model, config, raw_payload)``.

    Returns ``(None, None, None)`` if the checkpoint is missing — callers
    decide whether to raise.
    """
    payload = load_torch_payload(checkpoint_path)
    if payload is None:
        logger.warning("Block A checkpoint missing at %s", checkpoint_path)
        return None, None, None
    # Late import so the registries are usable without torch installed.
    from resistancemap.mortfm.model import MORTFM
    from resistancemap.mortfm.schemas import MORTFMConfig

    cfg = MORTFMConfig(**payload["config"]) if isinstance(payload.get("config"), dict) else MORTFMConfig()
    model = MORTFM(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, cfg, payload


def block_a_artifact_record(
    checkpoint_path: str = DEFAULT_BLOCK_A_CHECKPOINT,
    report_path: str = DEFAULT_BLOCK_A_REPORT,
) -> ArtifactRecord:
    payload = load_torch_payload(checkpoint_path)
    n_features = None
    if payload is not None and isinstance(payload.get("config"), dict):
        n_features = payload["config"].get("rna_input_dim")
    rep = read_json(report_path) or {}
    return ArtifactRecord(
        block_id="A_cellline_foundation",
        artifact_type="checkpoint",
        path=str(Path(checkpoint_path).resolve() if Path(checkpoint_path).exists() else checkpoint_path),
        data_source=[
            "DepMap RNA 24Q2 (1775 cell-lines × 19,215 genes)",
            "GDSC ln_IC50 (241,578 rows, 716 overlap × 286 drugs)",
            "STRING v12 PPI (473,618 edges)",
            "Reactome pathways (156,329 edges)",
            "ChEMBL drug-targets (6,577 edges)",
        ],
        n_samples=int(rep.get("n_train_cell_lines", 0) + rep.get("n_val_cell_lines", 0) + rep.get("n_test_cell_lines", 0)) or 716,
        n_features=n_features,
        feature_space_id="rna_block_a_top2000",
        endpoint_name="ln_ic50",
        allowed_claims=["technical", "static_drug_response"],
        blocked_claims=[
            "patient_level_clinical_prediction", "longitudinal_trajectory",
            "resistance_emergence", "causal_mechanism",
        ],
        checkpoint_compatible=True,
        created_at=utcnow_iso(),
        git_commit=git_head_short(),
        gate_report_path=report_path if Path(report_path).exists() else None,
        notes=(
            "Per-drug median Spearman = 0.191 over 286 GDSC drugs; "
            "124/286 sig at p<0.05; static_drug_response gate PASS."
        ),
    )
