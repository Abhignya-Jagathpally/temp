"""
resistancemap/mortfm/blocks/block_e_integrated.py
=================================================
Block E — integrated checkpoint adapter.

Block E doesn't add new training. It is the manifest-tagged bundle of
Blocks A+B+C+D. The adapter exposes the merged state-dict AND the
provenance manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path

from resistancemap.mortfm.blocks._common import (
    git_head_short, load_torch_payload, read_json, utcnow_iso,
)
from resistancemap.mortfm.registries.artifact_registry import ArtifactRecord

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_E_CHECKPOINT = "checkpoints/mortfm/mortfm_integrated.pt"
DEFAULT_BLOCK_E_MANIFEST = "checkpoints/mortfm/mortfm_integrated_manifest.json"


def load_block_e(
    checkpoint_path: str = DEFAULT_BLOCK_E_CHECKPOINT,
    manifest_path: str = DEFAULT_BLOCK_E_MANIFEST,
):
    """Load Block E as ``(merged_state_dict, config, manifest_dict)``."""
    payload = load_torch_payload(checkpoint_path)
    if payload is None:
        logger.warning("Block E checkpoint missing at %s", checkpoint_path)
        return None, None, None
    manifest = read_json(manifest_path)
    return payload.get("model_state_dict"), payload.get("config"), manifest


def block_e_artifact_record(
    checkpoint_path: str = DEFAULT_BLOCK_E_CHECKPOINT,
    manifest_path: str = DEFAULT_BLOCK_E_MANIFEST,
) -> ArtifactRecord:
    manifest = read_json(manifest_path) or {}
    blocks = manifest.get("blocks", {}) or {}
    block_statuses = {k: v.get("status") for k, v in blocks.items()}
    overlay = manifest.get("overlay", {}) or {}
    # Block E inherits the union of allowed_claims from its constituents.
    inherited_allowed: set[str] = set()
    for blk in blocks.values():
        rep = (blk or {}).get("report") or {}
        for c in rep.get("claim_levels_allowed", []) or []:
            inherited_allowed.add(c)
    # Block E itself does NOT introduce any new claim — only relays.
    return ArtifactRecord(
        block_id="E_integrated",
        artifact_type="checkpoint",
        path=str(Path(checkpoint_path).resolve() if Path(checkpoint_path).exists() else checkpoint_path),
        data_source=[
            f"Block A: {block_statuses.get('A_cellline_foundation', 'unknown')}",
            f"Block B: {block_statuses.get('B_beataml_finetuned', 'unknown')}",
            f"Block C: {block_statuses.get('C_scrna_state_encoder', 'unknown')}",
            f"Block D: {block_statuses.get('D_esm2_embeddings', 'unknown')}",
        ],
        n_samples=None,
        n_features=None,
        feature_space_id="rna_block_a_top2000",  # the primary axis Blocks A+B share
        allowed_claims=sorted(inherited_allowed | {"technical"}),
        blocked_claims=[
            "longitudinal_trajectory", "resistance_emergence", "causal_mechanism",
            "patient_level_clinical_prediction",
        ],
        checkpoint_compatible=True,
        created_at=utcnow_iso(),
        git_commit=git_head_short(),
        gate_report_path=manifest_path if Path(manifest_path).exists() else None,
        notes=(
            f"B-on-A overlay: {overlay.get('n_keys_overlayed_b_on_a', 0)} keys, "
            f"{overlay.get('n_shape_mismatches', 0)} shape mismatches. "
            "Block C is referenced as a separate pointer (different feature axis)."
        ),
    )
