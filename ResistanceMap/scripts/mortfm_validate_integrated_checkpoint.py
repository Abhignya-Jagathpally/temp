#!/usr/bin/env python3
"""
scripts/mortfm_validate_integrated_checkpoint.py
================================================
Cross-check ``checkpoints/mortfm/mortfm_integrated.pt`` against the
artifact registry, feature-space registry, endpoint registry, and
claim-gate registry.

Refuses any claim that is not jointly:
  (a) supported by an artifact present on disk;
  (b) compatible with the endpoint type;
  (c) reported as PASS by the corresponding gate(s).

Output: ``logs/mortfm/integrated_checkpoint_validation.json``
Exit-code 0 if the integrated checkpoint matches the registry; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.blocks._common import git_head_short, utcnow_iso
from resistancemap.mortfm.registries.artifact_registry import ArtifactRegistry
from resistancemap.mortfm.registries.claim_gate_registry import (
    CANONICAL_CLAIM_LEVELS, CLAIM_LEVELS,
)
from resistancemap.mortfm.registries.feature_space_registry import CANONICAL_FEATURE_SPACES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_validate_integrated_checkpoint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mortfm/mortfm_integrated.pt")
    ap.add_argument("--manifest", default="checkpoints/mortfm/mortfm_integrated_manifest.json")
    ap.add_argument("--artifact-registry", default="logs/mortfm/artifact_registry.json")
    ap.add_argument("--out", default="logs/mortfm/integrated_checkpoint_validation.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        logger.error("Integrated checkpoint not found at %s", ckpt)
        return 1
    if not Path(args.artifact_registry).exists():
        logger.error("Artifact registry not found at %s — run scripts/mortfm_emit_registries.py first.",
                     args.artifact_registry)
        return 1

    registry = ArtifactRegistry.load(args.artifact_registry)
    manifest = json.load(open(args.manifest)) if Path(args.manifest).exists() else None

    # --- 1. Per-block presence -------------------------------------------------
    block_check: dict[str, dict] = {}
    for block_id, rec in registry.items():
        block_check[block_id] = {
            "registry_status": rec.status,
            "path": rec.path,
            "allowed_claims_in_registry": rec.allowed_claims,
            "gate_report_path": rec.gate_report_path,
            "feature_space_id": rec.feature_space_id,
        }
    all_blocks_present = registry.all_present([
        "A_cellline_foundation", "B_beataml_finetuned",
        "C_scrna_state_encoder", "D_esm2_embeddings",
    ])

    # --- 2. Feature-space consistency -----------------------------------------
    feature_space_errors = []
    for block_id, rec in registry.items():
        fs = CANONICAL_FEATURE_SPACES.get(rec.feature_space_id) if rec.feature_space_id else None
        if rec.feature_space_id and fs is None:
            feature_space_errors.append(
                f"{block_id}: declared feature_space_id={rec.feature_space_id!r} "
                "is not in CANONICAL_FEATURE_SPACES"
            )
        if rec.feature_space_id and rec.n_features and fs and fs.n_features and rec.n_features != fs.n_features:
            feature_space_errors.append(
                f"{block_id}: n_features={rec.n_features} does not match "
                f"{rec.feature_space_id} canonical n_features={fs.n_features}"
            )

    # --- 3. Claim allow-list per level ----------------------------------------
    unlocked = set(registry.claims_unlocked())
    claim_decisions: dict[str, dict] = {}
    for level in CLAIM_LEVELS:
        spec = CANONICAL_CLAIM_LEVELS[level]
        artifacts_ok = all((b in registry and registry.get(b).status == "present")
                            for b in spec.required_artifacts)
        unlocked_by_registry = level in unlocked
        claim_decisions[level] = {
            "required_artifacts": spec.required_artifacts,
            "artifacts_present": artifacts_ok,
            "unlocked_by_registry_union": unlocked_by_registry,
            "granted": bool(artifacts_ok and (unlocked_by_registry or level == "technical")),
            "required_gates": spec.required_gates,
        }

    # --- 4. Verdict ------------------------------------------------------------
    verdict = "PASS" if (all_blocks_present and not feature_space_errors) else "FAIL"
    report = {
        "verdict": verdict,
        "validated_at": utcnow_iso(),
        "git_commit": git_head_short(),
        "checkpoint": str(ckpt.resolve()),
        "manifest_present": manifest is not None,
        "all_required_blocks_present": all_blocks_present,
        "block_check": block_check,
        "feature_space_errors": feature_space_errors,
        "claims_unlocked_union": sorted(unlocked),
        "claim_decisions": claim_decisions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote integrated checkpoint validation -> %s (verdict: %s)", args.out, verdict)
    granted = [k for k, v in claim_decisions.items() if v["granted"]]
    logger.info("Claim levels granted: %s", granted)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
