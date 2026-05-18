#!/usr/bin/env python3
"""
scripts/mortfm_emit_registries.py
=================================
Emit the four MORT-FM registries from the *current* on-disk state of the
project. No computation; pure introspection.

Outputs:
  * data/processed/features/mortfm_feature_spaces.json
  * logs/mortfm/endpoint_registry.json
  * logs/mortfm/claim_gate_registry.json
  * logs/mortfm/artifact_registry.json

Each output also bears the current git short-SHA so downstream consumers
can check whether the registry was emitted from the same commit as the
checkpoints it describes.

This script is a thin orchestrator. All semantics live in
``resistancemap.mortfm.registries``; the block adapters in
``resistancemap.mortfm.blocks`` are responsible for translating each
on-disk artifact into an ArtifactRecord.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.blocks import (
    block_a_artifact_record,
    block_b_artifact_record,
    block_c_artifact_record,
    block_d_artifact_record,
    block_e_artifact_record,
)
from resistancemap.mortfm.blocks._common import git_head_short, utcnow_iso
from resistancemap.mortfm.registries.artifact_registry import ArtifactRegistry
from resistancemap.mortfm.registries.claim_gate_registry import write_claim_gate_registry
from resistancemap.mortfm.registries.endpoint_registry import write_endpoint_registry
from resistancemap.mortfm.registries.feature_space_registry import write_feature_space_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_emit_registries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-feature-spaces",
                    default="data/processed/features/mortfm_feature_spaces.json")
    ap.add_argument("--out-endpoints",
                    default="logs/mortfm/endpoint_registry.json")
    ap.add_argument("--out-claim-gates",
                    default="logs/mortfm/claim_gate_registry.json")
    ap.add_argument("--out-artifacts",
                    default="logs/mortfm/artifact_registry.json")
    args = ap.parse_args()

    write_feature_space_registry(out_path=args.out_feature_spaces)
    write_endpoint_registry(out_path=args.out_endpoints)
    write_claim_gate_registry(out_path=args.out_claim_gates)

    registry = ArtifactRegistry([
        block_a_artifact_record(),
        block_b_artifact_record(),
        block_c_artifact_record(),
        block_d_artifact_record(),
        block_e_artifact_record(),
    ])
    registry.save(args.out_artifacts)

    summary = {
        "emitted_at": utcnow_iso(),
        "git_commit": git_head_short(),
        "outputs": {
            "feature_spaces": args.out_feature_spaces,
            "endpoints": args.out_endpoints,
            "claim_gates": args.out_claim_gates,
            "artifacts": args.out_artifacts,
        },
        "artifact_statuses": {
            bid: rec.status for bid, rec in registry.items()
        },
        "claims_unlocked_union": registry.claims_unlocked(),
    }
    Path("logs/mortfm").mkdir(parents=True, exist_ok=True)
    with open("logs/mortfm/registry_emission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Registry emission summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
