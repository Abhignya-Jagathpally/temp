#!/usr/bin/env python3
"""
scripts/mortfm_build_temporal_pairs.py
======================================
Build :class:`TemporalTrainingPair` objects from snapshot + outcome pickles
produced by the modality-specific preparation scripts.

This script does no science — it composes pickled inputs into the canonical
training substrate.

Run:
    python scripts/mortfm_build_temporal_pairs.py \\
        --snapshots data/processed/mortfm/snapshots.pkl \\
        --outcomes data/processed/mortfm/outcomes.pkl \\
        --out data/processed/mortfm/pairs.pkl \\
        --allowed-gaps 3 6 12
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.trajectory_pair_builder import (
    build_temporal_pairs,
    summarise_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_build_pairs")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MORT-FM temporal training pairs")
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allowed-gaps", type=float, nargs="*", default=None,
                    help="Allowed timepoint gaps in months; default = any positive")
    ap.add_argument("--gap-tolerance", type=float, default=1.0)
    ap.add_argument("--outcome-tolerance", type=float, default=1.0)
    ap.add_argument("--include-unlabelled", action="store_true")
    args = ap.parse_args()

    with open(args.snapshots, "rb") as f:
        snapshots = pickle.load(f)
    with open(args.outcomes, "rb") as f:
        outcomes = pickle.load(f)
    logger.info("Loaded %d snapshots, %d outcomes", len(snapshots), len(outcomes))

    pairs = build_temporal_pairs(
        snapshots, outcomes,
        allowed_time_gaps=args.allowed_gaps,
        gap_tolerance=args.gap_tolerance,
        outcome_match_tolerance=args.outcome_tolerance,
        include_unlabelled=args.include_unlabelled,
    )
    logger.info("Built pairs: %s", summarise_pairs(pairs))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pairs, f)
    logger.info("Wrote %d pairs -> %s", len(pairs), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
