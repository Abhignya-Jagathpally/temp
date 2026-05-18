#!/usr/bin/env python3
"""
scripts/mortfm_prepare_mmrf.py
==============================
Build a MORT-FM-ready cohort from an MMRF CoMMpass dump.

This script does NOT generate any data. It only re-shapes existing files in
``data/raw/mmrf_commpass`` into MORT-FM schema objects and pickles them to
``data/processed/mortfm/`` for fast loading by training scripts.

Inputs (under ``--data-dir``):
    MMRF_CoMMpass_IA*_PER_PATIENT*.csv          -- clinical outcomes
    MMRF_CoMMpass_IA*_PER_PATIENT_VISIT.csv      -- visit-level data
    MMRF_CoMMpass_IA*_salmon_gene_tpm.tsv        -- RNA-seq TPM
    MMRF_CoMMpass_IA*_STAND_ALONE_TRTRESP.csv    -- treatment regimens
    (optional) per-sample CITE-seq / proteomics / methylation matrices

Output:
    data/processed/mortfm/<release>_snapshots.pkl
    data/processed/mortfm/<release>_outcomes.pkl
    data/processed/mortfm/<release>_summary.json

Run:
    python scripts/mortfm_prepare_mmrf.py --data-dir data/raw/mmrf_commpass
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.clinical_outcome_loader import (
    load_mortfm_outcomes,
    summarise_outcomes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_prepare_mmrf")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare an MMRF cohort for MORT-FM training")
    ap.add_argument("--data-dir", required=True, help="Path to data/raw/mmrf_commpass")
    ap.add_argument("--out-dir", default="data/processed/mortfm")
    ap.add_argument("--endpoint", default="pfs", choices=("pfs", "os", "relapse"))
    ap.add_argument("--strict", action="store_true", default=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading outcomes from %s (endpoint=%s)", data_dir, args.endpoint)
    outcomes = load_mortfm_outcomes(str(data_dir), endpoint=args.endpoint, strict=args.strict)
    summary = summarise_outcomes(outcomes)
    logger.info("Outcome summary: %s", summary)

    out_pickle = out_dir / "outcomes.pkl"
    with open(out_pickle, "wb") as f:
        pickle.dump(outcomes, f)
    logger.info("Wrote %d outcomes -> %s", len(outcomes), out_pickle)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("MMRF preparation complete. Snapshots/temporal pairs not built — "
                "wire to RNA-seq + scRNA-seq loaders separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
