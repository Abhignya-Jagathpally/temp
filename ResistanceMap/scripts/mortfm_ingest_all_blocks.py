#!/usr/bin/env python3
"""
scripts/mortfm_ingest_all_blocks.py
===================================
Run all Block A/B/C/D ingestion scripts in order.

Each per-source script is invoked as a subprocess so that one missing source
does not abort the others — the function returns a summary dict naming which
blocks succeeded vs. failed (with the FileNotFoundError messages preserved).

Populate ``data/raw_public/`` first::

    python3 scripts/mortfm_download_public_data.py --link-legacy

See also ``docs/MORTFM_PUBLIC_DATA.md``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_all")


_STEPS = [
    ("DepMap (Block A)", ["scripts/mortfm_ingest_depmap.py"]),
    ("GDSC (Block A)", ["scripts/mortfm_ingest_gdsc.py"]),
    ("PRISM (Block A)", ["scripts/mortfm_ingest_prism.py"]),
    ("STRING (Block A)", ["scripts/mortfm_ingest_string.py"]),
    ("UniProt (Block A)", ["scripts/mortfm_ingest_uniprot.py"]),
    ("Reactome (Block A)", ["scripts/mortfm_ingest_reactome.py"]),
    ("ChEMBL (Block A)", ["scripts/mortfm_ingest_chembl.py"]),
    ("Biological graph (Block D)", ["scripts/mortfm_build_biological_graph.py"]),
    ("Beat AML (Block B)", ["scripts/mortfm_ingest_beataml.py"]),
    ("GEO single-cell (Block C)", ["scripts/mortfm_ingest_geo_singlecell.py"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-base", default="data/raw_public")
    ap.add_argument("--out-base", default="data/processed")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary: dict = {"ok": [], "skipped": [], "failed": []}
    for label, cmd in _STEPS:
        full_cmd = [sys.executable, str(repo_root / cmd[0])]
        logger.info("=== %s -> %s ===", label, " ".join(full_cmd))
        try:
            subprocess.run(full_cmd, check=True)
            summary["ok"].append(label)
        except subprocess.CalledProcessError as exc:
            logger.warning("%s FAILED (exit %d) -- continuing", label, exc.returncode)
            summary["failed"].append(label)
            if not args.continue_on_error:
                return exc.returncode
    logger.info("=== Ingestion summary ===")
    for k, v in summary.items():
        logger.info("  %s (%d): %s", k, len(v), v)
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
