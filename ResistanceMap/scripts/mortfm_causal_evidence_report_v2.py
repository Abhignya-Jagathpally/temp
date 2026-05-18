#!/usr/bin/env python3
"""
scripts/mortfm_causal_evidence_report_v2.py
=============================================
v17.8 — composes three evidence joiners (CRISPR, drug-target, Reactome
pathway) into a single per-edge evidence report PLUS the top-k pathway
enrichment permutation test.

The v16 report had only the CRISPR check. v17.8 adds:
  * drug-target either-endpoint support
  * Reactome pathway count per target HGNC
  * empirical pathway-enrichment p-value over k random null draws
  * strict v17 causal_mechanism gate that demands >=2 essentials AND
    >=2 drug-target-supported edges AND enriched pathway p<0.05

Inputs:
  results/mortfm/causal_edge_effects.csv (200 rows; produced by
                                          mortfm_causal_smoke_test.py)

Outputs:
  results/mortfm/causal_edge_evidence_v2.csv
  logs/mortfm/causal_evidence_v2_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.causal import build_edge_evidence_report
from resistancemap.mortfm.graph import IDHarmonizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_causal_evidence_report_v2")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="results/mortfm/causal_edge_effects.csv")
    ap.add_argument("--out-csv", default="results/mortfm/causal_edge_evidence_v2.csv")
    ap.add_argument("--out-summary", default="logs/mortfm/causal_evidence_v2_summary.json")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    edges = pd.read_csv(args.edges)
    # Bridge to HGNC (matches the v17.1 fix).
    h = IDHarmonizer.build_or_load()
    edges[["source_raw", "target_raw", "edge_type"]] = (
        edges["edge_id"].astype(str).str.split("::", n=2, expand=True)
    )
    edges["source_hgnc"] = h.any_alias_to_hgnc(edges["source_raw"])
    edges["target_hgnc"] = h.any_alias_to_hgnc(edges["target_raw"])

    report = build_edge_evidence_report(edges, top_k=args.top_k)
    table = report["table"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    logger.info("Wrote %d-row v2 evidence table -> %s", len(table), args.out_csv)

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-causal-evidence-v2",
        **report["summary"],
        "honest_note": (
            "v17.8 — three-way evidence join (CRISPR essentiality + ChEMBL "
            "drug-target either-endpoint + Reactome pathway count) with "
            "permutation-test enrichment. v17 gate refuses causal_mechanism "
            "unless ALL three lines of external evidence agree, with the "
            "pathway-enrichment p-value < 0.05."
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("v2 summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
