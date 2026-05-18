#!/usr/bin/env python3
"""
scripts/mortfm_ingest_crispr.py
================================
Ingest DepMap CRISPRGeneEffect.csv into a tidy parquet that the causal
counterfactual validator can read.

Input format (DepMap 23Q2): rows = ModelID, cols = "Gene (Entrez)".
The header is parsed to recover (gene_symbol, entrez_id) tuples; we
emit:

  * data/processed/crispr/gene_effect.parquet
      (model_id, gene_symbol, entrez_id, gene_effect_score)
  * data/processed/crispr/essentiality_summary.csv
      (gene_symbol, n_models, median_effect, frac_essential)

A gene is "essential" in a model if its effect score < -0.5 (DepMap's
common-essential threshold).

Output is structured so the PathwayCausalValidator can consume it without
re-parsing 380 MB on every call.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_crispr")

_HEADER_RE = re.compile(r"^(?P<symbol>[^()]+?)\s*\((?P<entrez>\d+)\)$")


def _parse_header_token(token: str) -> tuple[str, str]:
    m = _HEADER_RE.match(token.strip())
    if not m:
        return token.strip(), ""
    return m.group("symbol").strip(), m.group("entrez").strip()


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_public/depmap/CRISPRGeneEffect.csv")
    ap.add_argument("--out-parquet", default="data/processed/crispr/gene_effect.parquet")
    ap.add_argument("--out-summary", default="data/processed/crispr/essentiality_summary.csv")
    ap.add_argument("--out-report", default="logs/mortfm/crispr_ingest_report.json")
    ap.add_argument("--essential-threshold", type=float, default=-0.5)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        logger.error("CRISPR input not found at %s", src)
        return 1

    logger.info("Reading %s ...", src)
    df = pd.read_csv(src, index_col=0)
    logger.info("Loaded gene-effect: %d models x %d genes", df.shape[0], df.shape[1])

    # Parse the header to recover (symbol, entrez).
    cols = list(df.columns)
    parsed = [_parse_header_token(c) for c in cols]
    symbols = [p[0] for p in parsed]
    entrez = [p[1] for p in parsed]
    df.columns = symbols
    df.index.name = "model_id"

    # Long-format for the parquet (it's ~7M rows for 1k models x 18k genes —
    # write in tidy long format because the validator joins by (model, gene)).
    n_models = df.shape[0]
    n_genes = df.shape[1]
    logger.info("Melting to long format (~%d rows)", n_models * n_genes)
    long = df.reset_index().melt(
        id_vars="model_id", var_name="gene_symbol", value_name="gene_effect_score",
    )
    entrez_map = dict(zip(symbols, entrez))
    long["entrez_id"] = long["gene_symbol"].map(entrez_map)
    long = long.dropna(subset=["gene_effect_score"])
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.out_parquet, index=False)
    logger.info("Wrote tidy gene-effect parquet: %d rows -> %s",
                len(long), args.out_parquet)

    # Per-gene essentiality summary (used by the causal validator).
    grp = long.groupby("gene_symbol")["gene_effect_score"]
    summary = pd.DataFrame({
        "median_effect": grp.median(),
        "n_models": grp.count(),
        "n_essential": grp.apply(lambda x: int((x < args.essential_threshold).sum())),
    }).reset_index()
    summary["frac_essential"] = summary["n_essential"] / summary["n_models"].clip(lower=1)
    summary = summary.sort_values("median_effect")
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    # Honest summary.
    common_ess = summary[summary["frac_essential"] >= 0.9]
    report = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-crispr-ingest",
        "input": str(src.resolve()),
        "depmap_release": "23Q2 Public",
        "n_models": int(n_models),
        "n_genes": int(n_genes),
        "n_long_rows": int(len(long)),
        "essential_threshold": args.essential_threshold,
        "n_common_essentials_frac_ge_0_9": int(len(common_ess)),
        "top_essential_genes_by_median_effect": (
            summary.head(20)[["gene_symbol", "median_effect", "frac_essential"]]
            .to_dict("records")
        ),
        "outputs": {
            "tidy_parquet": args.out_parquet,
            "summary_csv": args.out_summary,
        },
        "unlocks_causal_mechanism_external_evidence": True,
        "honest_note": (
            "DepMap CRISPR gene-effect is the external-evidence channel the "
            "PathwayCausalValidator was waiting on. Counterfactual edge "
            "effects (Δ_e) can now be cross-checked against per-gene "
            "essentiality: an edge whose downstream gene has frac_essential "
            ">=0.9 is biologically plausible as a resistance driver. The "
            "validator will still REFUSE causal_mechanism unless the top-10 "
            "Δ edges separate from negative controls by >=2σ."
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("CRISPR ingest report -> %s", args.out_report)
    logger.info("Found %d common essentials (frac_essential >= 0.9)", len(common_ess))
    return 0


if __name__ == "__main__":
    sys.exit(main())
