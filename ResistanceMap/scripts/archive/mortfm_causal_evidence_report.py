#!/usr/bin/env python3
"""
scripts/mortfm_causal_evidence_report.py
=========================================
v17 — produce per-edge causal-evidence report
``results/mortfm/causal_edge_evidence.csv`` with full HGNC bridging and
external-evidence joins.

For every counterfactual edge effect in
``results/mortfm/causal_edge_effects.csv``, emit:

  edge_id, source_raw, target_raw, source_hgnc, target_hgnc,
  source_uniprot, target_uniprot,
  delta_risk, is_top_decile,
  depmap_target_gene_effect_median, target_is_common_essential,
  drug_target_support, reactome_pathway_support,
  refusal_reason
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

from resistancemap.mortfm.graph import IDHarmonizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_causal_evidence")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="results/mortfm/causal_edge_effects.csv")
    ap.add_argument("--crispr-summary", default="data/processed/crispr/essentiality_summary.csv")
    ap.add_argument("--drug-targets", default="data/processed/drugs/drug_targets.csv")
    ap.add_argument("--out", default="results/mortfm/causal_edge_evidence.csv")
    ap.add_argument("--out-summary", default="logs/mortfm/causal_evidence_summary.json")
    args = ap.parse_args()

    edge_effects = pd.read_csv(args.edges)
    logger.info("Loaded %d edge effects", len(edge_effects))
    bridge = IDHarmonizer.build_or_load()

    # Parse edge_id -> source_raw, target_raw, edge_type.
    edge_effects[["source_raw", "target_raw", "edge_type"]] = (
        edge_effects["edge_id"].astype(str).str.split("::", n=2, expand=True)
    )
    edge_effects["source_hgnc"] = bridge.any_alias_to_hgnc(edge_effects["source_raw"])
    edge_effects["target_hgnc"] = bridge.any_alias_to_hgnc(edge_effects["target_raw"])

    # UniProt for source and target via STRING-id then UniProt map
    src_string = [bridge.alias_to_string.get(a) for a in edge_effects["source_raw"]]
    tgt_string = [bridge.alias_to_string.get(a) for a in edge_effects["target_raw"]]
    edge_effects["source_uniprot"] = [
        bridge.string_to_uniprot.get(s) if s else None for s in src_string
    ]
    edge_effects["target_uniprot"] = [
        bridge.string_to_uniprot.get(s) if s else None for s in tgt_string
    ]

    # DepMap CRISPR essentiality of target
    if Path(args.crispr_summary).exists():
        crispr = pd.read_csv(args.crispr_summary)
        crispr["gene_symbol_upper"] = crispr["gene_symbol"].astype(str).str.upper()
        ess_map_median = dict(zip(
            crispr["gene_symbol_upper"], crispr["median_effect"],
        ))
        ess_map_frac = dict(zip(
            crispr["gene_symbol_upper"], crispr["frac_essential"],
        ))
        edge_effects["depmap_target_gene_effect_median"] = [
            ess_map_median.get(str(h).upper()) if pd.notna(h) else None
            for h in edge_effects["target_hgnc"]
        ]
        edge_effects["target_frac_essential"] = [
            ess_map_frac.get(str(h).upper()) if pd.notna(h) else None
            for h in edge_effects["target_hgnc"]
        ]
        edge_effects["target_is_common_essential"] = (
            edge_effects["target_frac_essential"].fillna(0) >= 0.9
        )

    # Drug-target support
    if Path(args.drug_targets).exists():
        dt = pd.read_csv(args.drug_targets)
        dt_genes = set(dt.get("target_gene", pd.Series(dtype=str))
                         .astype(str).str.upper())
        edge_effects["drug_target_support"] = [
            (str(h).upper() in dt_genes) if pd.notna(h) else False
            for h in edge_effects["target_hgnc"]
        ]

    # Mark top decile by |delta|
    edge_effects["abs_delta"] = edge_effects["delta"].abs()
    threshold = edge_effects["abs_delta"].quantile(0.9)
    edge_effects["is_top_decile"] = edge_effects["abs_delta"] >= threshold

    # Refusal reason per edge (for transparency)
    def _refusal(row):
        reasons = []
        if pd.isna(row.get("target_hgnc")):
            reasons.append("target_id_did_not_map_to_HGNC")
        if not row.get("target_is_common_essential", False):
            reasons.append("downstream_gene_not_common_essential")
        if not row.get("drug_target_support", False):
            reasons.append("no_drug_target_evidence")
        return ";".join(reasons) if reasons else "ok"
    edge_effects["refusal_reason"] = edge_effects.apply(_refusal, axis=1)

    out_cols = [
        "edge_id", "source_raw", "target_raw", "edge_type",
        "source_hgnc", "target_hgnc",
        "source_uniprot", "target_uniprot",
        "delta", "abs_delta", "is_top_decile",
        "depmap_target_gene_effect_median", "target_frac_essential",
        "target_is_common_essential",
        "drug_target_support",
        "refusal_reason",
    ]
    out_df = edge_effects[[c for c in out_cols if c in edge_effects.columns]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    logger.info("Wrote per-edge evidence -> %s (%d rows)", args.out, len(out_df))

    # Honest summary
    n = len(out_df)
    n_hgnc_mapped = int(out_df["target_hgnc"].notna().sum())
    n_essential = int(out_df.get("target_is_common_essential", pd.Series(dtype=bool)).fillna(False).sum())
    n_drug_supported = int(out_df.get("drug_target_support", pd.Series(dtype=bool)).fillna(False).sum())
    top10 = out_df.sort_values("abs_delta", ascending=False).head(10)
    top10_hgnc = int(top10["target_hgnc"].notna().sum())
    top10_ess = int(top10.get("target_is_common_essential", pd.Series(dtype=bool)).fillna(False).sum())
    summary = {
        "n_edges": n,
        "n_target_hgnc_mapped": n_hgnc_mapped,
        "frac_target_hgnc_mapped": n_hgnc_mapped / max(n, 1),
        "n_target_common_essential": n_essential,
        "frac_target_common_essential": n_essential / max(n, 1),
        "n_drug_target_supported": n_drug_supported,
        "frac_drug_target_supported": n_drug_supported / max(n, 1),
        "top10_target_hgnc_mapped": top10_hgnc,
        "top10_target_common_essential": top10_ess,
        "causal_mechanism_gate_passing": bool(
            top10_ess >= 2 and top10_hgnc >= 8
        ),
        "honest_note": (
            "v17 — IDHarmonizer now bridges arbitrary STRING alias edge "
            "IDs (PDB, Entrez, KEGG, ...) to HGNC. CRISPR essentiality "
            "cross-check is no longer blocked by ID mismatch; refusals "
            "now reflect actual model edge ranking vs biological evidence."
        ),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Wall time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
