#!/usr/bin/env python3
"""
scripts/mortfm_build_biological_graph.py
========================================
Block D — assemble the heterogeneous biological graph.

Takes the per-source processed files emitted by Block-A ingestion scripts and
constructs the canonical ``data/processed/graphs/biological_edges.parquet``
edge list consumed by the graph-conditioned dynamics module.

Edge sources stitched in:
    * STRING PPI                            -> ppi edges
    * ChEMBL drug-target                    -> drug_target edges
    * Reactome pathway membership           -> protein_in_pathway edges
    * (optional) CRISPR co-essentiality     -> crispr_dependency edges
    * (optional) TF-gene regulatory edges   -> tf_regulates_gene edges

Each edge row:
    source_id, target_id, source_type, target_type, edge_type, edge_weight,
    directed, evidence_source

This script does NOT fabricate edges. If a source file is missing, that
edge-type is skipped and a warning is logged. The script will refuse to write
an empty graph (it raises SystemExit 1) so the dynamics module never starts
with no graph.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.identifier_mapping import load_identifier_maps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_build_graph")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--maps-dir", default="data/raw_public/identifier_maps")
    ap.add_argument("--out", default="data/processed/graphs/biological_edges.parquet")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []

    # 1. STRING PPI -- protein-protein edges by UniProt
    string_edges = processed / "graphs" / "string_edges.parquet"
    aliases = processed / "graphs" / "string_aliases.parquet"
    if string_edges.exists() and aliases.exists():
        edges = pd.read_parquet(string_edges)
        al = pd.read_parquet(aliases).drop_duplicates("string_id").set_index("string_id")["uniprot_id"]
        edges["source_uniprot"] = edges["source_id"].map(al)
        edges["target_uniprot"] = edges["target_id"].map(al)
        edges = edges.dropna(subset=["source_uniprot", "target_uniprot"])
        parts.append(pd.DataFrame({
            "source_id": edges["source_uniprot"],
            "target_id": edges["target_uniprot"],
            "source_type": "protein",
            "target_type": "protein",
            "edge_type": "ppi",
            "edge_weight": edges["combined_score"].astype(float) / 1000.0,
            "directed": False,
            "evidence_source": "STRING",
        }))
        logger.info("STRING: %d PPI edges (UniProt-resolved)", len(edges))
    else:
        logger.warning("STRING edges/aliases not found at %s -- skipping ppi edges", string_edges)

    # 2. ChEMBL drug-target edges
    drug_targets = processed / "drugs" / "drug_targets.csv"
    if drug_targets.exists():
        dt_df = pd.read_csv(drug_targets, low_memory=False)
        parts.append(pd.DataFrame({
            "source_id": dt_df["drug_id"].astype(str),
            "target_id": dt_df["target_uniprot"].astype(str),
            "source_type": "drug",
            "target_type": "protein",
            "edge_type": "drug_target",
            "edge_weight": 1.0,
            "directed": True,
            "evidence_source": dt_df.get("evidence_source", "ChEMBL").astype(str),
        }))
        logger.info("ChEMBL: %d drug-target edges", len(dt_df))
    else:
        logger.warning("Drug-targets file not found at %s -- skipping drug_target edges",
                       drug_targets)

    # 3. Reactome pathway membership
    react = processed / "graphs" / "reactome_membership.parquet"
    if react.exists():
        r = pd.read_parquet(react)
        parts.append(pd.DataFrame({
            "source_id": r["source_id"].astype(str),         # could be HGNC, NCBI, or UniProt
            "target_id": r["reactome_id"].astype(str),
            "source_type": "protein_or_gene",
            "target_type": "pathway",
            "edge_type": "protein_in_pathway",
            "edge_weight": 1.0,
            "directed": True,
            "evidence_source": "Reactome",
        }))
        logger.info("Reactome: %d protein-in-pathway edges", len(r))
    else:
        logger.warning("Reactome membership not found at %s -- skipping pathway edges", react)

    # 4. CRISPR co-essentiality (optional)
    crispr = processed / "perturbation" / "crispr_gene_effect.parquet"
    if crispr.exists():
        # Co-essentiality is a downstream graph; we just register CRISPR availability.
        logger.info("CRISPR gene-effect present at %s (co-essentiality not auto-computed)", crispr)

    if not parts:
        logger.error("No edge sources found -- refusing to write an empty graph.")
        return 1

    edges_df = pd.concat(parts, ignore_index=True)
    edges_df.to_parquet(out_path)
    logger.info("Wrote %d total edges -> %s", len(edges_df), out_path)
    logger.info("Edge-type breakdown:\n%s", edges_df["edge_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
