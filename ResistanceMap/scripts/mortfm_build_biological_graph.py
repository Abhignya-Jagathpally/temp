#!/usr/bin/env python3
"""
scripts/mortfm_build_biological_graph.py
========================================
Block D — assemble the heterogeneous biological graph.

Edge sources (each optional; missing sources are reported, not fabricated):
    * STRING PPI                            -> ppi edges
    * ChEMBL drug-target                    -> drug_target edges
    * Reactome pathway membership           -> protein_in_pathway edges
    * UniProt sequences (node attribute)     -> protein_nodes.csv with sequence_present flag
    * (optional) CRISPR co-essentiality     -> crispr_dependency edges (TODO)
    * (optional) TF-gene regulatory edges   -> tf_regulates_gene edges (TODO)

Outputs:
    data/processed/graphs/biological_edges.parquet
    data/processed/graphs/protein_nodes.csv
    data/processed/graphs/drug_nodes.csv
    data/processed/graphs/pathway_nodes.csv
    logs/mortfm/graph_coverage_report.json

If only STRING is available the graph reduces to protein↔protein and the
coverage report explicitly flags that the pathway head must label its output
as "protein-level attribution" rather than "pathway mechanism".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_build_graph")


@dataclass
class GraphCoverageReport:
    ppi_edges: int = 0
    drug_target_edges: int = 0
    pathway_edges: int = 0
    crispr_edges: int = 0
    n_protein_nodes: int = 0
    n_drug_nodes: int = 0
    n_pathway_nodes: int = 0
    has_uniprot_sequences: bool = False
    has_drug_target: bool = False
    has_pathway_membership: bool = False
    edge_types_present: List[str] = field(default_factory=list)
    pathway_claim_allowed: bool = False
    sequence_aware_claim_allowed: bool = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-graph", default="data/processed/graphs/biological_edges.parquet")
    ap.add_argument("--report", default="logs/mortfm/graph_coverage_report.json")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_path = Path(args.out_graph)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    edge_parts: list[pd.DataFrame] = []
    coverage = GraphCoverageReport()

    # 1. STRING PPI
    string_edges = processed / "graphs" / "string_edges.parquet"
    aliases = processed / "graphs" / "string_aliases.parquet"
    if string_edges.exists() and aliases.exists():
        edges = pd.read_parquet(string_edges)
        al = pd.read_parquet(aliases).drop_duplicates("string_id").set_index("string_id")["uniprot_id"]
        edges["source_uniprot"] = edges["source_id"].map(al)
        edges["target_uniprot"] = edges["target_id"].map(al)
        edges = edges.dropna(subset=["source_uniprot", "target_uniprot"])
        edge_parts.append(pd.DataFrame({
            "source_id": edges["source_uniprot"],
            "target_id": edges["target_uniprot"],
            "source_type": "protein",
            "target_type": "protein",
            "edge_type": "ppi",
            "edge_weight": edges["combined_score"].astype(float) / 1000.0,
            "directed": False,
            "evidence_source": "STRING",
        }))
        coverage.ppi_edges = len(edges)
        coverage.edge_types_present.append("ppi")
        logger.info("STRING: %d PPI edges (UniProt-resolved)", len(edges))
    else:
        logger.warning("STRING edges/aliases missing -- skipping ppi edges")

    # 2. ChEMBL drug-target
    drug_targets = processed / "drugs" / "chembl_drug_targets.csv"
    if drug_targets.exists():
        dt_df = pd.read_csv(drug_targets, low_memory=False)
        edge_parts.append(pd.DataFrame({
            "source_id": dt_df["drug_id"].astype(str),
            "target_id": dt_df["target_uniprot"].astype(str),
            "source_type": "drug",
            "target_type": "protein",
            "edge_type": "drug_target",
            "edge_weight": 1.0,
            "directed": True,
            "evidence_source": "ChEMBL",
        }))
        coverage.drug_target_edges = len(dt_df)
        coverage.has_drug_target = True
        coverage.edge_types_present.append("drug_target")
        logger.info("ChEMBL: %d drug-target edges", len(dt_df))
    else:
        # Fall back to drug_ontology curated targets if no ChEMBL.
        dt_alt = processed / "drugs" / "drug_targets.csv"
        if dt_alt.exists():
            dt_df = pd.read_csv(dt_alt, low_memory=False)
            edge_parts.append(pd.DataFrame({
                "source_id": dt_df["drug_id"].astype(str),
                "target_id": dt_df["target_uniprot"].astype(str),
                "source_type": "drug",
                "target_type": "protein",
                "edge_type": "drug_target",
                "edge_weight": 1.0,
                "directed": True,
                "evidence_source": "curated",
            }))
            coverage.drug_target_edges = len(dt_df)
            coverage.has_drug_target = True
            coverage.edge_types_present.append("drug_target")
            logger.info("Curated drug-target fallback: %d edges (use ChEMBL for production)", len(dt_df))
        else:
            logger.warning("No drug-target file found -- skipping drug_target edges")

    # 3. Reactome pathway membership
    pw = processed / "pathways" / "reactome_pathway_membership.csv"
    if pw.exists():
        r = pd.read_csv(pw, low_memory=False)
        edge_parts.append(pd.DataFrame({
            "source_id": r["gene_symbol"].astype(str).where(
                r["gene_symbol"].astype(str) != "unknown",
                r["uniprot_id"].astype(str),
            ),
            "target_id": r["pathway_id"].astype(str),
            "source_type": "protein_or_gene",
            "target_type": "pathway",
            "edge_type": "protein_in_pathway",
            "edge_weight": 1.0,
            "directed": True,
            "evidence_source": "Reactome",
        }))
        coverage.pathway_edges = len(r)
        coverage.has_pathway_membership = True
        coverage.edge_types_present.append("protein_in_pathway")
        logger.info("Reactome: %d protein-in-pathway edges", len(r))
    else:
        # Older path fallback
        pw_old = processed / "graphs" / "reactome_membership.parquet"
        if pw_old.exists():
            r = pd.read_parquet(pw_old)
            edge_parts.append(pd.DataFrame({
                "source_id": r["source_id"].astype(str),
                "target_id": r["reactome_id"].astype(str),
                "source_type": "protein_or_gene",
                "target_type": "pathway",
                "edge_type": "protein_in_pathway",
                "edge_weight": 1.0,
                "directed": True,
                "evidence_source": "Reactome",
            }))
            coverage.pathway_edges = len(r)
            coverage.has_pathway_membership = True
            coverage.edge_types_present.append("protein_in_pathway")
            logger.info("Reactome (legacy): %d protein-in-pathway edges", len(r))
        else:
            logger.warning("No Reactome membership found -- skipping pathway edges")

    # 4. Protein sequences attribute
    seq_path = processed / "proteins" / "protein_sequences.fasta"
    coverage.has_uniprot_sequences = seq_path.exists()

    if not edge_parts:
        logger.error("No edge sources -- refusing to write an empty graph.")
        return 1

    edges_df = pd.concat(edge_parts, ignore_index=True)
    edges_df.to_parquet(out_path)
    logger.info("Wrote %d total edges -> %s", len(edges_df), out_path)
    logger.info("Edge-type breakdown:\n%s", edges_df["edge_type"].value_counts().to_string())

    # Node tables
    proteins = pd.DataFrame({
        "uniprot_id": pd.unique(
            edges_df.loc[
                (edges_df["source_type"] == "protein") | (edges_df["target_type"] == "protein"),
                ["source_id", "target_id"],
            ].values.ravel()
        ),
    })
    proteins.to_csv(out_path.parent / "protein_nodes.csv", index=False)
    coverage.n_protein_nodes = len(proteins)

    drugs = pd.DataFrame({
        "drug_id": pd.unique(
            edges_df.loc[edges_df["source_type"] == "drug", "source_id"].values
        ),
    })
    drugs.to_csv(out_path.parent / "drug_nodes.csv", index=False)
    coverage.n_drug_nodes = len(drugs)

    pathways = pd.DataFrame({
        "pathway_id": pd.unique(
            edges_df.loc[edges_df["target_type"] == "pathway", "target_id"].values
        ),
    })
    pathways.to_csv(out_path.parent / "pathway_nodes.csv", index=False)
    coverage.n_pathway_nodes = len(pathways)

    # Claim gates
    coverage.pathway_claim_allowed = (
        coverage.has_pathway_membership and coverage.has_drug_target
    )
    coverage.sequence_aware_claim_allowed = coverage.has_uniprot_sequences

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(asdict(coverage), f, indent=2)
    logger.info("Wrote graph coverage report -> %s", report_path)
    logger.info("pathway_claim_allowed = %s; sequence_aware_claim_allowed = %s",
                coverage.pathway_claim_allowed, coverage.sequence_aware_claim_allowed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
