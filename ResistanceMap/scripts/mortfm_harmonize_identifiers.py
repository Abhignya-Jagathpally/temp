#!/usr/bin/env python3
"""
scripts/mortfm_harmonize_identifiers.py
=======================================
Block D-1 — improve gene-symbol → UniProt coverage past the 70% gate.

Strategy: pull additional aliases from UniProt's protein_nodes table (already
on disk) AND from STRING aliases (multi-source lookup), and produce a single
canonical mapping table. The existing ``resistancemap.data.identifier_mapping``
module provides the lookup primitives; this script only assembles the inputs
and writes the canonical CSV + coverage report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_harmonize_identifiers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot-nodes", default="data/processed/graphs/protein_nodes.csv")
    ap.add_argument("--string-aliases", default="data/processed/graphs/string_aliases.parquet")
    ap.add_argument("--features", default="data/processed/omics/rna_matrix.parquet",
                    help="Reference RNA matrix from which to pull the feature gene set.")
    ap.add_argument("--extra-features", nargs="*", default=[
        "data/processed/beataml/beataml_expression.parquet",
    ], help="Additional matrices whose columns extend the feature gene set.")
    ap.add_argument("--out", default="data/processed/metadata/gene_protein_identifier_map.csv")
    ap.add_argument("--report", default="logs/mortfm/identifier_mapping_report.json")
    args = ap.parse_args()

    # 1. Build the feature gene set we need to cover.
    feature_genes: set[str] = set()
    for p in [args.features] + list(args.extra_features):
        path = Path(p)
        if not path.exists():
            logger.warning("Skipping missing feature file: %s", path)
            continue
        df = pd.read_parquet(path)
        feature_genes.update(df.columns.astype(str).tolist())
    feature_genes = {g for g in feature_genes if g and g != "nan"}
    logger.info("Reference feature gene set: %d unique symbols", len(feature_genes))

    # 2. UniProt nodes table: gene_symbol -> uniprot_id (Tier 1: exact HGNC).
    uniprot = pd.read_csv(args.uniprot_nodes)
    uniprot["gene_symbol"] = uniprot["gene_symbol"].astype(str)
    tier1 = uniprot.dropna(subset=["gene_symbol"])
    tier1 = tier1[tier1["gene_symbol"].str.len() > 0]
    tier1 = tier1[tier1["gene_symbol"] != "nan"]
    tier1_map = dict(zip(tier1["gene_symbol"], tier1["uniprot_id"]))
    logger.info("Tier 1 (HGNC symbol -> UniProt): %d entries", len(tier1_map))

    # 3. STRING aliases — extract HGNC-symbol-shaped aliases (uppercase, len<=15)
    #    that map onto a STRING protein that we can resolve back to UniProt via
    #    Tier 1. This lets us catch HGNC aliases (e.g. "AAA1" for the new symbol).
    string_aliases = pd.read_parquet(args.string_aliases)
    string_to_uniprot: dict[str, str] = (
        string_aliases.drop_duplicates("string_id")
        .set_index("string_id")["uniprot_id"]
        .to_dict()
    )
    # Tier 5: HGNC alias / previous symbol. Read every alias from STRING that
    # is *not* itself a UniProt accession (heuristic: uppercase + alpha + len<=15).
    # We attach it to whatever UniProt id its STRING protein resolves to.
    full_aliases = pd.read_parquet(args.string_aliases)
    full_aliases = full_aliases.rename(columns={"uniprot_id": "alias"})
    # For Tier 5 we need the *all* alias rows from STRING (not the UniProt-only ones).
    # If our parquet is only UniProt-filtered, fall back to using the HGNC mapping table.
    tier5_map: dict[str, str] = {}
    for _, row in full_aliases.iterrows():
        alias = str(row.get("alias", ""))
        sid = str(row.get("string_id", ""))
        if not alias or not sid:
            continue
        if alias.isalnum() and 2 <= len(alias) <= 15 and not (
            alias[0] in "OPQ" and len(alias) == 6
        ):
            u = string_to_uniprot.get(sid)
            if u and alias not in tier5_map:
                tier5_map[alias] = u

    # 4. Build the per-feature mapping.
    rows = []
    for gene in sorted(feature_genes):
        if gene in tier1_map:
            rows.append({"feature_gene": gene, "uniprot_id": tier1_map[gene], "tier": 1, "source": "HGNC_symbol"})
        elif gene.upper() in tier1_map:
            rows.append({"feature_gene": gene, "uniprot_id": tier1_map[gene.upper()], "tier": 1, "source": "HGNC_case_normalised"})
        elif gene in tier5_map:
            rows.append({"feature_gene": gene, "uniprot_id": tier5_map[gene], "tier": 5, "source": "STRING_alias"})
        elif gene.upper() in tier5_map:
            rows.append({"feature_gene": gene, "uniprot_id": tier5_map[gene.upper()], "tier": 5, "source": "STRING_alias"})
        else:
            rows.append({"feature_gene": gene, "uniprot_id": "", "tier": 0, "source": "unmapped"})
    mapping = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.out, index=False)
    n_mapped = int((mapping["tier"] > 0).sum())
    coverage = n_mapped / max(len(mapping), 1)
    logger.info(
        "Coverage: %d/%d (%.1f%%); tiers: %s",
        n_mapped, len(mapping), 100 * coverage,
        mapping["tier"].value_counts().to_dict(),
    )

    report = {
        "n_feature_genes_total": int(len(mapping)),
        "n_feature_genes_mapped": n_mapped,
        "feature_gene_to_uniprot_coverage": coverage,
        "by_tier": {
            int(t): int((mapping["tier"] == t).sum())
            for t in sorted(mapping["tier"].unique())
        },
        "by_source": mapping["source"].value_counts().to_dict(),
        "pathway_context_gate_pass": coverage >= 0.70,
        "out_csv": str(Path(args.out).resolve()),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote mapping -> %s; report -> %s; gate_pass=%s",
                args.out, args.report, report["pathway_context_gate_pass"])
    return 0 if report["pathway_context_gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
