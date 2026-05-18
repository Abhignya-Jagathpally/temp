"""
resistancemap/data/reactome_loader.py
=====================================
Reactome pathway-membership + hierarchy loader.

Reactome distributes flat TSVs of gene/protein → pathway membership and a
parent–child pathway hierarchy. This loader produces:

* ``pathway_membership.csv`` — one row per (pathway_id, gene_symbol_or_uniprot)
* ``pathway_hierarchy.csv``  — one row per (parent_pathway_id, child_pathway_id)

Both with explicit species filtering to ``Homo sapiens``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_PATHWAY_COLUMNS = (
    "pathway_id",
    "pathway_name",
    "gene_symbol",
    "uniprot_id",
    "species",
    "evidence_source",
)


def _find_one(base: Path, patterns: Sequence[str], label: str) -> Path:
    for pat in patterns:
        hits = list(base.rglob(pat))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(
        f"Reactome {label} not found under {base.resolve()} (tried {list(patterns)}). "
        f"Download from https://reactome.org/download/current/."
    )


def load_reactome_membership(reactome_dir: str) -> pd.DataFrame:
    """Load Reactome pathway membership for Homo sapiens.

    Tries HGNC2Reactome first (cleanest), then UniProt2Reactome, then
    NCBI2Reactome. Returns a DataFrame with :data:`REQUIRED_PATHWAY_COLUMNS`.
    """
    base = Path(reactome_dir)
    src = _find_one(
        base,
        ["HGNC2Reactome*All_Levels*.txt", "UniProt2Reactome*All_Levels*.txt",
         "NCBI2Reactome*All_Levels*.txt"],
        "membership file",
    )
    df = pd.read_csv(
        src, sep="\t", header=None, low_memory=False,
        names=["source_id", "pathway_id", "url", "pathway_name", "evidence_code", "species"],
    )
    df = df[df["species"].astype(str).str.startswith("Homo sapiens")].copy()
    logger.info("Reactome %s: %d human membership rows", src.name, len(df))
    # Normalise the identifier column to gene_symbol or uniprot_id based on file name.
    name = src.name.lower()
    if "hgnc" in name:
        df["gene_symbol"] = df["source_id"].astype(str)
        df["uniprot_id"] = "unknown"
    elif "uniprot" in name:
        df["uniprot_id"] = df["source_id"].astype(str)
        df["gene_symbol"] = "unknown"
    else:
        # NCBI Entrez gene IDs — leave both unknown and let identifier_mapping
        # downstream resolve.
        df["gene_symbol"] = "unknown"
        df["uniprot_id"] = "unknown"
    df["evidence_source"] = "Reactome"
    return df[list(REQUIRED_PATHWAY_COLUMNS)]


def load_reactome_hierarchy(reactome_dir: str) -> pd.DataFrame:
    """Parent-child pathway relationships restricted to ``Homo sapiens``."""
    base = Path(reactome_dir)
    src = _find_one(base, ["ReactomePathwaysRelation.txt"], "hierarchy file")
    df = pd.read_csv(src, sep="\t", header=None,
                     names=["parent_pathway_id", "child_pathway_id"])
    df = df[df["parent_pathway_id"].astype(str).str.startswith("R-HSA")].copy()
    logger.info("Reactome hierarchy: %d parent-child pairs (HSA-only)", len(df))
    return df


def write_pathway_coverage_report(
    *,
    membership: pd.DataFrame,
    feature_genes: Sequence[str],
    out_json: str = "logs/mortfm/reactome_coverage_report.json",
) -> Dict[str, float]:
    pathway_set = set(membership["pathway_id"].astype(str).unique())
    pathway_gene_set = set(membership["gene_symbol"].astype(str).unique())
    n_genes = len(feature_genes)
    mapped = sum(1 for g in feature_genes if g in pathway_gene_set)
    rep = {
        "n_pathways_total": len(pathway_set),
        "n_membership_rows": int(len(membership)),
        "n_feature_genes_total": n_genes,
        "n_feature_genes_in_a_pathway": mapped,
        "feature_gene_pathway_coverage_rate": mapped / max(n_genes, 1),
        "evidence_source": "Reactome",
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    logger.info("Wrote Reactome coverage report -> %s", out)
    return rep
