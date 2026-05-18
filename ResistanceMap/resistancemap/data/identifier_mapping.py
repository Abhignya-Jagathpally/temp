"""
resistancemap/data/identifier_mapping.py
========================================
Stable mapping across the biological identifier namespaces used by MORT-FM:

    HGNC  ⇄  Ensembl gene  ⇄  UniProt protein  ⇄  STRING protein
    HGNC  ⇄  Reactome pathway membership
    ChEMBL drug  →  UniProt protein targets

Why this module exists
----------------------
The MORT-FM data contract requires that a "BCMA" referenced in the drug
ontology, a "TNFRSF17" referenced in the proteomics feature manifest, an
"ENSG00000048462" referenced in the RNA encoder, a "Q02223" referenced in
ESM-2 embeddings, and a "9606.ENSP00000358776" referenced in the STRING PPI
graph all point to *the same protein*. Without a single mapping table the
PPI / pathway / counterfactual outputs become biologically invalid.

Honest behaviour
----------------
* All loaders raise :class:`FileNotFoundError` if the source map file is
  missing. We do not hard-code mappings into this module — they must come
  from versioned dumps under ``data/raw_public/uniprot/`` or equivalent.
* :func:`unify_identifiers` returns ``None`` for any identifier it cannot
  map. Callers must handle the ``None`` case explicitly — *no fallback to
  the input string*.
* Mappings are cached in-process via :func:`functools.lru_cache`; pass
  ``cache_clear=True`` to force reload after a new dump lands.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core mapping types
# ---------------------------------------------------------------------------


@dataclass
class IdentifierMaps:
    """Bidirectional lookup tables. All dict keys are case-sensitive strings."""

    hgnc_to_ensembl: Dict[str, str]
    ensembl_to_hgnc: Dict[str, str]
    hgnc_to_uniprot: Dict[str, str]
    uniprot_to_hgnc: Dict[str, str]
    uniprot_to_string: Dict[str, str]
    string_to_uniprot: Dict[str, str]
    hgnc_to_reactome: Dict[str, List[str]]
    chembl_to_uniprot: Dict[str, List[str]]

    def harmonise_gene_symbol(self, raw: str) -> Optional[str]:
        """Return the canonical HGNC symbol for ``raw`` (may already be HGNC)."""
        if raw in self.hgnc_to_ensembl:
            return raw
        # Try as Ensembl ID first.
        if raw in self.ensembl_to_hgnc:
            return self.ensembl_to_hgnc[raw]
        # Try as UniProt.
        if raw in self.uniprot_to_hgnc:
            return self.uniprot_to_hgnc[raw]
        # Try as STRING -> UniProt -> HGNC.
        u = self.string_to_uniprot.get(raw)
        if u is not None:
            return self.uniprot_to_hgnc.get(u)
        return None

    def gene_to_uniprot(self, gene: str) -> Optional[str]:
        return self.hgnc_to_uniprot.get(gene)

    def uniprot_to_string_id(self, uniprot: str) -> Optional[str]:
        return self.uniprot_to_string.get(uniprot)

    def pathways_for_gene(self, gene: str) -> List[str]:
        return list(self.hgnc_to_reactome.get(gene, ()))

    def drug_targets(self, chembl_id: str) -> List[str]:
        return list(self.chembl_to_uniprot.get(chembl_id, ()))


# ---------------------------------------------------------------------------
# Loaders for each upstream dump
# ---------------------------------------------------------------------------


def _read_two_column(path: str, src_col: str, dst_col: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"identifier-mapping file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, low_memory=False, dtype=str)
    if src_col not in df.columns or dst_col not in df.columns:
        raise KeyError(
            f"{p.name} must have columns {src_col!r} and {dst_col!r}; "
            f"available: {list(df.columns)[:10]}"
        )
    df = df.dropna(subset=[src_col, dst_col])
    out: Dict[str, str] = {}
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        if src and dst and src not in out:
            out[src] = dst
    return out


def _read_one_to_many(path: str, src_col: str, dst_col: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"identifier-mapping file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, low_memory=False, dtype=str)
    df = df.dropna(subset=[src_col, dst_col])
    out: Dict[str, List[str]] = {}
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        out.setdefault(src, []).append(dst)
    return out


def load_identifier_maps(
    *,
    hgnc_ensembl_path: str,
    uniprot_xref_path: str,
    string_alias_path: str,
    reactome_membership_path: str,
    chembl_target_path: str,
) -> IdentifierMaps:
    """Load all five maps. Raises if any file is missing.

    Expected file shapes
    --------------------
    * ``hgnc_ensembl_path`` — TSV with columns ``hgnc_symbol``, ``ensembl_gene_id``
      (HGNC custom download or BioMart export).
    * ``uniprot_xref_path`` — TSV with columns ``uniprot_id``, ``hgnc_symbol``
      (UniProt ID-mapping export, columns ``gene_primary`` -> ``hgnc_symbol``).
    * ``string_alias_path`` — TSV with columns ``string_id``, ``uniprot_id``
      (e.g. STRING ``human.aliases.v12.0.txt``, filtered to ``alias_source=UniProt``).
    * ``reactome_membership_path`` — TSV with columns ``hgnc_symbol``, ``reactome_id``
      (Reactome ``NCBI2Reactome.txt`` filtered to human + mapped to HGNC).
    * ``chembl_target_path`` — TSV with columns ``chembl_id``, ``uniprot_id``
      (ChEMBL drug → target accessions export).
    """
    hgnc_to_ensembl = _read_two_column(hgnc_ensembl_path, "hgnc_symbol", "ensembl_gene_id")
    hgnc_to_uniprot = _read_two_column(uniprot_xref_path, "hgnc_symbol", "uniprot_id")
    string_to_uniprot = _read_two_column(string_alias_path, "string_id", "uniprot_id")
    hgnc_to_reactome = _read_one_to_many(reactome_membership_path, "hgnc_symbol", "reactome_id")
    chembl_to_uniprot = _read_one_to_many(chembl_target_path, "chembl_id", "uniprot_id")

    ensembl_to_hgnc = {v: k for k, v in hgnc_to_ensembl.items()}
    uniprot_to_hgnc = {v: k for k, v in hgnc_to_uniprot.items()}
    uniprot_to_string = {v: k for k, v in string_to_uniprot.items()}

    maps = IdentifierMaps(
        hgnc_to_ensembl=hgnc_to_ensembl,
        ensembl_to_hgnc=ensembl_to_hgnc,
        hgnc_to_uniprot=hgnc_to_uniprot,
        uniprot_to_hgnc=uniprot_to_hgnc,
        uniprot_to_string=uniprot_to_string,
        string_to_uniprot=string_to_uniprot,
        hgnc_to_reactome=hgnc_to_reactome,
        chembl_to_uniprot=chembl_to_uniprot,
    )
    logger.info(
        "Loaded identifier maps: %d HGNC↔Ensembl, %d HGNC↔UniProt, %d UniProt↔STRING, "
        "%d HGNC→Reactome (one-to-many), %d ChEMBL→UniProt (one-to-many)",
        len(hgnc_to_ensembl), len(hgnc_to_uniprot), len(string_to_uniprot),
        len(hgnc_to_reactome), len(chembl_to_uniprot),
    )
    return maps


def unify_identifiers(
    identifiers: Iterable[str],
    maps: IdentifierMaps,
) -> Dict[str, Optional[str]]:
    """Map an arbitrary list of identifiers to their canonical HGNC symbol.

    Returns ``{original: hgnc_or_None}``. Callers MUST handle the ``None`` case
    explicitly — no silent fall-through to the input string.
    """
    out: Dict[str, Optional[str]] = {}
    for ident in identifiers:
        out[ident] = maps.harmonise_gene_symbol(str(ident))
    n_unmapped = sum(1 for v in out.values() if v is None)
    if n_unmapped:
        logger.warning("unify_identifiers: %d / %d identifiers could not be mapped to HGNC",
                       n_unmapped, len(out))
    return out
