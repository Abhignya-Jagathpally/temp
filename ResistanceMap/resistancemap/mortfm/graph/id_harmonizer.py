"""
resistancemap/mortfm/graph/id_harmonizer.py
============================================
Build + serve the STRING ENSP <-> UniProt <-> HGNC symbol bridge.

The STRING aliases file ``9606.protein.aliases.v12.0.txt.gz`` is the
canonical source. We index two source codes:

  * ``Ensembl_HGNC_symbol``  — preferred HGNC gene symbol (e.g. PSMB5)
  * ``UniProt_AC``           — UniProt accession (e.g. P28074)

A single STRING protein id (``9606.ENSP00000380155``) can map to one
HGNC symbol and one or more UniProt accessions. We honor whichever
mapping the file provides; missing mappings stay missing (no
fabrication, no fuzzy-string guessing).

This module is hot-cached in ``data/processed/metadata/string_id_bridge.parquet``
after first build so subsequent loads are sub-second.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


_PREFERRED_SOURCES = {
    # source-code -> (column, priority — lower wins)
    "Ensembl_HGNC_symbol": ("hgnc_symbol", 1),
    "Ensembl_HGNC": ("hgnc_symbol", 2),
    "UniProt_GN_Name": ("hgnc_symbol", 3),
    "UniProt_AC": ("uniprot_id", 1),
    "Ensembl_UniProt": ("uniprot_id", 2),
    "BioMart_HUGO": ("hgnc_symbol", 4),
}


@dataclass
class IDHarmonizer:
    """Per-protein ID lookup. Build once via :meth:`build_or_load`.

    The lookup table is indexed by ``string_id`` (full STRING form
    ``9606.ENSPxxxxx``); the bare ``ensp`` is also keyed if needed.

    Additionally an ``alias_to_string`` map covers ALL aliases the
    STRING file lists (PDB ids, Entrez ids, KEGG ids, etc.) so the
    InterventionGraph — which stores edges by whatever ID the graph
    builder happened to pick — can bridge to HGNC via this single map.
    """

    string_to_hgnc: Dict[str, str]
    string_to_uniprot: Dict[str, str]
    hgnc_to_string: Dict[str, str]
    uniprot_to_string: Dict[str, str]
    alias_to_string: Dict[str, str]
    n_rows: int

    @classmethod
    def build_or_load(
        cls,
        aliases_gz: str = "data/raw/string/9606.protein.aliases.v12.0.txt.gz",
        cache_parquet: str = "data/processed/metadata/string_id_bridge.parquet",
    ) -> "IDHarmonizer":
        cache = Path(cache_parquet)
        alias_cache = Path(cache_parquet).parent / "string_alias_to_id.parquet"
        if cache.exists() and alias_cache.exists():
            df = pd.read_parquet(cache)
            alias_df = pd.read_parquet(alias_cache)
            alias_to_string = dict(zip(alias_df["alias"].astype(str),
                                        alias_df["string_id"].astype(str)))
            logger.info("Loaded cached ID bridge: %d rows + %d aliases",
                        len(df), len(alias_to_string))
            return cls._from_frame(df, alias_to_string=alias_to_string)
        # Build from aliases file.
        gz = Path(aliases_gz)
        if not gz.exists():
            raise FileNotFoundError(f"STRING aliases not found at {gz}")
        logger.info("Building ID bridge from %s ...", gz)
        # Map per-string-id: keep the highest-priority alias per column.
        per_protein: Dict[str, Dict[str, tuple]] = {}
        alias_to_string: Dict[str, str] = {}
        with gzip.open(gz, "rt") as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                sid, alias, source = parts[0], parts[1], parts[2]
                # Reverse-index every alias to string_id (first-write wins).
                # This is what lets InterventionGraph bridge arbitrary edge IDs
                # — PDB codes, Entrez ints, KEGG ids — back to STRING then HGNC.
                if alias and alias not in alias_to_string:
                    alias_to_string[alias] = sid
                if source not in _PREFERRED_SOURCES:
                    continue
                col, prio = _PREFERRED_SOURCES[source]
                slot = per_protein.setdefault(sid, {})
                cur = slot.get(col)
                if cur is None or prio < cur[1]:
                    slot[col] = (alias, prio)
        rows: List[Dict[str, Optional[str]]] = []
        for sid, slot in per_protein.items():
            rows.append({
                "string_id": sid,
                "hgnc_symbol": slot.get("hgnc_symbol", (None, 0))[0],
                "uniprot_id": slot.get("uniprot_id", (None, 0))[0],
            })
        df = pd.DataFrame(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        # Cache the alias map separately because it's larger.
        alias_cache = cache.parent / "string_alias_to_id.parquet"
        pd.DataFrame(
            list(alias_to_string.items()),
            columns=["alias", "string_id"],
        ).to_parquet(alias_cache)
        logger.info("Wrote ID bridge cache: %d proteins (with %d aliases) -> %s",
                    len(df), len(alias_to_string), cache)
        return cls._from_frame(df, alias_to_string=alias_to_string)

    @classmethod
    def _from_frame(
        cls,
        df: pd.DataFrame,
        alias_to_string: Optional[Dict[str, str]] = None,
    ) -> "IDHarmonizer":
        sh = (
            df.dropna(subset=["hgnc_symbol"])
            .set_index("string_id")["hgnc_symbol"].to_dict()
        )
        su = (
            df.dropna(subset=["uniprot_id"])
            .set_index("string_id")["uniprot_id"].to_dict()
        )
        hs = (
            df.dropna(subset=["hgnc_symbol"])
            .drop_duplicates("hgnc_symbol")
            .set_index("hgnc_symbol")["string_id"].to_dict()
        )
        us = (
            df.dropna(subset=["uniprot_id"])
            .drop_duplicates("uniprot_id")
            .set_index("uniprot_id")["string_id"].to_dict()
        )
        return cls(
            string_to_hgnc=sh, string_to_uniprot=su,
            hgnc_to_string=hs, uniprot_to_string=us,
            alias_to_string=alias_to_string or {},
            n_rows=len(df),
        )

    def any_alias_to_hgnc(self, aliases: Iterable[str]) -> List[Optional[str]]:
        """Map any STRING-known alias (PDB, Entrez, KEGG, ...) -> HGNC symbol.

        This is the method InterventionGraph + PathwayCausalValidator call
        when they have an edge ID in a non-canonical namespace.
        """
        out: List[Optional[str]] = []
        for a in aliases:
            sid = self.alias_to_string.get(str(a))
            if sid is None:
                # Try uppercase + STRING-prefixed.
                sid = self.alias_to_string.get(str(a).upper())
            if sid is None and not str(a).startswith("9606."):
                sid = self.alias_to_string.get(f"9606.{a}")
            if sid is None:
                out.append(None)
                continue
            out.append(self.string_to_hgnc.get(sid))
        return out

    def string_to_ensp(self, ids: Iterable[str]) -> List[Optional[str]]:
        out = []
        for s in ids:
            s = str(s)
            if s.startswith("9606."):
                out.append(s.split(".", 1)[1])
            else:
                out.append(s)
        return out

    def string_to_hgnc_symbol(self, ids: Iterable[str]) -> List[Optional[str]]:
        return [self.string_to_hgnc.get(str(s)) for s in ids]

    def string_to_uniprot_acc(self, ids: Iterable[str]) -> List[Optional[str]]:
        return [self.string_to_uniprot.get(str(s)) for s in ids]

    def hgnc_to_string_id(self, symbols: Iterable[str]) -> List[Optional[str]]:
        return [self.hgnc_to_string.get(str(s).upper()) or self.hgnc_to_string.get(str(s))
                for s in symbols]

    def coverage_against_hgnc_set(self, hgnc_symbols: Sequence[str]) -> dict:
        """How many of the given HGNC symbols actually appear as targets of a STRING id?"""
        present = sum(1 for s in hgnc_symbols if s in self.hgnc_to_string)
        return {"n_queried": len(hgnc_symbols), "n_present": present,
                "coverage": present / max(len(hgnc_symbols), 1)}
