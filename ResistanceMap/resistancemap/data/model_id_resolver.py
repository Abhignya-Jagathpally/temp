"""
resistancemap/data/model_id_resolver.py
=======================================
Cell-line model-ID resolver for the Block-A public-data assembly.

The single most fragile join in cancer cell-line ML is the model identifier:
DepMap uses ``ACH-XXXXXX``, Sanger uses ``SIDM*``, COSMIC uses numeric IDs,
CCLE uses ``<NAME>_<TISSUE>``, GDSC and PRISM use yet other identifiers.
A wrong join silently fuses two unrelated cell lines and corrupts every
downstream metric. This module normalises all of them to a single
``model_id`` (DepMap_ID-style, e.g. ``ACH-000001``) and records the mapping
for audit.

Honest behaviour
----------------
* The full resolver requires DepMap ``Model.csv`` (download from
  https://depmap.org/portal/data_page/). Until that file is present,
  :func:`load_model_metadata` raises :class:`FileNotFoundError` with the
  download instructions.
* :func:`bootstrap_partial_resolver` is a *strictly limited* fallback for
  when only DepMap omics files (CRISPR / proteomics) are on disk — it
  extracts whatever DepMap IDs appear in those headers and emits a minimal
  ``model_metadata.csv`` with ``depmap_id`` populated but ``lineage`` /
  ``oncotree_code`` / ``ccle_name`` left ``unknown``. This bootstrap is
  ONLY usable for split-by-depmap_id; any mapping to GDSC/PRISM/COSMIC IDs
  via this resolver returns ``None``.
* The :class:`ModelIDResolver.report` method dumps a coverage report so the
  acceptance gate can refuse to run Block A pretraining when the GDSC/PRISM
  mapping rate is below threshold.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns the resolver attempts to populate from Model.csv.
REQUIRED_MODEL_COLUMNS: tuple[str, ...] = (
    "model_id",
    "depmap_id",
    "ach_id",
    "ccle_name",
    "sanger_model_id",
    "cosmic_id",
    "model_name",
    "lineage",
    "lineage_subtype",
    "disease",
    "oncotree_code",
    "primary_or_metastasis",
    "source",
)

#: Column-name guesses for each DepMap Model.csv release (24Q2, 24Q4, 25Q1, ...).
_MODEL_COL_HINTS: Dict[str, tuple[str, ...]] = {
    "depmap_id": ("ModelID", "DepMap_ID", "depmap_id"),
    "ccle_name": ("CCLEName", "CCLE_Name", "stripped_cell_line_name"),
    "sanger_model_id": ("SangerModelID", "Sanger_Model_ID"),
    "cosmic_id": ("COSMICID", "COSMIC_ID", "cosmic_id"),
    "model_name": ("ModelDisplayName", "StrippedCellLineName", "CellLineName"),
    "lineage": ("OncotreeLineage", "lineage_1", "Lineage"),
    "lineage_subtype": ("OncotreeSubtype", "OncotreeCode", "lineage_2"),
    "disease": ("OncotreePrimaryDisease", "PrimaryDisease"),
    "oncotree_code": ("OncotreeCode", "oncotree_code"),
    "primary_or_metastasis": ("PrimaryOrMetastasis", "primary_or_metastasis"),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _first_column(hints: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    for h in hints:
        if h in df.columns:
            return h
    return None


def load_model_metadata(model_csv_path: str) -> pd.DataFrame:
    """Parse DepMap ``Model.csv`` into the canonical ``model_metadata`` schema.

    Raises
    ------
    FileNotFoundError
        If the file is absent. Message names the DepMap download URL.
    """
    p = Path(model_csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"DepMap Model.csv not found at {p.resolve()}. "
            f"Download from https://depmap.org/portal/data_page/ "
            f"(release 24Q2+, file 'Model.csv') and place under data/raw/depmap/."
        )
    raw = pd.read_csv(p, low_memory=False)
    out = pd.DataFrame(index=raw.index)
    for col in REQUIRED_MODEL_COLUMNS:
        if col == "model_id":
            continue
        if col == "ach_id":
            # Alias for depmap_id
            continue
        src = _first_column(_MODEL_COL_HINTS.get(col, ()), raw)
        out[col] = raw[src].astype(str) if src else "unknown"
    # Canonical model_id := depmap_id when present, else fall back to ccle_name.
    out["model_id"] = out["depmap_id"].where(
        out["depmap_id"].astype(str).str.startswith("ACH"), out["ccle_name"]
    )
    out["ach_id"] = out["depmap_id"]
    out["source"] = "DepMap Model.csv"
    # Re-order columns to canonical schema.
    return out[list(REQUIRED_MODEL_COLUMNS)]


def bootstrap_partial_resolver(
    *,
    depmap_dir: str,
    out_csv: str,
) -> pd.DataFrame:
    """Build a *minimal* model_metadata from DepMap omics headers alone.

    Used only when Model.csv is absent but CRISPRGeneEffect.csv / proteomics
    are. Every row has ``depmap_id`` set but everything else is ``unknown``;
    callers MUST treat this as a stopgap, not a real resolver.
    """
    crispr = Path(depmap_dir) / "CRISPRGeneEffect.csv"
    prot = Path(depmap_dir) / "protein_quant_current_normalized.csv.gz"
    candidates: set[str] = set()
    if crispr.exists():
        ids = pd.read_csv(crispr, nrows=0).columns
        # First column is gene; DepMap IDs appear in rows for CRISPRGeneEffect.
        ids = pd.read_csv(crispr, usecols=[0]).iloc[:, 0].astype(str)
        candidates.update(ids[ids.str.startswith("ACH-")].tolist())
        logger.info("Bootstrap resolver: found %d DepMap IDs in CRISPRGeneEffect rows", len(candidates))
    elif prot.exists():
        ids = pd.read_csv(prot, compression="gzip", nrows=0).columns
        candidates.update(c for c in ids if c.startswith("ACH-"))
        logger.info("Bootstrap resolver: found %d DepMap IDs in proteomics header", len(candidates))
    else:
        raise FileNotFoundError(
            f"Bootstrap resolver: neither CRISPRGeneEffect.csv nor "
            f"protein_quant_current_normalized.csv.gz found under {depmap_dir}. "
            f"Cannot build even a partial model map without DepMap omics."
        )
    df = pd.DataFrame({"depmap_id": sorted(candidates)})
    for col in REQUIRED_MODEL_COLUMNS:
        if col == "model_id":
            df[col] = df["depmap_id"]
        elif col == "ach_id":
            df[col] = df["depmap_id"]
        elif col == "source":
            df[col] = "bootstrap (Model.csv absent)"
        elif col != "depmap_id":
            df[col] = "unknown"
    df = df[list(REQUIRED_MODEL_COLUMNS)]
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote bootstrap model_metadata: %d models -> %s", len(df), out)
    return df


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    n_models_total: int = 0
    n_with_depmap_id: int = 0
    n_with_ccle_name: int = 0
    n_with_sanger_model_id: int = 0
    n_with_cosmic_id: int = 0
    gdsc_mapping_rate: float = 0.0
    prism_mapping_rate: float = 0.0
    n_gdsc_unmapped: int = 0
    n_prism_unmapped: int = 0
    source: str = "unknown"
    unmapped_examples: List[str] = field(default_factory=list)


class ModelIDResolver:
    """Normalises GDSC / PRISM / CCLE / COSMIC IDs to a canonical ``model_id``.

    Parameters
    ----------
    metadata :
        DataFrame conforming to :data:`REQUIRED_MODEL_COLUMNS`.
    """

    def __init__(self, metadata: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in metadata.columns]
        if missing:
            raise ValueError(
                f"ModelIDResolver requires columns {REQUIRED_MODEL_COLUMNS}; missing {missing}"
            )
        self.metadata = metadata.copy()
        self._maps = {
            "depmap_id": dict(zip(metadata["depmap_id"].astype(str), metadata["model_id"])),
            "ccle_name": dict(zip(metadata["ccle_name"].astype(str), metadata["model_id"])),
            "sanger_model_id": dict(zip(metadata["sanger_model_id"].astype(str), metadata["model_id"])),
            "cosmic_id": dict(zip(metadata["cosmic_id"].astype(str), metadata["model_id"])),
            "model_name": dict(zip(metadata["model_name"].astype(str), metadata["model_id"])),
        }
        # Normalise name lookup: strip + uppercase.
        self._name_norm = {
            re.sub(r"[^A-Z0-9]", "", str(n).upper()): mid
            for n, mid in self._maps["model_name"].items()
            if str(n) and n != "unknown"
        }

    def to_model_id(self, raw: str) -> Optional[str]:
        """Map any-namespace identifier to canonical model_id; ``None`` if unmappable."""
        raw_str = str(raw).strip()
        if not raw_str or raw_str in {"nan", "None"}:
            return None
        for namespace_map in self._maps.values():
            if raw_str in namespace_map:
                return namespace_map[raw_str]
        norm = re.sub(r"[^A-Z0-9]", "", raw_str.upper())
        if norm in self._name_norm:
            return self._name_norm[norm]
        return None

    def coverage(
        self,
        *,
        gdsc_models: Optional[Sequence[str]] = None,
        prism_models: Optional[Sequence[str]] = None,
    ) -> CoverageReport:
        rep = CoverageReport(
            n_models_total=len(self.metadata),
            n_with_depmap_id=int(
                self.metadata["depmap_id"].astype(str).str.startswith("ACH").sum()
            ),
            n_with_ccle_name=int(
                (self.metadata["ccle_name"].astype(str) != "unknown").sum()
            ),
            n_with_sanger_model_id=int(
                self.metadata["sanger_model_id"].astype(str).str.startswith("SIDM").sum()
            ),
            n_with_cosmic_id=int(
                pd.to_numeric(self.metadata["cosmic_id"], errors="coerce").notna().sum()
            ),
            source=str(self.metadata["source"].iloc[0]) if len(self.metadata) else "unknown",
        )
        if gdsc_models is not None:
            mapped = [m for m in gdsc_models if self.to_model_id(m)]
            rep.gdsc_mapping_rate = len(mapped) / max(len(gdsc_models), 1)
            rep.n_gdsc_unmapped = len(gdsc_models) - len(mapped)
        if prism_models is not None:
            mapped = [m for m in prism_models if self.to_model_id(m)]
            rep.prism_mapping_rate = len(mapped) / max(len(prism_models), 1)
            rep.n_prism_unmapped = len(prism_models) - len(mapped)
        # Sample some unmapped IDs for the audit trail.
        rep.unmapped_examples = []
        for source_name, source_ids in [("gdsc", gdsc_models or []), ("prism", prism_models or [])]:
            unmapped = [m for m in source_ids if not self.to_model_id(m)][:5]
            for u in unmapped:
                rep.unmapped_examples.append(f"{source_name}:{u}")
        return rep


def write_coverage_report(
    resolver: ModelIDResolver,
    *,
    gdsc_models: Optional[Sequence[str]] = None,
    prism_models: Optional[Sequence[str]] = None,
    out_json: str = "logs/mortfm/model_id_mapping_report.json",
) -> CoverageReport:
    rep = resolver.coverage(gdsc_models=gdsc_models, prism_models=prism_models)
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep.__dict__, f, indent=2)
    logger.info("Wrote model-ID coverage report -> %s", out)
    return rep
