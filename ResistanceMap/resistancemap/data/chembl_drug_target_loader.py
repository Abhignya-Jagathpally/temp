"""
resistancemap/data/chembl_drug_target_loader.py
===============================================
ChEMBL drug→target loader with two input modes:

* **SQLite mode** — reads ``chembl_*.sqlite`` and runs a curated SQL query to
  extract ``(molecule_chembl_id, target_chembl_id, uniprot_id, gene_symbol,
  assay_type, standard_type, standard_value, standard_units, confidence_score)``.
* **CSV mode** — reads a pre-flattened CSV exported via the ChEMBL portal
  (or via the live ``mcp__claude_ai_ChEMBL__get_mechanism`` MCP tool).

Both modes produce the same canonical schema. The Block-A graph builder
consumes either.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


CHEMBL_TARGET_COLUMNS = (
    "drug_name",
    "drug_id",
    "chembl_id",
    "target_gene",
    "target_uniprot",
    "target_type",
    "assay_type",
    "standard_type",
    "standard_value",
    "standard_units",
    "confidence_score",
    "source",
)


# ---------------------------------------------------------------------------
# SQLite mode
# ---------------------------------------------------------------------------


_SQLITE_QUERY = """
SELECT
    md.pref_name AS drug_name,
    md.chembl_id AS chembl_id,
    cs.accession AS target_uniprot,
    cs.component_synonym AS target_gene,
    td.target_type AS target_type,
    a.assay_type AS assay_type,
    act.standard_type AS standard_type,
    act.standard_value AS standard_value,
    act.standard_units AS standard_units,
    a.confidence_score AS confidence_score
FROM molecule_dictionary md
JOIN activities act ON act.molregno = md.molregno
JOIN assays a ON a.assay_id = act.assay_id
JOIN target_dictionary td ON td.tid = a.tid
JOIN target_components tc ON tc.tid = td.tid
JOIN component_sequences cs ON cs.component_id = tc.component_id
WHERE td.organism = 'Homo sapiens'
  AND act.standard_relation IN ('=', '<')
  AND act.standard_type IN ('IC50','Ki','Kd','EC50','Potency')
  AND md.max_phase >= 1
ORDER BY md.chembl_id
"""


def load_from_sqlite(sqlite_path: str, *, limit: Optional[int] = None) -> pd.DataFrame:
    """Run the curated drug-target query against a ChEMBL SQLite dump.

    Returns a DataFrame with :data:`CHEMBL_TARGET_COLUMNS`. Note: the ChEMBL
    SQLite dump is ~5 GB; running the query the first time may take several
    minutes.
    """
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(
            f"ChEMBL SQLite not found at {p.resolve()}. "
            f"Download from https://chembl.gitbook.io/chembl-interface-documentation/downloads."
        )
    q = _SQLITE_QUERY + (f"\nLIMIT {int(limit)}" if limit else "")
    con = sqlite3.connect(str(p))
    try:
        df = pd.read_sql_query(q, con)
    finally:
        con.close()
    df["drug_id"] = df["chembl_id"]
    df["source"] = "ChEMBL SQLite"
    df = df[list(CHEMBL_TARGET_COLUMNS)]
    logger.info("ChEMBL SQLite: %d drug-target rows", len(df))
    return df


# ---------------------------------------------------------------------------
# CSV mode
# ---------------------------------------------------------------------------


def load_from_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"ChEMBL drug-target CSV not found at {p.resolve()}. "
            f"Generate via ChEMBL portal export or live MCP, place under data/raw/chembl/."
        )
    raw = pd.read_csv(p, low_memory=False, dtype=str)
    rename = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in {"molecule_chembl_id", "drug_chembl_id"}: rename[c] = "chembl_id"
        elif cl in {"pref_name", "drug_name"}: rename[c] = "drug_name"
        elif cl in {"accession", "target_uniprot", "uniprot_id"}: rename[c] = "target_uniprot"
        elif cl in {"target_gene", "gene_symbol", "component_synonym"}: rename[c] = "target_gene"
        elif cl in {"target_type"}: rename[c] = "target_type"
        elif cl in {"assay_type"}: rename[c] = "assay_type"
        elif cl in {"standard_type"}: rename[c] = "standard_type"
        elif cl in {"standard_value", "value"}: rename[c] = "standard_value"
        elif cl in {"standard_units", "units"}: rename[c] = "standard_units"
        elif cl in {"confidence_score"}: rename[c] = "confidence_score"
    raw = raw.rename(columns=rename)
    if "chembl_id" not in raw.columns or "target_uniprot" not in raw.columns:
        raise ValueError(
            f"ChEMBL CSV {p.name} must have molecule_chembl_id + target_uniprot columns "
            f"after normalisation. Have: {list(raw.columns)[:15]}"
        )
    for required in CHEMBL_TARGET_COLUMNS:
        if required not in raw.columns:
            raw[required] = "" if required != "standard_value" else None
    raw["drug_id"] = raw["chembl_id"]
    raw["source"] = "ChEMBL CSV"
    return raw[list(CHEMBL_TARGET_COLUMNS)]


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def write_chembl_coverage_report(
    *,
    drug_targets: pd.DataFrame,
    gdsc_drugs: Sequence[str],
    prism_drugs: Sequence[str],
    out_json: str = "logs/mortfm/drug_target_coverage_report.json",
) -> Dict[str, float]:
    # The processed CSV may carry either ``drug_name`` (when the source export
    # included one) or only ``drug_id`` / ``chembl_id``. Honour whichever exists.
    name_col = "drug_name" if "drug_name" in drug_targets.columns else None
    drugs_with_targets = (
        set(drug_targets[name_col].astype(str).str.lower())
        if name_col is not None else set()
    )
    chembl_ids = set(drug_targets["chembl_id"].astype(str)) if "chembl_id" in drug_targets.columns else set()
    drug_ids = set(drug_targets["drug_id"].astype(str)) if "drug_id" in drug_targets.columns else set()

    def _coverage(drug_list: Sequence[str]) -> float:
        if not drug_list:
            return 0.0
        mapped = sum(
            1 for d in drug_list
            if (str(d).lower() in drugs_with_targets)
            or (str(d) in chembl_ids)
            or (str(d) in drug_ids)
        )
        return mapped / len(drug_list)

    rep = {
        "n_drug_target_rows": int(len(drug_targets)),
        "n_unique_drugs": int(drug_targets["drug_id"].nunique()) if "drug_id" in drug_targets.columns else 0,
        "n_unique_targets": int(drug_targets["target_uniprot"].nunique()) if "target_uniprot" in drug_targets.columns else 0,
        "n_gdsc_drugs": len(gdsc_drugs),
        "n_prism_drugs": len(prism_drugs),
        "gdsc_drug_target_coverage": _coverage(gdsc_drugs),
        "prism_drug_target_coverage": _coverage(prism_drugs),
        "evidence_source": "ChEMBL",
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    logger.info("Wrote ChEMBL drug-target coverage report -> %s", out)
    return rep
