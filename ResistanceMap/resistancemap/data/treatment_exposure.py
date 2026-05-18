"""
resistancemap/data/treatment_exposure.py
========================================
Convert MMRF treatment regimen tables into per-snapshot
:class:`~resistancemap.mortfm.schemas.DrugContext` records.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from resistancemap.data.drug_ontology import lookup_entry, materialise_drug_context
from resistancemap.mortfm.schemas import DrugContext

logger = logging.getLogger(__name__)


def _split_regimen_name(regimen: str) -> List[str]:
    """Split a regimen string like 'VRd' or 'Daratumumab + Bortezomib + Dex' into drug names."""
    if regimen is None or (isinstance(regimen, float)):
        return []
    s = str(regimen).strip()
    if not s:
        return []
    # Try common abbreviations first.
    abbrev_expansions = {
        "VRD": ["Bortezomib", "Lenalidomide", "Dexamethasone"],
        "VRd": ["Bortezomib", "Lenalidomide", "Dexamethasone"],
        "VCD": ["Bortezomib", "Cyclophosphamide", "Dexamethasone"],
        "CyBorD": ["Cyclophosphamide", "Bortezomib", "Dexamethasone"],
        "KRd": ["Carfilzomib", "Lenalidomide", "Dexamethasone"],
        "DaraVRd": ["Daratumumab", "Bortezomib", "Lenalidomide", "Dexamethasone"],
        "DaraRd": ["Daratumumab", "Lenalidomide", "Dexamethasone"],
        "Pd": ["Pomalidomide", "Dexamethasone"],
        "EPd": ["Elotuzumab", "Pomalidomide", "Dexamethasone"],
    }
    if s in abbrev_expansions:
        return abbrev_expansions[s]
    # Split on common separators.
    parts = re.split(r"[+/,]| and ", s)
    drugs: List[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        # Look up; if found, use canonical name; otherwise keep raw.
        entry = lookup_entry(token)
        drugs.append(entry.canonical_name if entry else token)
    return drugs


def parse_treatment_exposures(
    treatment_df: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    regimen_col: str = "regimen_name",
    line_col: Optional[str] = "line_number",
    start_day_col: Optional[str] = "trtstdy",
    end_day_col: Optional[str] = "trtendy",
) -> Dict[str, List[DrugContext]]:
    """Group treatment rows by patient and emit ``DrugContext`` lists.

    Each row in ``treatment_df`` represents one regimen/line. Combination
    therapies expand to one ``DrugContext`` per component drug, with a shared
    ``combination_id`` so the model can recognise them as co-administered.
    """
    if patient_id_col not in treatment_df.columns:
        raise KeyError(f"Treatment dataframe missing {patient_id_col!r}")
    if regimen_col not in treatment_df.columns:
        raise KeyError(f"Treatment dataframe missing {regimen_col!r}")

    out: Dict[str, List[DrugContext]] = {}
    for _, row in treatment_df.iterrows():
        pid = str(row[patient_id_col])
        regimen = str(row[regimen_col])
        line = int(row[line_col]) if line_col and line_col in row and pd.notna(row[line_col]) else 1
        start = None
        end = None
        if start_day_col and start_day_col in row and pd.notna(row[start_day_col]):
            start = float(row[start_day_col]) / 30.44
        if end_day_col and end_day_col in row and pd.notna(row[end_day_col]):
            end = float(row[end_day_col]) / 30.44

        components = _split_regimen_name(regimen)
        combination_id = f"{regimen}__line{line}"
        for comp in components:
            ctx = materialise_drug_context(
                comp,
                dose=None,
                start_time=start,
                end_time=end,
                combination_id=combination_id,
            )
            out.setdefault(pid, []).append(ctx)

    logger.info(
        "parse_treatment_exposures: %d patients, %d total exposure rows",
        len(out),
        sum(len(v) for v in out.values()),
    )
    return out


def primary_drug_for_patient(
    exposures: Dict[str, List[DrugContext]],
    patient_id: str,
    *,
    at_time: Optional[float] = None,
) -> Optional[DrugContext]:
    """Pick the most-recent active drug for ``patient_id`` at ``at_time``.

    "Active" means ``start_time <= at_time <= end_time`` (or unspecified bounds).
    Returns ``None`` if no exposure is recorded.
    """
    rows = exposures.get(patient_id, [])
    if not rows:
        return None
    if at_time is None:
        return rows[-1]
    candidates = []
    for r in rows:
        s = r.start_time if r.start_time is not None else -1e9
        e = r.end_time if r.end_time is not None else 1e9
        if s <= at_time <= e:
            candidates.append(r)
    return candidates[-1] if candidates else rows[-1]
