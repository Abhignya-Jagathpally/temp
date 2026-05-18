"""
resistancemap/data/drug_ontology.py
===================================
Drug class + target ontology for MORT-FM.

This is the *minimal* MM-relevant ontology — the source of truth for canonical
drug names, mechanism classes, and primary protein targets. The drug encoder
uses it to expand a :class:`~resistancemap.mortfm.schemas.DrugContext` into
identity / class / target features.

The ontology is intentionally small and curated. For broader coverage (off-MM
indications), use :mod:`resistancemap.data.drug_target_loader` to ingest
DrugBank / ChEMBL / DGIdb dumps and *extend* (not overwrite) this base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from resistancemap.mortfm.schemas import DrugContext


@dataclass(frozen=True)
class DrugOntologyEntry:
    canonical_name: str
    drug_class: str
    target_genes: tuple  # HGNC symbols
    target_proteins: tuple  # UniProt accessions
    synonyms: tuple = ()


# ---------------------------------------------------------------------------
# Curated MM-relevant ontology
# ---------------------------------------------------------------------------

_ENTRIES: Dict[str, DrugOntologyEntry] = {
    e.canonical_name: e for e in [
        # Proteasome inhibitors
        DrugOntologyEntry("Bortezomib", "proteasome_inhibitor",
                          ("PSMB5", "PSMB1", "PSMB2"), ("P28074",),
                          ("Velcade", "BTZ", "PS-341")),
        DrugOntologyEntry("Carfilzomib", "proteasome_inhibitor",
                          ("PSMB5", "PSMB8"), ("P28074", "P28062"),
                          ("Kyprolis", "CFZ")),
        DrugOntologyEntry("Ixazomib", "proteasome_inhibitor",
                          ("PSMB5",), ("P28074",),
                          ("Ninlaro", "IXA")),
        # IMiDs / CRBN binders
        DrugOntologyEntry("Lenalidomide", "IMiD",
                          ("CRBN", "IKZF1", "IKZF3"), ("Q96SW2",),
                          ("Revlimid", "LEN")),
        DrugOntologyEntry("Pomalidomide", "IMiD",
                          ("CRBN", "IKZF1", "IKZF3"), ("Q96SW2",),
                          ("Pomalyst", "POM")),
        DrugOntologyEntry("Thalidomide", "IMiD",
                          ("CRBN", "IKZF1", "IKZF3"), ("Q96SW2",),
                          ("Thalomid",)),
        # HDAC inhibitors
        DrugOntologyEntry("Panobinostat", "HDAC_inhibitor",
                          ("HDAC1", "HDAC2", "HDAC3", "HDAC6"),
                          ("Q13547", "Q92769"), ("Farydak",)),
        DrugOntologyEntry("Vorinostat", "HDAC_inhibitor",
                          ("HDAC1", "HDAC2", "HDAC3", "HDAC6"),
                          ("Q13547",), ("Zolinza", "SAHA")),
        DrugOntologyEntry("Romidepsin", "HDAC_inhibitor",
                          ("HDAC1", "HDAC2", "HDAC3", "HDAC8"),
                          ("Q13547",), ("Istodax",)),
        # BCL2 / BH3 mimetics
        DrugOntologyEntry("Venetoclax", "BCL2_inhibitor",
                          ("BCL2",), ("P10415",), ("Venclexta", "ABT-199")),
        # CDK inhibitors
        DrugOntologyEntry("Dinaciclib", "CDK_inhibitor",
                          ("CDK1", "CDK2", "CDK5", "CDK9"),
                          ("P06493", "P24941"), ("SCH-727965",)),
        DrugOntologyEntry("Palbociclib", "CDK_inhibitor",
                          ("CDK4", "CDK6"), ("P11802", "Q00534"), ("Ibrance",)),
        # Cytotoxics
        DrugOntologyEntry("Doxorubicin", "anthracycline",
                          ("TOP2A", "TOP2B"), ("P11388",), ("Adriamycin",)),
        DrugOntologyEntry("Etoposide", "TOP2_inhibitor",
                          ("TOP2A", "TOP2B"), ("P11388",), ("VP-16",)),
        DrugOntologyEntry("Cyclophosphamide", "alkylating_agent",
                          (), (), ("Cytoxan", "CTX")),
        DrugOntologyEntry("Melphalan", "alkylating_agent",
                          (), (), ("Alkeran",)),
        # mAbs (CD38 / BCMA / SLAMF7 / GPRC5D)
        DrugOntologyEntry("Daratumumab", "anti-CD38_mAb",
                          ("CD38",), ("P28907",), ("Darzalex",)),
        DrugOntologyEntry("Elotuzumab", "anti-SLAMF7_mAb",
                          ("SLAMF7",), ("Q9NQ25",), ("Empliciti",)),
        DrugOntologyEntry("Isatuximab", "anti-CD38_mAb",
                          ("CD38",), ("P28907",), ("Sarclisa",)),
        # Steroid
        DrugOntologyEntry("Dexamethasone", "corticosteroid",
                          ("NR3C1",), ("P04150",), ("Decadron", "DEX")),
        # Selinexor (XPO1 inhibitor) — late-line MM
        DrugOntologyEntry("Selinexor", "XPO1_inhibitor",
                          ("XPO1",), ("O14980",), ("Xpovio",)),
    ]
}

# Build a synonyms -> canonical_name map.
_SYNONYM_LOOKUP: Dict[str, str] = {}
for canon, e in _ENTRIES.items():
    _SYNONYM_LOOKUP[canon.lower()] = canon
    for s in e.synonyms:
        _SYNONYM_LOOKUP[s.lower()] = canon


def lookup_entry(name: str) -> Optional[DrugOntologyEntry]:
    """Return the ontology entry for ``name`` (case-insensitive, synonym-aware)."""
    canon = _SYNONYM_LOOKUP.get(name.strip().lower())
    return _ENTRIES.get(canon) if canon else None


import re as _re

_NORMALIZE_RE = _re.compile(r"[^a-z0-9]+")


def normalize_drug_name(name: str) -> str:
    """Strip parentheticals, punctuation, whitespace; lowercase.

    Used for cross-source matching when a registry stores
    "Bortezomib (Velcade)" but ChEMBL has "BORTEZOMIB".
    Returns "" for None / NaN.
    """
    if name is None:
        return ""
    s = str(name)
    if s.lower() in {"nan", "none"}:
        return ""
    s = _re.sub(r"\([^)]*\)", " ", s)  # drop parenthetical suffixes
    s = _NORMALIZE_RE.sub("", s.lower())
    return s


def lookup_entry_loose(name: str) -> Optional[DrugOntologyEntry]:
    """Like :func:`lookup_entry` but tries a normalized-name fallback.

    First tries the exact synonym lookup; if that misses, retries with
    parentheticals stripped and punctuation collapsed.
    """
    e = lookup_entry(name)
    if e is not None:
        return e
    norm = normalize_drug_name(name)
    if not norm:
        return None
    for entry in _ENTRIES.values():
        if normalize_drug_name(entry.canonical_name) == norm:
            return entry
        for s in entry.synonyms:
            if normalize_drug_name(s) == norm:
                return entry
    return None


def known_classes() -> List[str]:
    """All canonical drug classes in the ontology."""
    return sorted({e.drug_class for e in _ENTRIES.values()})


def known_drugs() -> List[str]:
    return sorted(_ENTRIES.keys())


def materialise_drug_context(
    name: str,
    *,
    dose: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    combination_id: Optional[str] = None,
) -> DrugContext:
    """Make a :class:`DrugContext` from a drug name, looking up class+targets.

    If the name is not in the ontology, returns a ``DrugContext`` with class
    set to ``"unknown"`` and empty target lists — but still a usable record.
    """
    entry = lookup_entry(name)
    if entry is None:
        return DrugContext(
            drug_name=name,
            drug_class="unknown",
            target_genes=[],
            target_proteins=[],
            dose=dose,
            start_time=start_time,
            end_time=end_time,
            combination_id=combination_id,
        )
    return DrugContext(
        drug_name=entry.canonical_name,
        drug_class=entry.drug_class,
        target_genes=list(entry.target_genes),
        target_proteins=list(entry.target_proteins),
        dose=dose,
        start_time=start_time,
        end_time=end_time,
        combination_id=combination_id,
    )


def class_to_index() -> Dict[str, int]:
    """Stable integer encoding for drug classes (used by the drug encoder)."""
    classes = ["unknown"] + known_classes()
    return {c: i for i, c in enumerate(classes)}


def drug_to_index() -> Dict[str, int]:
    """Stable integer encoding for drug identities."""
    drugs = ["unknown"] + known_drugs()
    return {d: i for i, d in enumerate(drugs)}
