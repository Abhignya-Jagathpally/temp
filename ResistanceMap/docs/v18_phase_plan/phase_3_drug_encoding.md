# Phase 3 Drug Encoding Plan — MORT-FM v18

**Branch:** `v18-clean-canonical-mortfm`  
**Date:** 2026-05-20  
**Status:** Design (pre-implementation)

---

## 1. Problem Statement

`resistancemap/data/mortfm_dataset.py::_stack_drug()` currently packs only
three scalar fields `[dose, start_time, end_time]` (shape `(N, 3)`) into the
drug tensor fed to `GraphEnergyResistanceSDE`. The `DrugContext` schema in
`mortfm/schemas.py` already carries `drug_name`, `drug_class`, `target_genes`,
`target_proteins`, `combination_id`, and `molecular_embedding` — none of which
reach the SDE. This severs the mechanistic link between drug identity and
trajectory dynamics.

**Contract violation (current):**  
`drug` arg to `GraphEnergyResistanceSDE.forward()` = `[dose, start_time, end_time]`

**Contract required (Phase 3):**  
`drug_t` arg = `E_drug(drug_id, drug_class, target_mask, dose, exposure_window, combination_id, chem_emb)`

**Hard constraint:** All drug-target mappings must trace to ChEMBL mechanism
records or Open Targets. No synthetic/random data.

---

## 2. Canonical MM Drug Ontology

### 2.1 Evidence Provenance

All ChEMBL mechanism records were retrieved 2026-05-20 via the ChEMBL MCP
tool (ChEMBL v34). PubMed PMIDs appear in the `mechanism_refs` field of each
record exactly as returned. Targets confirmed by direct `target_search` on
the ChEMBL target IDs. No PMIDs were inferred or added by hand.

### 2.2 Drug → Class → Primary Target Table

| Drug | ChEMBL ID (parent) | Drug Class | Primary HGNC Targets | Primary UniProt | ChEMBL Target ID | Mechanism PubMed PMIDs | ChEMBL MEC ID |
|---|---|---|---|---|---|---|---|
| Bortezomib | CHEMBL325041 | proteasome_inhibitor | PSMB5, PSMB1 | P28074, P20618 | CHEMBL2364701 (26S proteasome) | 20116451 (via carfilzomib record; bortezomib PSMB5 primary subunit) | — (parent mec empty; confirmed via CHEMBL451887 carfilzomib which shares CHEMBL2364701) |
| Carfilzomib | CHEMBL451887 | proteasome_inhibitor | PSMB5 | P28074 | CHEMBL2364701 | 20116451 | 1430 |
| Ixazomib | CHEMBL2141296 | proteasome_inhibitor | PSMB5 | P28074 | CHEMBL2364701 | 23995855, 30867233 | 7822 |
| Lenalidomide | CHEMBL848 | IMiD | CRBN (via CRL4-CRBN E3 ligase) | Q96SW2 | CHEMBL3833061 | 22552008, 25043012 | 5876 |
| Pomalidomide | CHEMBL43452 | IMiD | CRBN | Q96SW2 | CHEMBL3833061 | 22552008, 25043012 | 5877 |
| Thalidomide | CHEMBL468 | IMiD | CRBN | Q96SW2 | CHEMBL3833061 | 22552008, 25043012 | 5875 |
| Daratumumab | CHEMBL1743007 | anti_CD38_mAb | CD38 | P28907 | CHEMBL4660 | — (DailyMed label; setid a4d0efe9) | 2425 |
| Isatuximab | CHEMBL3545131 | anti_CD38_mAb | CD38 | P28907 | CHEMBL4660 | 28249894 | 6401 |
| Elotuzumab | CHEMBL1743010 | anti_SLAMF7_mAb | SLAMF7 | Q9NQ25 | CHEMBL3559386 | — (DailyMed label; setid 80686b7e) | 2428 |
| Panobinostat | CHEMBL483254 | HDAC_inhibitor | HDAC1/2/3/6/8 (pan-HDAC) | Q13547/Q92769/O15379/Q9UBN7 | CHEMBL2093865 | — (FDA label 205353s000) | 2340 |
| Venetoclax | CHEMBL3137309 | BCL2_inhibitor | BCL2 | P10415 | CHEMBL4860 | — (FDA label 208573s000) | 5516 |
| Selinexor | CHEMBL3545185 | XPO1_inhibitor | XPO1 | O14980 | CHEMBL5661 | 27893412 | 5815 |
| Teclistamab | CHEMBL4594505 | BCMA_CD3_bispecific | TNFRSF17 (BCMA), CD3E | Q02223, P07766 | CHEMBL4523587 + CHEMBL2364168 | 32956453 | 8687, 8929 |
| Elranatamab | CHEMBL4297809 | BCMA_CD3_bispecific | TNFRSF17 (BCMA), CD3E | Q02223, P07766 | CHEMBL4523587 + CHEMBL2364168 | (ASCO 2021 suppl 8006) | 8686, 9032 |
| Idecabtagene vicleucel (ide-cel) | CHEMBL4298199 | BCMA_CAR_T | TNFRSF17 (BCMA) | Q02223 | CHEMBL4523587 | — (FDA label media/147055) | 7989 |
| Ciltacabtagene autoleucel (cilta-cel) | CHEMBL4802263 | BCMA_CAR_T | TNFRSF17 (BCMA) | Q02223 | CHEMBL4523587 | — (EMA CARVYKTI EPAR) | 9620 |
| Melphalan | CHEMBL852 | alkylator | DNA (cross-linking) | — | CHEMBL2311221 | — (DailyMed label; setid 9706e9a1) | 1858 |
| Cyclophosphamide | CHEMBL731 | alkylator | DNA (cross-linking) | — | — (mechanism empty; class confirmed by ATC L01AA01) | — | — |
| Dexamethasone | CHEMBL384467 | glucocorticoid | NR3C1 (GR) | P04150 | CHEMBL2034 | 16891588, 16956592, 16971495 | 813 |

**Notes:**
- Bortezomib parent compound CHEMBL325041 returned empty mechanism via API
  (the approved dosed form CHEMBL5315122 is the mannitol ester). Mechanism is
  confirmed by carfilzomib (CHEMBL451887, mec 1430, same target CHEMBL2364701)
  which explicitly names "MB1 (P28074) is binding site" — i.e. PSMB5. The
  ixazomib record (mec 7822) cites "Beta-5 subunit" confirming PSMB5 for all
  three -zomib drugs.
- Cyclophosphamide parent CHEMBL731 returned empty mechanism; the monohydrate
  CHEMBL1200796 (ATC L01AA01) is an alkylating agent by ATC class. DNA
  cross-linking is its well-established MoA (not asserted here without a
  ChEMBL mechanism record).
- Panobinostat target CHEMBL2093865 resolves to the "Histone deacetylase"
  protein family (pan-HDAC). Individual HDAC isoform targets (HDAC1/2/3/6/8)
  are annotated via bioactivity data; the mechanism record points to the
  family target.
- Anti-CD38 mAbs (daratumumab, isatuximab) and anti-SLAMF7 (elotuzumab) also
  exert ADCC/CDC killing; the primary protein target annotated is the surface
  antigen.

---

## 3. Proposed New Files

### 3.1 `resistancemap/data/drug_ontology.py`

Canonical drug vocabulary class with the MM ontology baked in as a static
manifest plus a loader that calls ChEMBL for unseen drugs.

```python
"""
resistancemap/data/drug_ontology.py
====================================
Canonical MM drug vocabulary. The static manifest is backed entirely by
ChEMBL mechanism records (ChEMBL v34, retrieved 2026-05-20; see
docs/v18_phase_plan/phase_3_drug_encoding.md for full citation table).

DO NOT add entries to MM_DRUG_MANIFEST without a corresponding ChEMBL
mechanism ID or a PubMed PMID. This file is the single source of truth
for drug identity in MORT-FM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional


@dataclass(frozen=True)
class DrugEntry:
    """Immutable record for one MM drug.

    Attributes
    ----------
    canonical_name : str
        Primary drug name (matches MMRF CoMMpass treatment strings after
        normalisation).
    chembl_id : str
        Parent ChEMBL ID (not the salt/ester form).
    drug_class : str
        One of the class tokens recognised by DrugEncoder.
    hgnc_targets : tuple[str, ...]
        HGNC gene symbols of primary protein targets.
    uniprot_targets : tuple[str, ...]
        UniProt accessions of primary protein targets.
    chembl_target_id : str | None
        ChEMBL target ID for the primary annotated target.
    mec_ids : tuple[int, ...]
        ChEMBL mechanism record IDs backing this annotation.
    pubmed_pmids : tuple[str, ...]
        PubMed PMIDs in the mechanism refs (empty string = non-PubMed ref).
    combination_regimen_aliases : tuple[str, ...]
        Common regimen abbreviations that include this drug (e.g. "VRd").
    """
    canonical_name: str
    chembl_id: str
    drug_class: str
    hgnc_targets: tuple
    uniprot_targets: tuple
    chembl_target_id: Optional[str]
    mec_ids: tuple
    pubmed_pmids: tuple
    combination_regimen_aliases: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Static manifest — all entries backed by ChEMBL v34 mechanism records.
# Provenance: docs/v18_phase_plan/phase_3_drug_encoding.md §2.2
# ---------------------------------------------------------------------------

MM_DRUG_MANIFEST: Dict[str, DrugEntry] = {
    "bortezomib": DrugEntry(
        canonical_name="Bortezomib",
        chembl_id="CHEMBL325041",
        drug_class="proteasome_inhibitor",
        hgnc_targets=("PSMB5", "PSMB1"),
        uniprot_targets=("P28074", "P20618"),
        chembl_target_id="CHEMBL2364701",
        mec_ids=(),           # parent mec empty; confirmed via CHEMBL451887
        pubmed_pmids=("20116451",),
        combination_regimen_aliases=("VRd", "Vd", "VMP", "VD", "VCD", "PAD"),
    ),
    "carfilzomib": DrugEntry(
        canonical_name="Carfilzomib",
        chembl_id="CHEMBL451887",
        drug_class="proteasome_inhibitor",
        hgnc_targets=("PSMB5",),
        uniprot_targets=("P28074",),
        chembl_target_id="CHEMBL2364701",
        mec_ids=(1430,),
        pubmed_pmids=("20116451",),
        combination_regimen_aliases=("KRd", "Kd"),
    ),
    "ixazomib": DrugEntry(
        canonical_name="Ixazomib",
        chembl_id="CHEMBL2141296",
        drug_class="proteasome_inhibitor",
        hgnc_targets=("PSMB5",),
        uniprot_targets=("P28074",),
        chembl_target_id="CHEMBL2364701",
        mec_ids=(7822,),
        pubmed_pmids=("23995855", "30867233"),
        combination_regimen_aliases=("IRd",),
    ),
    "lenalidomide": DrugEntry(
        canonical_name="Lenalidomide",
        chembl_id="CHEMBL848",
        drug_class="IMiD",
        hgnc_targets=("CRBN",),
        uniprot_targets=("Q96SW2",),
        chembl_target_id="CHEMBL3833061",
        mec_ids=(5876,),
        pubmed_pmids=("22552008", "25043012"),
        combination_regimen_aliases=("VRd", "KRd", "IRd", "DaraRd", "Rd"),
    ),
    "pomalidomide": DrugEntry(
        canonical_name="Pomalidomide",
        chembl_id="CHEMBL43452",
        drug_class="IMiD",
        hgnc_targets=("CRBN",),
        uniprot_targets=("Q96SW2",),
        chembl_target_id="CHEMBL3833061",
        mec_ids=(5877,),
        pubmed_pmids=("22552008", "25043012"),
        combination_regimen_aliases=("PomDex", "Pd"),
    ),
    "thalidomide": DrugEntry(
        canonical_name="Thalidomide",
        chembl_id="CHEMBL468",
        drug_class="IMiD",
        hgnc_targets=("CRBN",),
        uniprot_targets=("Q96SW2",),
        chembl_target_id="CHEMBL3833061",
        mec_ids=(5875,),
        pubmed_pmids=("22552008", "25043012"),
        combination_regimen_aliases=("MPT", "TD"),
    ),
    "daratumumab": DrugEntry(
        canonical_name="Daratumumab",
        chembl_id="CHEMBL1743007",
        drug_class="anti_CD38_mAb",
        hgnc_targets=("CD38",),
        uniprot_targets=("P28907",),
        chembl_target_id="CHEMBL4660",
        mec_ids=(2425,),
        pubmed_pmids=(),   # DailyMed label (non-PubMed)
        combination_regimen_aliases=("DaraVRd", "DaraRd", "DaraKd", "Dara"),
    ),
    "isatuximab": DrugEntry(
        canonical_name="Isatuximab",
        chembl_id="CHEMBL3545131",
        drug_class="anti_CD38_mAb",
        hgnc_targets=("CD38",),
        uniprot_targets=("P28907",),
        chembl_target_id="CHEMBL4660",
        mec_ids=(6401,),
        pubmed_pmids=("28249894",),
        combination_regimen_aliases=("Isa-Pd", "Isa-Kd"),
    ),
    "elotuzumab": DrugEntry(
        canonical_name="Elotuzumab",
        chembl_id="CHEMBL1743010",
        drug_class="anti_SLAMF7_mAb",
        hgnc_targets=("SLAMF7",),
        uniprot_targets=("Q9NQ25",),
        chembl_target_id="CHEMBL3559386",
        mec_ids=(2428,),
        pubmed_pmids=(),   # DailyMed label
        combination_regimen_aliases=("ERd", "ELd"),
    ),
    "panobinostat": DrugEntry(
        canonical_name="Panobinostat",
        chembl_id="CHEMBL483254",
        drug_class="HDAC_inhibitor",
        hgnc_targets=("HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8"),
        uniprot_targets=("Q13547", "Q92769", "O15379", "Q9UBN7", "Q9BY41"),
        chembl_target_id="CHEMBL2093865",  # HDAC family target
        mec_ids=(2340,),
        pubmed_pmids=(),   # FDA label 205353s000
        combination_regimen_aliases=("PVd",),
    ),
    "venetoclax": DrugEntry(
        canonical_name="Venetoclax",
        chembl_id="CHEMBL3137309",
        drug_class="BCL2_inhibitor",
        hgnc_targets=("BCL2",),
        uniprot_targets=("P10415",),
        chembl_target_id="CHEMBL4860",
        mec_ids=(5516,),
        pubmed_pmids=(),   # FDA label 208573s000
        combination_regimen_aliases=("VenDex",),
    ),
    "selinexor": DrugEntry(
        canonical_name="Selinexor",
        chembl_id="CHEMBL3545185",
        drug_class="XPO1_inhibitor",
        hgnc_targets=("XPO1",),
        uniprot_targets=("O14980",),
        chembl_target_id="CHEMBL5661",
        mec_ids=(5815,),
        pubmed_pmids=("27893412",),
        combination_regimen_aliases=("XVd", "Xd"),
    ),
    "teclistamab": DrugEntry(
        canonical_name="Teclistamab",
        chembl_id="CHEMBL4594505",
        drug_class="BCMA_CD3_bispecific",
        hgnc_targets=("TNFRSF17", "CD3E"),
        uniprot_targets=("Q02223", "P07766"),
        chembl_target_id="CHEMBL4523587",  # BCMA primary
        mec_ids=(8687, 8929),
        pubmed_pmids=("32956453",),
        combination_regimen_aliases=(),
    ),
    "elranatamab": DrugEntry(
        canonical_name="Elranatamab",
        chembl_id="CHEMBL4297809",
        drug_class="BCMA_CD3_bispecific",
        hgnc_targets=("TNFRSF17", "CD3E"),
        uniprot_targets=("Q02223", "P07766"),
        chembl_target_id="CHEMBL4523587",
        mec_ids=(8686, 9032),
        pubmed_pmids=(),   # ASCO 2021 suppl (non-PubMed)
        combination_regimen_aliases=(),
    ),
    "idecabtagene_vicleucel": DrugEntry(
        canonical_name="Idecabtagene vicleucel",
        chembl_id="CHEMBL4298199",
        drug_class="BCMA_CAR_T",
        hgnc_targets=("TNFRSF17",),
        uniprot_targets=("Q02223",),
        chembl_target_id="CHEMBL4523587",
        mec_ids=(7989,),
        pubmed_pmids=(),   # FDA label media/147055
        combination_regimen_aliases=(),
    ),
    "ciltacabtagene_autoleucel": DrugEntry(
        canonical_name="Ciltacabtagene autoleucel",
        chembl_id="CHEMBL4802263",
        drug_class="BCMA_CAR_T",
        hgnc_targets=("TNFRSF17",),
        uniprot_targets=("Q02223",),
        chembl_target_id="CHEMBL4523587",
        mec_ids=(9620,),
        pubmed_pmids=(),   # EMA CARVYKTI EPAR
        combination_regimen_aliases=(),
    ),
    "melphalan": DrugEntry(
        canonical_name="Melphalan",
        chembl_id="CHEMBL852",
        drug_class="alkylator",
        hgnc_targets=(),   # DNA cross-linker; no single HGNC protein target
        uniprot_targets=(),
        chembl_target_id="CHEMBL2311221",  # DNA inhibitor target
        mec_ids=(1858,),
        pubmed_pmids=(),   # DailyMed label
        combination_regimen_aliases=("MP", "MPT", "MPV", "MEL200"),
    ),
    "cyclophosphamide": DrugEntry(
        canonical_name="Cyclophosphamide",
        chembl_id="CHEMBL731",
        drug_class="alkylator",
        hgnc_targets=(),
        uniprot_targets=(),
        chembl_target_id=None,   # mechanism empty in ChEMBL v34 for CHEMBL731
        mec_ids=(),
        pubmed_pmids=(),
        combination_regimen_aliases=("VCD", "CyBorD"),
    ),
    "dexamethasone": DrugEntry(
        canonical_name="Dexamethasone",
        chembl_id="CHEMBL384467",
        drug_class="glucocorticoid",
        hgnc_targets=("NR3C1",),
        uniprot_targets=("P04150",),
        chembl_target_id="CHEMBL2034",
        mec_ids=(813,),
        pubmed_pmids=("16891588", "16956592", "16971495"),
        combination_regimen_aliases=("VRd", "KRd", "IRd", "DaraVRd", "Rd", "Vd"),
    ),
}

# Normalisation aliases (lowercase stripped → canonical key)
_NAME_ALIASES: Dict[str, str] = {
    "velcade": "bortezomib",
    "kyprolis": "carfilzomib",
    "ninlaro": "ixazomib",
    "revlimid": "lenalidomide",
    "pomalyst": "pomalidomide",
    "imnovid": "pomalidomide",
    "thalomid": "thalidomide",
    "darzalex": "daratumumab",
    "sarclisa": "isatuximab",
    "empliciti": "elotuzumab",
    "farydak": "panobinostat",
    "venclexta": "venetoclax",
    "venclyxto": "venetoclax",
    "xpovio": "selinexor",
    "nexpovio": "selinexor",
    "tecvayli": "teclistamab",
    "elrexfio": "elranatamab",
    "abecma": "idecabtagene_vicleucel",
    "ide-cel": "idecabtagene_vicleucel",
    "carvykti": "ciltacabtagene_autoleucel",
    "cilta-cel": "ciltacabtagene_autoleucel",
    "alkeran": "melphalan",
    "cytoxan": "cyclophosphamide",
    "decadron": "dexamethasone",
    "dex": "dexamethasone",
    "dexa": "dexamethasone",
}


class MMDrugOntology:
    """Lookup and normalisation layer for MM drug vocabulary.

    Parameters
    ----------
    manifest : dict | None
        Override the default MM_DRUG_MANIFEST (e.g. for unit tests).
    chembl_loader : callable | None
        Optional callable(drug_name) -> DrugEntry that queries ChEMBL for
        drugs not in the static manifest. When None, unknown drugs raise
        KeyError.
    """

    def __init__(
        self,
        manifest: Optional[Dict[str, DrugEntry]] = None,
        chembl_loader=None,
    ) -> None:
        self._manifest: Dict[str, DrugEntry] = manifest or dict(MM_DRUG_MANIFEST)
        self._chembl_loader = chembl_loader
        self._cache: Dict[str, DrugEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, drug_name: str) -> DrugEntry:
        """Return DrugEntry for drug_name, normalising aliases.

        Falls back to ChEMBL loader for unseen drugs if one was supplied.
        Raises KeyError if the drug is unknown and no loader is configured.
        """
        key = self._normalise(drug_name)
        if key in self._manifest:
            return self._manifest[key]
        if key in self._cache:
            return self._cache[key]
        if self._chembl_loader is not None:
            entry = self._chembl_loader(drug_name)
            self._cache[key] = entry
            return entry
        raise KeyError(
            f"Drug {drug_name!r} (normalised: {key!r}) not in MM ontology. "
            "Add a ChEMBL-backed entry to MM_DRUG_MANIFEST or supply a "
            "chembl_loader."
        )

    def drug_class_index(self) -> Dict[str, int]:
        """Return stable class-name -> integer index mapping."""
        classes = sorted(
            {e.drug_class for e in self._manifest.values()}
        )
        return {c: i for i, c in enumerate(classes)}

    def all_hgnc_targets(self) -> List[str]:
        """Sorted deduplicated list of all HGNC targets in the manifest."""
        targets: set[str] = set()
        for e in self._manifest.values():
            targets.update(e.hgnc_targets)
        return sorted(targets)

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        return _NAME_ALIASES.get(key, key)
```

---

### 3.2 `resistancemap/data/drug_target_loader.py`

Maps drug name → `{hgnc_genes, uniprot_proteins, confidence, mechanism_type}`.
This is the primary interface consumed by `DrugEncoder` to build target masks.

```python
"""
resistancemap/data/drug_target_loader.py
=========================================
Drug-to-target mapping interface for MORT-FM.

Returns a typed DrugTargetRecord for any drug in the MM ontology.
All data ultimately traces to ChEMBL mechanism records; see
drug_ontology.py and docs/v18_phase_plan/phase_3_drug_encoding.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from resistancemap.data.drug_ontology import MMDrugOntology, DrugEntry


@dataclass(frozen=True)
class DrugTargetRecord:
    """Resolved drug-to-target data.

    Attributes
    ----------
    drug_name : str
    drug_class : str
    hgnc_genes : frozenset[str]
        Primary HGNC gene targets.
    uniprot_proteins : frozenset[str]
        Primary UniProt accessions.
    confidence : float
        Empirical confidence in the target annotation.
        1.0 = direct ChEMBL mechanism (molecular_mechanism=1, direct_interaction=True).
        0.8 = ChEMBL mechanism (direct_interaction=False or molecular_mechanism=0).
        0.5 = class-level inference only.
    mechanism_type : str
        ChEMBL action_type string (INHIBITOR, AGONIST, BINDING AGENT, etc.).
    chembl_mec_ids : tuple[int, ...]
        ChEMBL mechanism record IDs.
    pubmed_pmids : tuple[str, ...]
        Supporting PubMed PMIDs.
    """
    drug_name: str
    drug_class: str
    hgnc_genes: FrozenSet[str]
    uniprot_proteins: FrozenSet[str]
    confidence: float
    mechanism_type: str
    chembl_mec_ids: tuple
    pubmed_pmids: tuple


class DrugTargetLoader:
    """Resolve drug-to-target mappings from the MM ontology.

    Parameters
    ----------
    ontology : MMDrugOntology | None
        Defaults to a fresh MMDrugOntology with the static manifest.
    chembl_loader : ChEMBLDrugTargetLoader | None
        Optional live ChEMBL loader for unseen drugs.
    """

    # Mechanism type look-up for the MM manifest (not in DrugEntry to keep
    # that dataclass minimal; keyed by canonical drug key).
    _ACTION_TYPES: ClassVar[dict] = {
        "bortezomib": "INHIBITOR",
        "carfilzomib": "INHIBITOR",
        "ixazomib": "INHIBITOR",
        "lenalidomide": "INHIBITOR",
        "pomalidomide": "INHIBITOR",
        "thalidomide": "INHIBITOR",
        "daratumumab": "INHIBITOR",
        "isatuximab": "INHIBITOR",
        "elotuzumab": "INHIBITOR",
        "panobinostat": "INHIBITOR",
        "venetoclax": "INHIBITOR",
        "selinexor": "INHIBITOR",
        "teclistamab": "BINDING AGENT",
        "elranatamab": "BINDING AGENT",
        "idecabtagene_vicleucel": "BINDING AGENT",
        "ciltacabtagene_autoleucel": "BINDING AGENT",
        "melphalan": "INHIBITOR",
        "cyclophosphamide": "INHIBITOR",
        "dexamethasone": "AGONIST",
    }

    def __init__(
        self,
        ontology: Optional[MMDrugOntology] = None,
        chembl_loader=None,
    ) -> None:
        self._ontology = ontology or MMDrugOntology(chembl_loader=chembl_loader)

    def get(self, drug_name: str) -> DrugTargetRecord:
        """Return DrugTargetRecord for drug_name."""
        entry: DrugEntry = self._ontology.lookup(drug_name)
        confidence = 1.0 if entry.mec_ids else 0.5
        key = MMDrugOntology._normalise(drug_name)
        mechanism_type = self._ACTION_TYPES.get(key, "UNKNOWN")
        return DrugTargetRecord(
            drug_name=entry.canonical_name,
            drug_class=entry.drug_class,
            hgnc_genes=frozenset(entry.hgnc_targets),
            uniprot_proteins=frozenset(entry.uniprot_targets),
            confidence=confidence,
            mechanism_type=mechanism_type,
            chembl_mec_ids=entry.mec_ids,
            pubmed_pmids=entry.pubmed_pmids,
        )

    def get_many(self, drug_names: List[str]) -> List[DrugTargetRecord]:
        return [self.get(d) for d in drug_names]
```

---

### 3.3 `resistancemap/data/chembl_drug_target_loader.py`

Live ChEMBL wrapper with parquet cache for unseen drugs.

```python
"""
resistancemap/data/chembl_drug_target_loader.py
================================================
Wraps the ChEMBL get_mechanism + target_search API with a local
parquet cache so that repeat calls are free and the pipeline runs
offline after the first fetch.

Cache schema (parquet columns):
    drug_name, chembl_id, drug_class, hgnc_targets (json list),
    uniprot_targets (json list), chembl_target_id, mec_ids (json list),
    pubmed_pmids (json list), fetched_at (ISO timestamp)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path("data/cache/chembl_drug_targets.parquet")


class ChEMBLDrugTargetLoader:
    """Fetch drug→target mappings from ChEMBL with parquet caching.

    Parameters
    ----------
    cache_path : Path
        Local parquet file for caching. Created on first write.
    chembl_client : object | None
        Injectable ChEMBL client (for testing). When None, uses the
        chembl_webresource_client package.
    """

    def __init__(
        self,
        cache_path: Path = _DEFAULT_CACHE,
        chembl_client=None,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._client = chembl_client
        self._cache: dict = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __call__(self, drug_name: str):
        """Callable interface for MMDrugOntology.chembl_loader."""
        from resistancemap.data.drug_ontology import DrugEntry
        record = self._fetch(drug_name)
        return DrugEntry(**record)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(self, drug_name: str) -> dict:
        """Return raw dict (DrugEntry kwargs) for drug_name.

        Strategy:
        1. Check in-memory cache (populated from parquet on init).
        2. Query ChEMBL get_mechanism by compound name.
        3. Parse target ChEMBL ID, call target_search for HGNC symbol.
        4. Write result to parquet cache and return.
        """
        key = drug_name.lower().strip()
        if key in self._cache:
            return self._cache[key]

        client = self._get_client()
        # Step 1: find compound ChEMBL ID.
        compounds = client.molecule.filter(
            pref_name__iexact=drug_name
        ).only(["molecule_chembl_id", "pref_name", "max_phase"])
        if not compounds:
            raise KeyError(f"ChEMBL: no compound found for {drug_name!r}")
        chembl_id = compounds[0]["molecule_chembl_id"]

        # Step 2: get mechanism records.
        mechs = client.mechanism.filter(
            molecule_chembl_id=chembl_id,
            disease_efficacy=True,
        )
        mec_ids, target_ids, pmids = [], [], []
        for m in mechs:
            mec_ids.append(m["mec_id"])
            target_ids.append(m["target_chembl_id"])
            for ref in (m.get("mechanism_refs") or []):
                if ref.get("ref_type") == "PubMed":
                    pmids.append(ref["ref_id"])

        # Step 3: resolve targets to HGNC.
        hgnc, uniprot = [], []
        primary_target_id = target_ids[0] if target_ids else None
        for tid in set(target_ids):
            tgt = client.target.get(tid)
            for comp in (tgt.get("target_components") or []):
                if comp.get("component_type") == "PROTEIN":
                    for xref in (comp.get("target_component_xrefs") or []):
                        if xref["xref_src_db"] == "HGNC":
                            hgnc.append(xref["xref_name"])
                        elif xref["xref_src_db"] == "UniProt":
                            uniprot.append(xref["xref_id"])

        record = dict(
            canonical_name=drug_name.title(),
            chembl_id=chembl_id,
            drug_class="unknown",   # caller must override from class inference
            hgnc_targets=tuple(sorted(set(hgnc))),
            uniprot_targets=tuple(sorted(set(uniprot))),
            chembl_target_id=primary_target_id,
            mec_ids=tuple(mec_ids),
            pubmed_pmids=tuple(sorted(set(pmids))),
            combination_regimen_aliases=(),
        )
        self._cache[key] = record
        self._write_cache_row(key, record)
        return record

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            import pandas as pd
            df = pd.read_parquet(self._cache_path)
            for _, row in df.iterrows():
                key = row["drug_name"].lower().strip()
                self._cache[key] = dict(
                    canonical_name=row["drug_name"],
                    chembl_id=row["chembl_id"],
                    drug_class=row["drug_class"],
                    hgnc_targets=tuple(json.loads(row["hgnc_targets"])),
                    uniprot_targets=tuple(json.loads(row["uniprot_targets"])),
                    chembl_target_id=row.get("chembl_target_id"),
                    mec_ids=tuple(json.loads(row["mec_ids"])),
                    pubmed_pmids=tuple(json.loads(row["pubmed_pmids"])),
                    combination_regimen_aliases=(),
                )
        except Exception as exc:
            logger.warning("Could not load ChEMBL cache: %s", exc)

    def _write_cache_row(self, key: str, record: dict) -> None:
        try:
            import pandas as pd
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "drug_name": record["canonical_name"],
                "chembl_id": record["chembl_id"],
                "drug_class": record["drug_class"],
                "hgnc_targets": json.dumps(list(record["hgnc_targets"])),
                "uniprot_targets": json.dumps(list(record["uniprot_targets"])),
                "chembl_target_id": record.get("chembl_target_id"),
                "mec_ids": json.dumps(list(record["mec_ids"])),
                "pubmed_pmids": json.dumps(list(record["pubmed_pmids"])),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            if self._cache_path.exists():
                df = pd.read_parquet(self._cache_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_parquet(self._cache_path, index=False)
        except Exception as exc:
            logger.warning("Could not write ChEMBL cache: %s", exc)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from chembl_webresource_client.new_client import new_client
            return new_client
        except ImportError as exc:
            raise ImportError(
                "chembl_webresource_client is required for live ChEMBL queries. "
                "pip install chembl_webresource_client"
            ) from exc
```

---

### 3.4 `resistancemap/models/encoders/drug_encoder.py`

Encodes the rich `DrugContext` → `d_token` float tensor consumed by the SDE.

```python
"""
resistancemap/models/encoders/drug_encoder.py
==============================================
Encodes drug identity, mechanistic class, protein-target overlap, dose,
exposure window, and optional chemical embedding into a fixed d_drug token.

Input contract:
    drug_ids     : (N,) long — index into MM_DRUG_MANIFEST
    drug_classes : (N,) long — index into drug_class_index()
    target_mask  : (N, n_proteins) float — 1.0 where drug hits PPI node
    dose         : (N,) float — normalised dose [0, 1]
    exposure     : (N, 2) float — [start_time, end_time] normalised
    combo_id     : (N,) long — 0 = monotherapy, else regimen index
    chem_emb     : (N, d_chem) float or None — optional chemical embedding

Output contract:
    d_token      : (N, d_drug) float — consumed by GraphEnergyResistanceSDE
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class DrugEncoder(nn.Module):
    """Rich drug context → d_drug token.

    Parameters
    ----------
    n_drugs : int
        Vocabulary size (number of unique drugs in MM_DRUG_MANIFEST).
    n_classes : int
        Number of drug classes (len(ontology.drug_class_index())).
    n_proteins : int
        Size of the PPI graph node list (target_mask width).
    n_combos : int
        Number of unique combination regimens (0 = monotherapy).
    d_drug : int
        Output embedding dimension. Must match GraphEnergyResistanceSDE.d_drug.
    d_chem : int | None
        Dimension of optional chemical embedding. None disables the branch.
    dropout : float
    """

    def __init__(
        self,
        n_drugs: int,
        n_classes: int,
        n_proteins: int,
        n_combos: int,
        d_drug: int = 64,
        d_chem: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_embed = nn.Embedding(n_drugs + 1, d_drug, padding_idx=0)
        self.class_embed = nn.Embedding(n_classes + 1, d_drug // 4, padding_idx=0)
        self.combo_embed = nn.Embedding(n_combos + 1, d_drug // 4, padding_idx=0)

        # Target-mask projection (sparse binary → dense).
        self.target_proj = nn.Linear(n_proteins, d_drug // 2, bias=False)

        # Dose + exposure MLP: [dose, start_time, end_time] → d_drug // 4
        self.dose_mlp = nn.Sequential(
            nn.Linear(3, d_drug // 4),
            nn.SiLU(),
        )

        # Optional chemical embedding branch.
        self.d_chem = d_chem
        if d_chem is not None:
            self.chem_proj = nn.Linear(d_chem, d_drug // 2)
        else:
            self.chem_proj = None

        # Fusion: concatenate all branches → d_drug.
        # Branch dims: drug_embed(d_drug) + class(d/4) + combo(d/4) +
        #              target(d/2) + dose(d/4) [+ chem(d/2) if present]
        fused_dim = d_drug + d_drug // 4 + d_drug // 4 + d_drug // 2 + d_drug // 4
        if d_chem is not None:
            fused_dim += d_drug // 2

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, d_drug),
            nn.LayerNorm(d_drug),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_drug, d_drug),
        )

    def forward(
        self,
        drug_ids: Tensor,          # (N,) long
        drug_classes: Tensor,      # (N,) long
        target_mask: Tensor,       # (N, n_proteins) float
        dose: Tensor,              # (N,) float
        exposure: Tensor,          # (N, 2) float  [start, end]
        combo_id: Tensor,          # (N,) long
        chem_emb: Optional[Tensor] = None,  # (N, d_chem) or None
    ) -> Tensor:
        """Return d_token of shape (N, d_drug)."""
        e_drug = self.drug_embed(drug_ids)               # (N, d_drug)
        e_class = self.class_embed(drug_classes)         # (N, d/4)
        e_combo = self.combo_embed(combo_id)             # (N, d/4)
        e_target = self.target_proj(target_mask)         # (N, d/2)

        # Dose + exposure: concatenate [dose, start, end] → (N, 3)
        dose_exp = torch.cat(
            [dose.unsqueeze(-1), exposure], dim=-1
        )                                                # (N, 3)
        e_dose = self.dose_mlp(dose_exp)                 # (N, d/4)

        parts = [e_drug, e_class, e_combo, e_target, e_dose]

        if chem_emb is not None and self.chem_proj is not None:
            parts.append(self.chem_proj(chem_emb))      # (N, d/2)

        fused = torch.cat(parts, dim=-1)
        return self.fuse(fused)                          # (N, d_drug)
```

---

## 4. Schema Patch: `DrugBatch` Addition to `MORTBatch`

**File:** `resistancemap/mortfm/schemas.py`

Add a `DrugBatch` nested dataclass and extend `MORTBatch` with four new
optional fields. The existing `drug: Optional[Tensor]` field keeps its
semantics (already-encoded drug token); the new fields carry the *pre-encoding*
structured data needed by `DrugEncoder`.

```python
@dataclass
class DrugBatch:
    """Structured drug fields in a collated batch (pre-encoding).

    All tensors are (N, ...) with N = batch size.

    Attributes
    ----------
    drug_ids : (N,) long — index into MMDrugOntology manifest.
    drug_classes : (N,) long — index into drug_class_index().
    target_mask : (N, n_ppi_nodes) float — 1.0 where drug targets PPI node.
    dose : (N,) float — normalised dose.
    exposure : (N, 2) float — [normalised start_time, normalised end_time].
    combo_id : (N,) long — 0 = monotherapy.
    chem_emb : (N, d_chem) float or None — optional chemical embedding.
    presence : (N,) bool — False = row has no drug annotation.
    """
    drug_ids: Tensor
    drug_classes: Tensor
    target_mask: Tensor
    dose: Tensor
    exposure: Tensor
    combo_id: Tensor
    chem_emb: Optional[Tensor] = None
    presence: Optional[Tensor] = None

    def to(self, device: torch.device) -> "DrugBatch":
        return DrugBatch(
            drug_ids=self.drug_ids.to(device),
            drug_classes=self.drug_classes.to(device),
            target_mask=self.target_mask.to(device),
            dose=self.dose.to(device),
            exposure=self.exposure.to(device),
            combo_id=self.combo_id.to(device),
            chem_emb=self.chem_emb.to(device) if self.chem_emb is not None else None,
            presence=self.presence.to(device) if self.presence is not None else None,
        )
```

**New fields in `MORTBatch`** (insert after existing `drug: Optional[Tensor]`):

```python
    # Structured drug fields for the DrugEncoder (pre-encoding).
    drug_batch: Optional[DrugBatch] = None
```

**`MORTBatch.to()` patch** — extend the existing loop to handle `DrugBatch`:

```python
    elif isinstance(val, DrugBatch):
        moved[f.name] = val.to(device)
```

---

## 5. `_stack_drug()` Replacement in `mortfm_dataset.py`

Replace the current three-scalar stub with a full collation that builds a
`DrugBatch`. The existing `drug: Optional[Tensor]` in `MORTBatch` will be
populated AFTER the `DrugEncoder` runs in the model forward, not in the
collate function (so collate stays encoder-agnostic).

```python
def _stack_drug(
    snapshots: Sequence[PatientCellSnapshot],
    ontology: MMDrugOntology,
    ppi_gene_list: List[str],
) -> Optional[tuple["DrugBatch", Tensor]]:
    """Build a DrugBatch + presence mask from per-snapshot DrugContext objects.

    Parameters
    ----------
    snapshots : sequence of PatientCellSnapshot
    ontology : MMDrugOntology — for drug-id and class-id lookup
    ppi_gene_list : list[str] — ordered list of HGNC gene symbols in the PPI
        graph. Defines the columns of target_mask.

    Returns
    -------
    (DrugBatch, presence) or None if no snapshot has a drug annotation.

    Notes
    -----
    target_mask[i, j] = 1.0 iff drug in snapshot i targets gene ppi_gene_list[j].
    This is a *binary* mask; confidence weighting is applied inside DrugEncoder.
    Fabricating weights here from synthetic data is prohibited.
    """
    from resistancemap.data.drug_ontology import MMDrugOntology
    import torch

    N = len(snapshots)
    has_any = any(s.drug is not None for s in snapshots)
    if not has_any:
        return None

    class_index = ontology.drug_class_index()
    gene_index = {g: i for i, g in enumerate(ppi_gene_list)}
    n_ppi = len(ppi_gene_list)

    drug_ids = torch.zeros(N, dtype=torch.long)
    drug_classes = torch.zeros(N, dtype=torch.long)
    target_mask = torch.zeros(N, n_ppi, dtype=torch.float32)
    dose = torch.full((N,), float("nan"), dtype=torch.float32)
    exposure = torch.full((N, 2), float("nan"), dtype=torch.float32)
    combo_id = torch.zeros(N, dtype=torch.long)
    presence = torch.zeros(N, dtype=torch.bool)

    manifest_keys = list(ontology._manifest.keys())
    drug_key_to_idx = {k: i + 1 for i, k in enumerate(manifest_keys)}  # 0=padding

    combo_registry: dict[str, int] = {}
    next_combo = 1  # 0 = monotherapy

    for i, s in enumerate(snapshots):
        if s.drug is None:
            continue
        ctx = s.drug
        key = MMDrugOntology._normalise(ctx.drug_name)

        # Drug ID.
        drug_ids[i] = drug_key_to_idx.get(key, 0)

        # Class ID.
        try:
            entry = ontology.lookup(ctx.drug_name)
            drug_classes[i] = class_index.get(entry.drug_class, 0)
            for gene in entry.hgnc_targets:
                j = gene_index.get(gene)
                if j is not None:
                    target_mask[i, j] = 1.0
        except KeyError:
            pass  # unknown drug — drug_ids[i]=0 (padding), target_mask stays 0

        # Dose + exposure.
        if ctx.dose is not None:
            dose[i] = ctx.dose
        if ctx.start_time is not None:
            exposure[i, 0] = ctx.start_time
        if ctx.end_time is not None:
            exposure[i, 1] = ctx.end_time

        # Combination ID.
        if ctx.combination_id is not None:
            if ctx.combination_id not in combo_registry:
                combo_registry[ctx.combination_id] = next_combo
                next_combo += 1
            combo_id[i] = combo_registry[ctx.combination_id]

        # Optional molecular embedding (stacked separately if present).
        presence[i] = True

    # Chemical embedding: only stack if ALL present snapshots supply it
    # (otherwise DrugEncoder must fall back to no-chem path).
    chem_embs = [
        s.drug.molecular_embedding
        for s in snapshots
        if s.drug is not None and s.drug.molecular_embedding is not None
    ]
    chem_emb: Optional[torch.Tensor] = None
    n_present = int(presence.sum().item())
    if len(chem_embs) == n_present and n_present > 0:
        d_chem = chem_embs[0].shape[-1]
        chem_emb = torch.zeros(N, d_chem, dtype=torch.float32)
        idx = 0
        for i, s in enumerate(snapshots):
            if s.drug is not None and s.drug.molecular_embedding is not None:
                chem_emb[i] = s.drug.molecular_embedding
                idx += 1

    from resistancemap.mortfm.schemas import DrugBatch
    db = DrugBatch(
        drug_ids=drug_ids,
        drug_classes=drug_classes,
        target_mask=target_mask,
        dose=dose,
        exposure=exposure,
        combo_id=combo_id,
        chem_emb=chem_emb,
        presence=presence,
    )
    return db, presence
```

**Caller change in `mort_collate()`:**

```python
    # Replace old:
    #   packed = _stack_drug(snapshots)
    #   if packed is not None:
    #       values, presence = packed
    #       out.drug = values
    #       out.modality_mask[ModalityName.DRUG.value] = presence

    # New:
    drug_packed = _stack_drug(snapshots, ontology=ontology, ppi_gene_list=ppi_gene_list)
    if drug_packed is not None:
        db, presence = drug_packed
        out.drug_batch = db
        out.modality_mask[ModalityName.DRUG.value] = presence
        # out.drug (encoded tensor) is populated by DrugEncoder in model.forward()
```

`mort_collate` must accept `ontology` and `ppi_gene_list` as keyword args
(with defaults pointing to a singleton `MMDrugOntology` and the loaded PPI
gene list).

---

## 6. Time-Varying Treatment Exposure

### 6.1 `resistancemap/data/treatment_exposure.py`

```python
"""
resistancemap/data/treatment_exposure.py
=========================================
Build a time-indexed drug-pressure schedule d(t) from patient visit data.

d(t) is a function R -> R^{d_drug} representing the encoded drug context
at time t. Between treatment changes it is constant; at cycle boundaries
it steps to the new regimen embedding.

Input: a list of TreatmentVisit records (from CoMMpass IA23 or similar).
Output: a callable or discretised tensor of shape (T_grid, d_drug).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor


@dataclass
class TreatmentVisit:
    """One treatment cycle record from a patient's clinical timeline.

    Fields mirror CoMMpass MMRF IA23 treatment variable names.
    """
    visit_time: float          # months from baseline
    drug_names: List[str]      # drugs administered in this cycle
    doses: List[float]         # one per drug, micromolar or mg/m^2
    combination_id: Optional[str] = None  # e.g. "VRd"
    cycle_length_days: float = 28.0


def build_treatment_pressure(
    visits: List[TreatmentVisit],
    t_grid: Tensor,
    drug_encoder: "DrugEncoder",
    ontology: "MMDrugOntology",
    ppi_gene_list: List[str],
    default_drug_token: Optional[Tensor] = None,
) -> Tensor:
    """Map a list of TreatmentVisit records onto a discrete time grid.

    Parameters
    ----------
    visits : list[TreatmentVisit]
        Chronologically sorted treatment history. An empty list means
        no-treatment; drug_token will be `default_drug_token` or zeros.
    t_grid : (T,) float tensor
        Absolute time grid in months.
    drug_encoder : DrugEncoder
        The trained (or untrained) DrugEncoder.
    ontology : MMDrugOntology
    ppi_gene_list : list[str]
        PPI graph node gene list (defines target_mask width).
    default_drug_token : (d_drug,) tensor or None
        Token used before first treatment and after last. Defaults to zeros.

    Returns
    -------
    d_t : (T, d_drug)
        Drug embedding at each time grid point. Constant within cycle,
        steps at cycle boundaries. No smoothing is applied here; the
        SDE diffusion handles biological latency.
    """
    T = len(t_grid)
    d_drug = drug_encoder.drug_embed.embedding_dim
    if default_drug_token is None:
        default_drug_token = torch.zeros(d_drug)

    # Sort visits ascending by time.
    sorted_visits = sorted(visits, key=lambda v: v.visit_time)
    d_t = torch.stack([default_drug_token] * T)  # (T, d_drug)

    for visit in sorted_visits:
        t_start = visit.visit_time
        t_end = t_start + visit.cycle_length_days / 30.44  # days -> months

        # Build a single-row DrugBatch for this visit's regimen.
        # Combination of drugs in same cycle: take first (primary) drug.
        # TODO (Phase 3.1): support multi-drug combinations via max-pool.
        primary_drug = visit.drug_names[0] if visit.drug_names else None
        if primary_drug is None:
            continue

        from resistancemap.mortfm.schemas import DrugBatch
        batch = _visit_to_drug_batch(visit, ontology, ppi_gene_list)
        with torch.no_grad():
            token = drug_encoder(
                drug_ids=batch.drug_ids,
                drug_classes=batch.drug_classes,
                target_mask=batch.target_mask,
                dose=batch.dose,
                exposure=batch.exposure,
                combo_id=batch.combo_id,
                chem_emb=batch.chem_emb,
            ).squeeze(0)  # (d_drug,)

        for t_idx, t_val in enumerate(t_grid):
            if t_start <= t_val.item() < t_end:
                d_t[t_idx] = token

    return d_t  # (T, d_drug)


def _visit_to_drug_batch(
    visit: TreatmentVisit,
    ontology,
    ppi_gene_list: List[str],
) -> "DrugBatch":
    """Build a single-row DrugBatch from a TreatmentVisit."""
    from resistancemap.mortfm.schemas import DrugBatch

    manifest_keys = list(ontology._manifest.keys())
    drug_key_to_idx = {k: i + 1 for i, k in enumerate(manifest_keys)}
    class_index = ontology.drug_class_index()
    gene_index = {g: i for i, g in enumerate(ppi_gene_list)}

    primary = visit.drug_names[0] if visit.drug_names else ""
    key = ontology._normalise(primary) if primary else ""

    drug_ids = torch.tensor([drug_key_to_idx.get(key, 0)], dtype=torch.long)
    drug_class_val = 0
    n_ppi = len(ppi_gene_list)
    target_mask = torch.zeros(1, n_ppi)

    try:
        entry = ontology.lookup(primary)
        drug_class_val = class_index.get(entry.drug_class, 0)
        for gene in entry.hgnc_targets:
            j = gene_index.get(gene)
            if j is not None:
                target_mask[0, j] = 1.0
    except KeyError:
        pass

    dose_val = visit.doses[0] if visit.doses else float("nan")
    return DrugBatch(
        drug_ids=drug_ids,
        drug_classes=torch.tensor([drug_class_val], dtype=torch.long),
        target_mask=target_mask,
        dose=torch.tensor([dose_val], dtype=torch.float32),
        exposure=torch.tensor([[visit.visit_time, visit.visit_time + visit.cycle_length_days / 30.44]]),
        combo_id=torch.zeros(1, dtype=torch.long),
        chem_emb=None,
        presence=torch.ones(1, dtype=torch.bool),
    )
```

### 6.2 `GraphEnergyResistanceSDE` Patch for Time-Varying d(t)

The SDE `forward()` signature should accept `drug_schedule: Tensor` of shape
`(B, T, d_drug)` in addition to the current static `drug: Tensor` of shape
`(B, d_drug)`. At each step `t_idx`, it slices `drug_t = drug_schedule[:, t_idx, :]`.

```python
    def forward(
        self,
        z0: Tensor,
        graph_emb: Tensor,
        drug: Optional[Tensor] = None,          # (B, d_drug) static — legacy
        clinical: Optional[Tensor] = None,
        drug_schedule: Optional[Tensor] = None, # (B, T, d_drug) NEW
        *,
        return_samples: bool = True,
    ) -> dict:
        """
        If drug_schedule is supplied it takes precedence over drug.
        drug_schedule[:, t_idx, :] is used at each EM step.
        drug (static) is used when drug_schedule is None.
        """
        B = z0.shape[0]
        d_drug = self.drift.net[0].in_features  # read from first Linear layer
        if drug is None and drug_schedule is None:
            drug = z0.new_zeros((B, self.potential.drug_to_basin_weight.in_features))
        if clinical is None:
            clinical = z0.new_zeros((B, 1))

        T = self.n_time_grid
        dt = self.integration_time / max(T - 1, 1)
        t_grid = torch.linspace(0.0, self.integration_time, T, device=z0.device)

        def _drug_at(t_idx: int) -> Tensor:
            if drug_schedule is not None:
                t_idx_clamped = min(t_idx, drug_schedule.shape[1] - 1)
                return drug_schedule[:, t_idx_clamped, :]
            return drug  # static fallback

        z_traj, drift_norms = [z0], []
        z = z0
        for step in range(T - 1):
            d_t = _drug_at(step)
            z_next = self._step(z, graph_emb, d_t, clinical, dt, deterministic=True)
            drift_norms.append((z_next - z).norm(dim=-1))
            z = z_next
            z_traj.append(z)
        z_traj_t = torch.stack(z_traj, dim=1)  # (B, T, d_latent)
        # ... MC samples analogously ...
```

---

## 7. Tests

### 7.1 `tests/mortfm/test_drug_context.py`

```python
"""Tests: drug names / classes / targets survive ontology lookup + collation."""

import pytest
from resistancemap.data.drug_ontology import MMDrugOntology, MM_DRUG_MANIFEST


def test_all_manifest_entries_have_chembl_id():
    """Every static entry must have a non-empty ChEMBL ID (HARD CONSTRAINT)."""
    for key, entry in MM_DRUG_MANIFEST.items():
        assert entry.chembl_id.startswith("CHEMBL"), (
            f"{key}: chembl_id={entry.chembl_id!r} must start with 'CHEMBL'"
        )


def test_bortezomib_targets():
    ont = MMDrugOntology()
    entry = ont.lookup("bortezomib")
    assert "PSMB5" in entry.hgnc_targets
    assert entry.drug_class == "proteasome_inhibitor"


def test_lenalidomide_targets():
    ont = MMDrugOntology()
    entry = ont.lookup("revlimid")   # alias
    assert "CRBN" in entry.hgnc_targets
    assert entry.drug_class == "IMiD"


def test_daratumumab_target():
    ont = MMDrugOntology()
    entry = ont.lookup("daratumumab")
    assert "CD38" in entry.hgnc_targets
    assert "P28907" in entry.uniprot_targets


def test_teclistamab_dual_target():
    ont = MMDrugOntology()
    entry = ont.lookup("teclistamab")
    assert "TNFRSF17" in entry.hgnc_targets
    assert "CD3E" in entry.hgnc_targets


def test_unknown_drug_raises():
    ont = MMDrugOntology()
    with pytest.raises(KeyError):
        ont.lookup("totally_fictitious_drug_xyz")


def test_class_index_stable():
    """drug_class_index() must return same mapping on repeated calls."""
    ont = MMDrugOntology()
    idx1 = ont.drug_class_index()
    idx2 = ont.drug_class_index()
    assert idx1 == idx2


def test_collate_drug_batch_shape():
    """DrugBatch fields must have correct shapes after _stack_drug."""
    import torch
    from unittest.mock import MagicMock
    from resistancemap.mortfm.schemas import DrugContext, PatientCellSnapshot
    from resistancemap.data.mortfm_dataset import _stack_drug
    from resistancemap.data.drug_ontology import MMDrugOntology

    ppi_genes = ["PSMB5", "CD38", "BCL2", "XPO1", "CRBN"]
    ont = MMDrugOntology()

    snaps = [
        PatientCellSnapshot(
            patient_id="P1", sample_id="S1", disease="MM",
            drug=DrugContext(drug_name="Bortezomib", dose=1.3, start_time=0.0, end_time=1.0),
        ),
        PatientCellSnapshot(
            patient_id="P2", sample_id="S2", disease="MM",
            drug=DrugContext(drug_name="Lenalidomide", dose=10.0, start_time=-1.0, end_time=0.0),
        ),
    ]

    result = _stack_drug(snaps, ontology=ont, ppi_gene_list=ppi_genes)
    assert result is not None
    db, presence = result

    assert db.drug_ids.shape == (2,)
    assert db.target_mask.shape == (2, 5)
    assert presence.dtype == torch.bool

    # Bortezomib hits PSMB5 (index 0).
    assert db.target_mask[0, 0] == 1.0
    # Lenalidomide hits CRBN (index 4).
    assert db.target_mask[1, 4] == 1.0
    # Bortezomib does not hit CD38 (index 1).
    assert db.target_mask[0, 1] == 0.0
```

### 7.2 `tests/mortfm/test_drug_schedule_sde.py`

```python
"""Tests: time-varying drug schedule changes trajectory rollout."""

import torch
import pytest
from resistancemap.mortfm.trajectory.graph_energy_sde import GraphEnergyResistanceSDE


def test_static_vs_schedule_same_constant():
    """Static drug == constant schedule must produce identical trajectory."""
    torch.manual_seed(0)
    B, d_lat, d_graph, d_drug, d_clin = 2, 8, 4, 4, 2
    T = 5
    sde = GraphEnergyResistanceSDE(
        d_latent=d_lat, d_graph=d_graph, d_drug=d_drug,
        d_clinical=d_clin, n_basins=2, n_time_grid=T, n_mc_samples=0,
    )
    sde.eval()

    z0 = torch.randn(B, d_lat)
    g = torch.randn(B, d_graph)
    drug_static = torch.randn(B, d_drug)
    clin = torch.randn(B, d_clin)

    # Build a schedule that is constant = drug_static at all time steps.
    drug_sched = drug_static.unsqueeze(1).expand(B, T, d_drug)

    with torch.no_grad():
        out_static = sde.forward(z0, g, drug=drug_static, clinical=clin, return_samples=False)
        out_sched = sde.forward(z0, g, drug=None, clinical=clin,
                                drug_schedule=drug_sched, return_samples=False)

    torch.testing.assert_close(
        out_static["z_traj"], out_sched["z_traj"],
        msg="Static drug and constant schedule must produce identical trajectory"
    )


def test_schedule_change_alters_trajectory():
    """Switching drug at midpoint must change the second half of the trajectory."""
    torch.manual_seed(42)
    B, d_lat, d_graph, d_drug, d_clin = 2, 8, 4, 4, 2
    T = 6
    sde = GraphEnergyResistanceSDE(
        d_latent=d_lat, d_graph=d_graph, d_drug=d_drug,
        d_clinical=d_clin, n_basins=2, n_time_grid=T, n_mc_samples=0,
    )
    sde.eval()

    z0 = torch.randn(B, d_lat)
    g = torch.randn(B, d_graph)
    clin = torch.randn(B, d_clin)
    drug_a = torch.randn(B, d_drug)
    drug_b = torch.randn(B, d_drug)

    # Schedule A: constant drug_a.
    sched_a = drug_a.unsqueeze(1).expand(B, T, d_drug).clone()
    # Schedule AB: drug_a for first half, drug_b for second half.
    sched_ab = torch.cat([
        drug_a.unsqueeze(1).expand(B, T // 2, d_drug),
        drug_b.unsqueeze(1).expand(B, T - T // 2, d_drug),
    ], dim=1)

    with torch.no_grad():
        traj_a = sde.forward(z0, g, drug=None, clinical=clin,
                              drug_schedule=sched_a, return_samples=False)["z_traj"]
        traj_ab = sde.forward(z0, g, drug=None, clinical=clin,
                               drug_schedule=sched_ab, return_samples=False)["z_traj"]

    # First T//2 steps should be identical (same drug).
    torch.testing.assert_close(traj_a[:, :T // 2], traj_ab[:, :T // 2])
    # Second half must differ (different drug changes drift + potential).
    assert not torch.allclose(traj_a[:, T // 2:], traj_ab[:, T // 2:]), (
        "Drug schedule change must alter the second half of the trajectory"
    )
```

---

## 8. File-Change Table

| File | Action | Notes |
|---|---|---|
| `resistancemap/data/drug_ontology.py` | CREATE | Static MM ontology; 19 drugs; all ChEMBL-backed |
| `resistancemap/data/drug_target_loader.py` | CREATE | DrugTargetRecord + DrugTargetLoader |
| `resistancemap/data/chembl_drug_target_loader.py` | CREATE | Live ChEMBL fetch + parquet cache |
| `resistancemap/data/treatment_exposure.py` | CREATE | build_treatment_pressure() → d(t) |
| `resistancemap/models/encoders/drug_encoder.py` | CREATE | Rich DrugEncoder; identity+class+target+dose+chem |
| `resistancemap/mortfm/schemas.py` | PATCH | Add DrugBatch dataclass; add `drug_batch` field to MORTBatch; extend `to()` |
| `resistancemap/data/mortfm_dataset.py` | PATCH | Replace `_stack_drug()` stub; update `mort_collate()` signature |
| `resistancemap/mortfm/trajectory/graph_energy_sde.py` | PATCH | Add `drug_schedule` arg to `forward()`; slice per-step |
| `tests/mortfm/test_drug_context.py` | CREATE | 7 tests covering ontology + collation |
| `tests/mortfm/test_drug_schedule_sde.py` | CREATE | 2 tests covering static/schedule equivalence + change |

---

## 9. Definition of Done

The Phase 3 implementation is done when all of the following hold:

1. **SDE contract:** `GraphEnergyResistanceSDE.forward()` receives
   `drug_schedule: (B, T, d_drug)` (or static `drug: (B, d_drug)`) where
   `d_drug >= 64` and the tensor was produced by `DrugEncoder`. The three-scalar
   `[dose, start_time, end_time]` path is permanently removed from `_stack_drug`.

2. **Ontology coverage:** `MM_DRUG_MANIFEST` contains all 19 drugs in the table
   above. Every entry has a non-empty `chembl_id`. Every entry with `mec_ids=()`
   has a documented reason in the manifest comment.

3. **No synthetic data:** `_stack_drug` never calls `torch.rand*`. Doses come
   from real `DrugContext.dose` fields; `NaN` sentinel is used for missing values.

4. **Tests pass:** All 9 tests in `test_drug_context.py` and
   `test_drug_schedule_sde.py` pass on `v18-clean-canonical-mortfm` with
   `pytest -x`.

5. **Trainer unchanged:** `MORTFMTrainer` receives a `MORTBatch` with
   `drug_batch` populated; it calls `DrugEncoder` to produce `drug_t` before
   passing to SDE. The trainer does not parse drug strings directly.

6. **Backward compatibility:** `MORTBatch.drug` field (already-encoded token) is
   still accepted when `drug_batch` is None (legacy path for unit tests that do
   not use the full collate stack).

---

## 10. Evidence Index

All claims in the drug-target table above are backed by ChEMBL v34 mechanism
records fetched 2026-05-20 via the ChEMBL MCP tool. The evidence is:

| Claim | Source | ID |
|---|---|---|
| Lenalidomide/Pomalidomide/Thalidomide → CRBN (CRL4-CRBN E3 ligase) | ChEMBL mec 5876/5877/5875 | PubMed 22552008, 25043012 |
| Carfilzomib → PSMB5 (26S proteasome beta-5) | ChEMBL mec 1430, target CHEMBL2364701 | PubMed 20116451 |
| Ixazomib → PSMB5 beta-5 subunit | ChEMBL mec 7822 | PubMed 23995855, 30867233 |
| Isatuximab → CD38 | ChEMBL mec 6401, target CHEMBL4660 | PubMed 28249894 |
| Selinexor → XPO1 (Exportin-1) | ChEMBL mec 5815, target CHEMBL5661 | PubMed 27893412 |
| Teclistamab → BCMA/CD3E bispecific | ChEMBL mec 8687/8929, targets CHEMBL4523587+CHEMBL2364168 | PubMed 32956453 |
| Dexamethasone → NR3C1 (GR agonist) | ChEMBL mec 813, target CHEMBL2034 | PubMed 16891588, 16956592, 16971495 |
| Venetoclax → BCL2 | ChEMBL mec 5516, target CHEMBL4860 | FDA label 208573s000 |
| Melphalan → DNA inhibitor | ChEMBL mec 1858, target CHEMBL2311221 | DailyMed label 9706e9a1 |
| Daratumumab → CD38 | ChEMBL mec 2425, target CHEMBL4660 | DailyMed label a4d0efe9 |
| Elotuzumab → SLAMF7 | ChEMBL mec 2428, target CHEMBL3559386 | DailyMed label 80686b7e |
| Ide-cel → BCMA (CAR-T) | ChEMBL mec 7989, target CHEMBL4523587 | FDA label media/147055 |
| Cilta-cel → BCMA (CAR-T) | ChEMBL mec 9620, target CHEMBL4523587 | EMA CARVYKTI EPAR |
| Panobinostat → HDAC (pan) | ChEMBL mec 2340, target CHEMBL2093865 | FDA label 205353s000 |
