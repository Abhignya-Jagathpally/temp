"""
resistancemap/mortfm/longitudinal/cohort_registry.py
=====================================================
Typed catalogue of longitudinal cohorts MORT-FM can pull from. Each
record describes WHAT a cohort contributes (paired molecular pairs,
event times, treatment exposure) so the gate auditor can decide which
gates a particular cohort actually advances.

This is the typed mirror of ``docs/LONGITUDINAL_DATA_INVENTORY.md``;
the registry decides claim semantics, the doc describes data provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


AccessTier = Literal["public", "registered", "controlled_dbgap", "controlled_ega"]
Modality = Literal["bulk_rna", "scrna", "bulk_proteomics", "wes", "wgs", "methylation"]


@dataclass
class LongitudinalCohortSpec:
    name: str
    disease: str
    molecular_modality: Modality
    baseline_available: bool
    relapse_available: bool
    event_time_available: bool
    treatment_available: bool
    public_or_controlled: AccessTier
    expected_pairs: int
    feature_id_type: str            # ENSG, HGNC, UniProt, ...
    on_disk_path: Optional[str] = None
    ingested_at: Optional[str] = None
    notes: str = ""

    @property
    def contributes_to(self) -> List[str]:
        """Which claim levels this cohort can plausibly advance."""
        levels: List[str] = []
        if self.baseline_available:
            levels.append("static_drug_response" if self.treatment_available else "single_cell_state")
        if self.baseline_available and self.relapse_available:
            levels.append("longitudinal_trajectory")
        if self.event_time_available:
            levels.append("survival_prediction")
            if self.treatment_available:
                levels.append("resistance_emergence")
        return sorted(set(levels))


# Authoritative cohort catalogue. Synchronised with
# docs/LONGITUDINAL_DATA_INVENTORY.md.
CANONICAL_COHORTS: Dict[str, LongitudinalCohortSpec] = {
    "MMRF_IA12_paired": LongitudinalCohortSpec(
        name="MMRF_IA12_paired",
        disease="MM",
        molecular_modality="bulk_rna",
        baseline_available=True,
        relapse_available=True,
        event_time_available=True,
        treatment_available=True,
        public_or_controlled="controlled_dbgap",
        expected_pairs=29,
        feature_id_type="ENSG",
        on_disk_path="data/processed/mmrf_paired_z64.npz",
        ingested_at="2026-05-17",
        notes="29 patients with t0+t1 RNA-seq; 20 observed TT2L events. "
              "Real backbone of v15+v16 MORT-FM longitudinal substrate.",
    ),
    "PADIMAC_GSE116324": LongitudinalCohortSpec(
        name="PADIMAC_GSE116324",
        disease="MM",
        molecular_modality="bulk_rna",
        baseline_available=True,
        relapse_available=False,
        event_time_available=False,
        treatment_available=True,
        public_or_controlled="public",
        expected_pairs=0,
        feature_id_type="ENSG",
        on_disk_path="data/processed/padimac/padimac_expression.parquet",
        ingested_at="2026-05-18",
        notes="44 newly diagnosed MM patients, baseline-only RNA-seq under "
              "PAD therapy. Static_drug_response augmentation only — NOT "
              "longitudinal evidence.",
    ),
    "BeatAML_1.0": LongitudinalCohortSpec(
        name="BeatAML_1.0",
        disease="AML",
        molecular_modality="bulk_rna",
        baseline_available=True,
        relapse_available=False,
        event_time_available=False,
        treatment_available=True,
        public_or_controlled="public",
        expected_pairs=0,
        feature_id_type="HGNC",
        on_disk_path="data/processed/beataml/beataml_expression.parquet",
        ingested_at="2026-05-17",
        notes="451 specimens / 303 patients with ex-vivo AUC drug response. "
              "Used for Block B hematologic-specimen-drug-response track.",
    ),
    "GSE124310_GSE271107_scRNA": LongitudinalCohortSpec(
        name="GSE124310_GSE271107_scRNA",
        disease="MM",
        molecular_modality="scrna",
        baseline_available=True,
        relapse_available=False,
        event_time_available=False,
        treatment_available=False,
        public_or_controlled="public",
        expected_pairs=0,
        feature_id_type="HGNC",
        on_disk_path="data/processed/single_cell/scrna_manifest.csv",
        ingested_at="2026-05-17",
        notes="151,997 cells / 51 pseudobulks across HD/MGUS/SMM/MM stages. "
              "Pseudotime-ordinal; single_cell_state evidence ONLY.",
    ),
    # Cohorts catalogued but NOT yet ingested. The gate auditor uses these
    # entries to compute the n-pairs ceiling assuming a successful ingest.
    "MMRF_IA19_planned": LongitudinalCohortSpec(
        name="MMRF_IA19_planned",
        disease="MM",
        molecular_modality="bulk_rna",
        baseline_available=True,
        relapse_available=True,
        event_time_available=True,
        treatment_available=True,
        public_or_controlled="controlled_dbgap",
        expected_pairs=80,
        feature_id_type="ENSG",
        notes="Next MMRF interim release; advertised 60-120 additional paired patients. dbGaP IRB required.",
    ),
    "GSE39754_Brioli_paired": LongitudinalCohortSpec(
        name="GSE39754_Brioli_paired",
        disease="MM",
        molecular_modality="bulk_rna",
        baseline_available=True,
        relapse_available=True,
        event_time_available=False,
        treatment_available=True,
        public_or_controlled="public",
        expected_pairs=25,
        feature_id_type="ENSG",
        notes="Real public-GEO paired diagnosis/relapse MM RNA-seq.",
    ),
    "Tirier2021_EGA_scRNA_paired": LongitudinalCohortSpec(
        name="Tirier2021_EGA_scRNA_paired",
        disease="MM",
        molecular_modality="scrna",
        baseline_available=True,
        relapse_available=True,
        event_time_available=False,
        treatment_available=True,
        public_or_controlled="controlled_ega",
        expected_pairs=14,
        feature_id_type="HGNC",
        notes="14 MM patients with baseline + relapse scRNA-seq; EGA controlled access.",
    ),
}


def lookup_cohort(name: str) -> Optional[LongitudinalCohortSpec]:
    return CANONICAL_COHORTS.get(name)


def write_cohort_registry(
    out_path: str = "logs/mortfm/longitudinal_cohort_registry.json",
) -> str:
    payload = {n: {**asdict(c), "contributes_to": c.contributes_to}
               for n, c in CANONICAL_COHORTS.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return str(out)


def total_pairs_in_state(state: Literal["ingested", "ingested_plus_planned"] = "ingested") -> int:
    """How many same-patient molecular pairs do we *currently* have, vs available."""
    if state == "ingested":
        return sum(c.expected_pairs for c in CANONICAL_COHORTS.values()
                   if c.on_disk_path is not None
                   and c.baseline_available and c.relapse_available)
    return sum(c.expected_pairs for c in CANONICAL_COHORTS.values()
               if c.baseline_available and c.relapse_available)
