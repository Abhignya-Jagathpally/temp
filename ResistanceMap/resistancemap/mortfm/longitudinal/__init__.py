"""
resistancemap.mortfm.longitudinal
=================================

Schemas + builders for true longitudinal MORT-FM training:

  * :class:`PatientTimeline` — full per-patient history (samples, treatments,
    response/resistance events).
  * :class:`ResistanceEndpoint` — typed resistance event with explicit
    evidence level (clinical, molecular, ex_vivo, weak).
  * :class:`LongitudinalPair` — same-patient baseline → followup pair with
    treatment exposure between timepoints.
  * :func:`build_pairs_from_mmrf` — concrete builder that emits
    :class:`LongitudinalPair` records from the MMRF processed tables we
    already have on disk (787 baseline + 29 paired + 994 outcomes).

This is the substrate for the longitudinal_trajectory and
resistance_emergence claim gates.
"""

from resistancemap.mortfm.longitudinal.schemas import (  # noqa: F401
    ClinicalEvent,
    LongitudinalPair,
    MolecularSample,
    PatientTimeline,
    ResistanceEndpoint,
    ResponseEvent,
    TreatmentExposure,
)
from resistancemap.mortfm.longitudinal.resistance_endpoint_builder import (  # noqa: F401
    build_endpoints_from_mmrf_outcomes,
)
from resistancemap.mortfm.longitudinal.temporal_pair_builder import (  # noqa: F401
    build_pairs_from_mmrf,
)
from resistancemap.mortfm.longitudinal.longitudinal_dataset import (  # noqa: F401
    LongitudinalDataset,
)
