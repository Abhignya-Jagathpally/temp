"""
resistancemap.mortfm
====================
Multi-Omic Resistance Trajectory Foundation Model (MORT-FM).

MORT-FM is the v15 research-track namespace inside ResistanceMap. It is a
*separate* model family from the v14 static cell-line ResistanceMap pipeline;
the two share a package root but should not share trained checkpoints,
configs, or claims.

Scientific scope (what MORT-FM is intended to predict)
------------------------------------------------------
Given a patient/cell molecular snapshot at baseline (scRNA + scATAC/methylation/
histone PTMs + proteomics/phosphoproteomics + drug exposure + clinical
covariates) and a biological graph (PPI + GRN + drug-target + pathway
membership), predict:

  * future resistance state (which attractor basin)
  * time-to-resistance (survival)
  * trajectory distribution (Wasserstein-matched future state)
  * pathway route (which proteins/edges mediate the transition)
  * drug-specific risk
  * uncertainty (posterior over paths — one genotype -> many phenotypes)
  * counterfactual intervention (which node/drug perturbation delays it)

Architectural primitives
------------------------
1. ``schemas.PatientCellSnapshot``     — typed molecular state at a timepoint
2. ``schemas.ResistanceOutcome``       — censored survival + future state labels
3. ``schemas.TemporalTrainingPair``    — (X_t, X_{t+delta}, outcome) supervision
4. ``schemas.MORTBatch``               — canonical batch fed to the model
5. ``schemas.DrugContext``             — drug identity + class + targets + dose
6. ``schemas.MORTFMConfig``            — config dataclass mirroring configs/mortfm.yaml

Honest defaults
---------------
* Loaders raise ``FileNotFoundError`` (or return empty objects) when upstream
  raw data is absent. They never fabricate.
* Dataclasses default the labelled fields to ``None`` so the model can run on
  unlabelled patients (for pre-training) but the training loop explicitly
  rejects ``MORTBatch`` objects missing the labels required for the loss head
  being optimised.
* No claim that MORT-FM has been trained on patient outcomes is permitted
  until ``resistancemap.mortfm.trainer`` has produced a checkpoint that was
  fit on a real :class:`ResistanceOutcome` cohort with N >= the validated
  identifiability threshold (see ``docs/MORTFM_LIMITATIONS.md``).
"""

from __future__ import annotations

from resistancemap.mortfm.schemas import (
    DrugContext,
    FoundationState,
    MORTBatch,
    MORTFMConfig,
    ModalityName,
    ModalityTensor,
    PatientCellSnapshot,
    ResistanceOutcome,
    TemporalTrainingPair,
    TrajectoryPrediction,
)

__all__ = [
    "DrugContext",
    "FoundationState",
    "MORTBatch",
    "MORTFMConfig",
    "ModalityName",
    "ModalityTensor",
    "PatientCellSnapshot",
    "ResistanceOutcome",
    "TemporalTrainingPair",
    "TrajectoryPrediction",
]
