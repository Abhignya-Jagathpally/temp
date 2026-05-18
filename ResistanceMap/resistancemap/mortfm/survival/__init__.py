"""
resistancemap.mortfm.survival
==============================

Survival / time-to-event modelling for MORT-FM-LENS.

  * :class:`CompetingRiskHead` — discrete-time hazard head with one logit
    per (timestep, event-type). Supports right-censoring and an arbitrary
    number of competing risks.
  * :func:`concordance_index` — Harrell C-index on right-censored time-to-
    event data; pure stdlib, no lifelines dependency.
  * :func:`integrated_brier_score` — IBS computed over the discrete time
    grid the hazard head emits.
  * :class:`SurvivalCalibration` — temperature scaling on the predicted
    survival curve; refit on a held-out fold.

This package supplies the metrics + head that gate_resistance_emergence
and gate_survival_prediction read.
"""

from resistancemap.mortfm.survival.competing_risk_head import (  # noqa: F401
    CompetingRiskHead,
    discrete_time_grid,
    survival_curve_from_hazards,
)
from resistancemap.mortfm.survival.survival_calibration import (  # noqa: F401
    SurvivalCalibration,
)
from resistancemap.mortfm.survival.time_to_event_metrics import (  # noqa: F401
    concordance_index,
    concordance_index_bootstrap_ci,
    integrated_brier_score,
    km_strata_separation,
)
from resistancemap.mortfm.survival.baselines import (  # noqa: F401
    ClinicalOnlyCox,
    cox_fit_predict_loo,
    permutation_null_cindex,
)
from resistancemap.mortfm.survival.regularized_lens import (  # noqa: F401
    RegularizedLENS,
    RegularizedLENSConfig,
    bootstrap_optimism_correction,
    fit_predict_loo_regularized,
)
