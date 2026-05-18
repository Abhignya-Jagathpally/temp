"""Continuous-time dynamics for MORT-FM.

Modules
-------
* :mod:`neural_ode` — deterministic ``dz/dt = f(z, t, d, G)`` via torchdiffeq
* :mod:`neural_sde` — stochastic ``dz = -grad U dt + g dt + sigma dW``
* :mod:`jump_sde`   — adds jump diffusion for clonal/therapy-switch events
* :mod:`graph_conditioned_drift` — drift term that consumes a PPI graph
* :mod:`treatment_conditioned_dynamics` — drift term that consumes a drug token
* :mod:`trajectory_sampler` — wrapper exposing sample_paths(z0, ...) API
* :mod:`temporal_decoder` — decodes z(t) back to an observable cell-state

All modules fall back to a torch-only Euler integrator if ``torchdiffeq`` /
``torchsde`` is not installed, so smoke-tests work in a minimal environment.
"""

from resistancemap.legacy.v15_planned_mortfm.models.dynamics.neural_ode import GraphConditionedNeuralODE
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.neural_sde import GraphConditionedNeuralSDE
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.jump_sde import NeuralJumpSDE
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.graph_conditioned_drift import GraphConditionedDrift
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.treatment_conditioned_dynamics import (
    TreatmentConditionedDynamics,
)
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.trajectory_sampler import TrajectorySampler
from resistancemap.legacy.v15_planned_mortfm.models.dynamics.temporal_decoder import TemporalDecoder

__all__ = [
    "GraphConditionedNeuralODE",
    "GraphConditionedNeuralSDE",
    "NeuralJumpSDE",
    "GraphConditionedDrift",
    "TreatmentConditionedDynamics",
    "TrajectorySampler",
    "TemporalDecoder",
]
