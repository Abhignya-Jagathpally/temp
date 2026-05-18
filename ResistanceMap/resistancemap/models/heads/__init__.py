"""MORT-FM prediction heads.

Each head consumes the foundation latent ``z0`` or a trajectory rollout
``z_path`` and produces one of the model's outputs (resistance state,
time-to-resistance, future trajectory, pathway route, drug-specific risk,
uncertainty, counterfactual interventions).
"""

from resistancemap.models.heads.resistance_state_head import ResistanceStateHead
from resistancemap.models.heads.time_to_resistance_head import (
    CoxSurvivalHead,
    DiscreteTimeSurvivalHead,
    TimeToResistanceHead,
)
from resistancemap.models.heads.trajectory_head import TrajectoryDistributionHead
from resistancemap.models.heads.pathway_route_head import PathwayRouteHead
from resistancemap.models.heads.drug_risk_head import DrugSpecificTrajectoryRiskHead
from resistancemap.models.heads.uncertainty_head import EvidentialUncertaintyHead
from resistancemap.models.heads.counterfactual_head import CounterfactualInterventionHead

__all__ = [
    "ResistanceStateHead",
    "CoxSurvivalHead",
    "DiscreteTimeSurvivalHead",
    "TimeToResistanceHead",
    "TrajectoryDistributionHead",
    "PathwayRouteHead",
    "DrugSpecificTrajectoryRiskHead",
    "EvidentialUncertaintyHead",
    "CounterfactualInterventionHead",
]
