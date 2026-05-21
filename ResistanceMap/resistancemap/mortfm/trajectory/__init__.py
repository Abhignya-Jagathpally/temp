"""
resistancemap.mortfm.trajectory
================================

MORT-FM-LENS trajectory primitives. Together these implement the
graph-conditioned, drug-pressured neural SDE described in the v15 plan
section 5:

    dz_t = [-grad U_theta(z_t, d) + f_theta(z_t, g_t, d, c)] dt
           + sigma_theta(z_t, d) dW_t

  * :class:`WaddingtonPotential`  -> scalar U_theta(z, d); -grad U is the
    attractive drift toward learned cell-state basins.
  * :class:`GraphConditionedDrift` -> f_theta(z, g, d, c); receives a
    pre-pooled graph embedding so the SDE conditions on biological context.
  * :class:`GraphEnergyResistanceSDE` -> the composed module that rolls out
    z_t over a fixed time grid via Euler-Maruyama. Returns the trajectory
    and Monte-Carlo samples for downstream basin/hitting-time scoring.
  * :class:`ResistanceBasin`  -> learned basin centroids + assignment
    probabilities. K basins (sensitive, persister, MRD-like, relapse-like,
    drug-specific resistant, ...).
  * :class:`HittingTime`     -> estimator for tau^R = inf{t : z_t in B_R}
    using the SDE samples; reports P(tau^R <= T).

Honest behaviour
----------------
* All modules are *standard nn.Module*; no opaque training-time-only
  randomness. The Euler-Maruyama Brownian noise is the ONLY stochastic
  element and is drawn from torch.randn at forward time. The
  fabrication-sentinel allows torch.randn inside model code because
  modelling stochastic dynamics is not the same as fabricating data.
* The SDE never invents target labels. If the trainer cannot supply
  longitudinal supervision the loss router (see resistancemap.training)
  refuses to add the trajectory loss; it never silently substitutes.
"""

from resistancemap.mortfm.trajectory.graph_energy_sde import (  # noqa: F401
    GraphConditionedDrift,
    GraphEnergyResistanceSDE,
    WaddingtonPotential,
)
from resistancemap.mortfm.trajectory.graph_projector import (  # noqa: F401
    LatentToGraphProjector,
)
from resistancemap.mortfm.trajectory.grid import (  # noqa: F401
    CanonicalTimeGridConfig,
    canonical_time_grid,
    time_grid_meta,
)
from resistancemap.mortfm.trajectory.hitting_time import (  # noqa: F401
    HittingTime,
)
from resistancemap.mortfm.trajectory.resistance_basin import (  # noqa: F401
    ResistanceBasin,
)
