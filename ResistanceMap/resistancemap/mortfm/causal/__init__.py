"""
resistancemap.mortfm.causal
============================

Causal counterfactual primitives for MORT-FM-LENS.

The causal pathway-mechanism claim (the highest bar in the claim-gate
registry) requires evidence at three layers:

  1. :class:`InterventionGraph` — perform ``do(e=0)`` or ``do(e=1)`` on a
     biological-graph edge or node. The intervention zeroes or amplifies
     the contribution that node/edge makes to the graph embedding the SDE
     consumes.
  2. :class:`CounterfactualRunner` — for each intervention, re-run the
     SDE + hitting time and compute Δ_e = E[P(τ^R ≤ T) | do(e=1)]
     − E[P(τ^R ≤ T) | do(e=0)]. The runner returns a tidy table of
     edge/node-level effects.
  3. :class:`PathwayCausalValidator` — cross-checks Δ_e against external
     evidence (CRISPR DepMap gene-effect, drug-target Open Targets, known
     Reactome pathway). Without external agreement the causal_mechanism
     claim is refused.

Honest constraints
------------------
* The InterventionGraph cannot intervene on edges that don't exist in
  the graph. ``intervene(...)`` raises rather than silently returning
  identity. The runner will record an explicit "missing_edge" reason in
  the output table.
* The validator never grants causal_mechanism on agreement statistics
  alone — it requires *negative controls* (random edges should NOT score
  high) and reports the negative-control distribution.
"""

from resistancemap.mortfm.causal.intervention_graph import (  # noqa: F401
    InterventionGraph,
    InterventionType,
)
from resistancemap.mortfm.causal.counterfactual_runner import (  # noqa: F401
    CounterfactualRunner,
    EdgeEffect,
)
from resistancemap.mortfm.causal.pathway_causal_validator import (  # noqa: F401
    PathwayCausalValidator,
)
