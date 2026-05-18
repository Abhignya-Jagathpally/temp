"""
resistancemap.mortfm.registries
================================

First-class registries that track *what* MORT-FM contains:

* :mod:`feature_space_registry` — every distinct feature axis (RNA top-2000,
  scRNA HVG-union, protein/drug/pathway/graph node spaces).
* :mod:`endpoint_registry` — every clinical or molecular endpoint, with
  allowed and forbidden claim levels per type. Prevents OS from being reused
  as a resistance endpoint.
* :mod:`claim_gate_registry` — twelve claim levels (technical, static_drug,
  pathway_context, longitudinal_trajectory, resistance_emergence, ...) with
  the artifacts and gates each one demands.
* :mod:`artifact_registry` — the catalogue of all on-disk MORT-FM artifacts
  with their data sources, sizes, and the claims they unlock or block.

Together these are the *meta-layer* on top of the existing model and data
modules. They do not replace any existing code; they make it auditable.
"""

from resistancemap.mortfm.registries.artifact_registry import (  # noqa: F401
    ArtifactRecord,
    ArtifactRegistry,
)
from resistancemap.mortfm.registries.claim_gate_registry import (  # noqa: F401
    CANONICAL_CLAIM_LEVELS,
    CLAIM_LEVELS,
    ClaimLevel,
    lookup_claim_level,
)
from resistancemap.mortfm.registries.endpoint_registry import (  # noqa: F401
    CANONICAL_ENDPOINTS,
    EndpointSpec,
    lookup_endpoint,
)
from resistancemap.mortfm.registries.feature_space_registry import (  # noqa: F401
    CANONICAL_FEATURE_SPACES,
    FeatureSpace,
    lookup_feature_space,
)
