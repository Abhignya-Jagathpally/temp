"""
resistancemap.mortfm.drug_response
====================================

Drug-conditioned response head for the BeatAML hematologic specimen
track. The v16 head was mean-aggregated across drugs — per-drug
supervision was impossible because the head predicted a single vector
of (drug-candidate) risks regardless of which drug the specimen was
exposed to. v17 introduces an explicit (specimen, drug) -> response
head with per-drug supervision.
"""

from resistancemap.mortfm.drug_response.drug_conditioned_head import (  # noqa: F401
    DrugConditionedSpecimenResponseHead,
)
from resistancemap.mortfm.drug_response.per_drug_sampler import (  # noqa: F401
    PerDrugPairSampler,
)
