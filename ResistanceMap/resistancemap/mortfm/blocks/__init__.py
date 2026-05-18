"""
resistancemap.mortfm.blocks
============================

Thin adapters that wrap the on-disk MORT-FM artifacts (Block A cell-line
foundation, Block B BeatAML fine-tune, Block C scRNA state encoder,
Block D ESM-2 cache, Block E integrated checkpoint) into Python objects
the rest of the pipeline can compose.

Each adapter exposes:
  * a ``load_*`` function that returns the artifact (model, cache, manifest)
  * an ``artifact_record`` factory that emits a populated
    :class:`~resistancemap.mortfm.registries.artifact_registry.ArtifactRecord`

The adapters never re-train. They never re-derive feature sets — that
already lives in the existing data-prep scripts. They only describe.
"""

from resistancemap.mortfm.blocks.block_a_cellline import (  # noqa: F401
    load_block_a,
    block_a_artifact_record,
)
from resistancemap.mortfm.blocks.block_b_beataml import (  # noqa: F401
    load_block_b,
    block_b_artifact_record,
)
from resistancemap.mortfm.blocks.block_c_scrna import (  # noqa: F401
    load_block_c,
    block_c_artifact_record,
)
from resistancemap.mortfm.blocks.block_d_graph import (  # noqa: F401
    load_block_d,
    block_d_artifact_record,
)
from resistancemap.mortfm.blocks.block_e_integrated import (  # noqa: F401
    load_block_e,
    block_e_artifact_record,
)
