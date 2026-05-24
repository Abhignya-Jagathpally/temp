"""
resistancemap.mortfm.graph
===========================

ID harmonization + patient-side graph construction for MORT-FM.

  * :class:`IDHarmonizer` — bidirectional bridge between the four ID
    namespaces that show up in the biological graph: STRING
    ``9606.ENSP*``, ENSP, UniProt accession, HGNC symbol. Built from
    the on-disk STRING aliases file ``data/raw/string/9606.protein.
    aliases.v12.0.txt.gz`` — uses `Ensembl_HGNC_symbol` and
    `UniProt_AC` source codes, no fabrication.

  * :class:`HeteroBiologicalGraph` — builds a PyG HeteroData from the
    on-disk ``biological_edges.parquet`` + ``protein_nodes.csv`` with
    four node types (gene, protein, drug, pathway) and three edge types.

  * :class:`PatientGraphEncoder` — patient-conditioned heterogeneous
    GNN that propagates patient RNA expression and drug exposure over
    the biological graph via multi-layer HeteroGATConv, producing a
    per-patient ``(B, d_graph)`` embedding for the SDE drift.

  * :func:`build_patient_graph_emb_fn` — factory returning a callable
    compatible with :class:`CanonicalMORTFMTrainer`'s ``graph_emb_fn``.

The InterventionGraph + PathwayCausalValidator both consume this so the
causal-mechanism CRISPR cross-check no longer fails for a label-mapping
artifact (v16 limitation, fixed here).
"""

from resistancemap.mortfm.graph.id_harmonizer import IDHarmonizer  # noqa: F401
from resistancemap.mortfm.graph.patient_graph_encoder import (  # noqa: F401
    HeteroBiologicalGraph,
    PatientGraphEncoder,
    build_patient_graph_emb_fn,
)
