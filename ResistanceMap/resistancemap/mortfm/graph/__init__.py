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

  * :class:`GraphStore` — loads and caches the Block A biological graph
    from ``biological_edges.parquet`` + ``protein_nodes.csv``.

  * :class:`PatientNodeFeatureBuilder` — maps patient molecular features
    (RNA, proteomics) onto graph nodes via HGNC symbol matching.

  * :class:`PatientGraphEncoder` — GNN (GAT with fallback) that
    propagates patient-specific features over the PPI graph, producing a
    per-patient ``(B, d_graph)`` embedding for the SDE drift.

  * :class:`PatientGraphEmbeddingFn` — callable wrapper compatible with
    :class:`CanonicalMORTFMTrainer`'s ``graph_emb_fn`` parameter.

  * :func:`build_patient_graph_embedding_fn` — factory returning a
    ready-to-use :class:`PatientGraphEmbeddingFn`.

  * :class:`HeteroBiologicalGraph` — legacy alias for GraphStore (v17
    backward compatibility).

  * :func:`build_patient_graph_emb_fn` — legacy factory (v17 backward
    compatibility).

The InterventionGraph + PathwayCausalValidator both consume this so the
causal-mechanism CRISPR cross-check no longer fails for a label-mapping
artifact (v16 limitation, fixed here).
"""

from resistancemap.mortfm.graph.id_harmonizer import IDHarmonizer  # noqa: F401
from resistancemap.mortfm.graph.patient_graph_encoder import (  # noqa: F401
    GraphStore,
    HeteroBiologicalGraph,
    PatientGraphEmbeddingFn,
    PatientGraphEncoder,
    PatientNodeFeatureBuilder,
    build_patient_graph_emb_fn,
    build_patient_graph_embedding_fn,
)
