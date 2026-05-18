"""
resistancemap.mortfm.graph
===========================

ID harmonization + patient-side graph construction for MORT-FM v17.

  * :class:`IDHarmonizer` — bidirectional bridge between the four ID
    namespaces that show up in the biological graph: STRING
    ``9606.ENSP*``, ENSP, UniProt accession, HGNC symbol. Built from
    the on-disk STRING aliases file ``data/raw/string/9606.protein.
    aliases.v12.0.txt.gz`` — uses `Ensembl_HGNC_symbol` and
    `UniProt_AC` source codes, no fabrication.

The InterventionGraph + PathwayCausalValidator both consume this so the
causal-mechanism CRISPR cross-check no longer fails for a label-mapping
artifact (v16 limitation, fixed here).
"""

from resistancemap.mortfm.graph.id_harmonizer import IDHarmonizer  # noqa: F401
