"""
resistancemap/mortfm/registries/feature_space_registry.py
=========================================================
Canonical catalogue of every feature axis MORT-FM operates on.

Why: Block A's RNA encoder is trained on the variance-top-2000 DepMap genes;
Block C's encoder lives on the HVG-union of two scRNA datasets (3,535 genes);
Block D's graph indexes proteins/drugs/pathways by their own IDs. The
integration step *cannot* silently broadcast a Block-C input into a Block-A
encoder. This registry makes the mapping explicit so the routing decision
is data-driven, not heuristic.

Honest behaviour
----------------
* Each registered space carries a ``feature_names_path`` pointing to the
  on-disk file that lists its features. Callers MUST resolve the names via
  this path rather than re-deriving from the data each time — otherwise
  Block A's selection logic could drift from the integrated checkpoint.
* If a space's source file is missing, ``status="missing"`` is recorded;
  consumers raise rather than silently default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

logger = logging.getLogger(__name__)


Modality = Literal[
    "rna", "protein", "drug", "pathway", "graph_node", "scrna", "atac",
    "methylation", "histone_ptm", "phosphoproteomics",
]


@dataclass
class FeatureSpace:
    feature_space_id: str
    modality: Modality
    n_features: int
    source: str
    feature_names_path: Optional[str] = None
    notes: str = ""
    derived_from: List[str] = field(default_factory=list)
    block_owner: Optional[str] = None  # which block defined this space

    @property
    def status(self) -> str:
        if self.feature_names_path is None:
            return "implicit"
        return "present" if Path(self.feature_names_path).exists() else "missing"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


CANONICAL_FEATURE_SPACES: Dict[str, FeatureSpace] = {
    "rna_block_a_top2000": FeatureSpace(
        feature_space_id="rna_block_a_top2000",
        modality="rna",
        n_features=2000,
        source="DepMap RNA × GDSC overlap, top-2000 variance genes",
        feature_names_path=None,  # derived at checkpoint-load time by mortfm_align_beataml_features._features_from_block_a
        notes=(
            "Block A's RNA feature axis. Reproduced deterministically by "
            "scripts/mortfm_align_beataml_features._features_from_block_a from "
            "the Block A checkpoint config; not stored as a flat CSV by default."
        ),
        block_owner="A",
    ),
    "rna_block_b_aligned_to_a": FeatureSpace(
        feature_space_id="rna_block_b_aligned_to_a",
        modality="rna",
        n_features=2000,
        source="BeatAML 22,843 genes projected onto rna_block_a_top2000 (1,674 native + 326 zero-filled)",
        feature_names_path=None,
        notes=(
            "Same column order as rna_block_a_top2000 — that's the whole point "
            "of the alignment step. BeatAML mask at "
            "data/processed/beataml/beataml_feature_mask.parquet records which "
            "columns are native vs zero-filled."
        ),
        derived_from=["rna_block_a_top2000"],
        block_owner="B",
    ),
    "scrna_block_c_hvg_union_3535": FeatureSpace(
        feature_space_id="scrna_block_c_hvg_union_3535",
        modality="scrna",
        n_features=3535,
        source="HVG-union of GSE124310 + GSE271107 (top-2000 HVGs per dataset)",
        feature_names_path=None,
        notes=(
            "Block C's pseudobulk axis. DIFFERENT from Block A's; do NOT feed a "
            "Block C latent into Block A's encoder or vice versa without "
            "re-alignment via resistancemap.data.feature_alignment."
        ),
        block_owner="C",
    ),
    "protein_uniprot_human": FeatureSpace(
        feature_space_id="protein_uniprot_human",
        modality="protein",
        n_features=20659,
        source="UniProt UP000005640_9606 human reference proteome",
        feature_names_path="data/processed/metadata/gene_protein_identifier_map.csv",
        notes=(
            "20,544 of these are mapped to feature_genes via the harmonization "
            "map (73.2% coverage of all feature genes)."
        ),
        block_owner="D",
    ),
    "protein_esm2_8m_cached": FeatureSpace(
        feature_space_id="protein_esm2_8m_cached",
        modality="protein",
        n_features=1781,  # number of cached sequences; per-sequence embedding dim is 320
        source="ESM-2 8M (facebook/esm2_t6_8M_UR50D) cached subset",
        feature_names_path="data/processed/proteins/esm_embeddings__facebook__esm2_t6_8M_UR50D.pt",
        notes=(
            "Mean-pooled last-hidden-state, 320-dim per sequence. n_features here "
            "counts the cached sequences (1,781); per-sequence vector length is 320. "
            "Expand by re-running scripts/mortfm_compute_esm2_embeddings.py with "
            "--max-sequences."
        ),
        derived_from=["protein_uniprot_human"],
        block_owner="D",
    ),
    "drug_chembl_v34_filtered": FeatureSpace(
        feature_space_id="drug_chembl_v34_filtered",
        modality="drug",
        n_features=5092,
        source="ChEMBL v34 drug-targets export (approved + mechanism slice)",
        feature_names_path="data/processed/drugs/drug_ontology.csv",
        notes=(
            "Covers 32.5% GDSC / 37.6% PRISM / 46.7% BeatAML drug names "
            "after 5-tier matching; honest gap documented in "
            "logs/mortfm/drug_identifier_mapping_report.json."
        ),
        block_owner="D",
    ),
    "pathway_reactome": FeatureSpace(
        feature_space_id="pathway_reactome",
        modality="pathway",
        n_features=2845,
        source="Reactome pathway membership table",
        feature_names_path="data/processed/pathways/reactome_membership.parquet",
        notes="2,845 pathway nodes attached to the biological graph by Block A.",
        block_owner="A",
    ),
    "graph_node_string_v12": FeatureSpace(
        feature_space_id="graph_node_string_v12",
        modality="graph_node",
        n_features=20919,
        source="STRING v12 9606 protein nodes (graph builder output)",
        feature_names_path="data/processed/graphs/protein_nodes_from_graph.csv",
        notes=(
            "STRING-side protein identifiers — 9606.ENSP*. NOT UniProt; "
            "use the harmonization map to bridge to UniProt accessions."
        ),
        block_owner="A",
    ),
}


def lookup_feature_space(fs_id: str) -> Optional[FeatureSpace]:
    return CANONICAL_FEATURE_SPACES.get(fs_id)


def write_feature_space_registry(
    out_path: str = "data/processed/features/mortfm_feature_spaces.json",
    extras: Optional[Iterable[FeatureSpace]] = None,
) -> str:
    spaces = dict(CANONICAL_FEATURE_SPACES)
    for fs in extras or []:
        spaces[fs.feature_space_id] = fs
    payload = {fid: fs.to_dict() for fid, fs in spaces.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote feature-space registry (%d entries) -> %s", len(payload), out)
    return str(out)
