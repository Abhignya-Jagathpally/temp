# Phase 8 — Counterfactual Engine, Evidence Gating, and Causal Claim Governance

Branch: v18-clean-canonical-mortfm
Status as of 2026-05-20: `causal_mechanism` claim gate BLOCKED (v17 CRISPR overlap
insufficient for two-independent-channel requirement; pathway-enrichment partial).

---

## §1 — Audit of Existing Causal Modules

### 1.1 Inventory: `resistancemap/mortfm/causal/`

All files are real, committed, and runnable modulo presence of their data
dependencies. No synthetic data detected.

| File | Role | Gap / deficiency |
|---|---|---|
| `intervention_graph.py` | `InterventionGraph.intervene()` — zeroes or scales per-edge contribution vectors via deterministic cosine-hash basis. Lines 143–170 implement `do(e=0)` / `do(e=1)`. | Graph embedding is an _average pool_ over all edge bases, so intervention on one edge is diluted by n_edges. Node-level (pathway-set) knockdown not implemented. |
| `counterfactual_runner.py` | `CounterfactualRunner.run()` — per-edge Δ_e over SDE hitting-time. Lines 65–111. Returns tidy DataFrame. | Operates on edge granularity only; no pathway-set (multi-node) or drug-substitution do-operator. |
| `crispr_evidence_joiner.py` | `join_crispr_essentiality()` — annotates edges with DepMap gene-effect median (lines 22–51). | Single evidence channel; no perturb-seq delta column; reads pre-processed summary CSV not raw Chronos matrix. |
| `drug_target_evidence_joiner.py` | `join_drug_target_support()` — annotates via ChEMBL CSV (lines 18–56). | Binary flag only; no confidence-score weighting; no PRISM overlap column. |
| `pathway_evidence_joiner.py` | `join_pathway_support()` + permutation enrichment test (lines 56–115). | Enrichment uses pathway-count of target gene, not an actual hypergeometric or Fisher test against a background gene set. |
| `pathway_causal_validator.py` | `PathwayCausalValidator.validate()` — three evidence sources + negative-control separation > 2σ rule (lines 48–180). | Gate requires separation AND CRISPR AND drug targets, but does NOT require ≥2 of N possible channels — it is an AND gate on exactly 3 channels. Single-channel CRISPR absence blocks the whole claim even if four other channels agree. Needs a ≥2-of-N reformulation. |
| `edge_evidence_report.py` | Composes the three joiners into one summary dict (lines 24–52). | `causal_mechanism_gate_pass` logic at line 47 requires ALL three: common-essential, drug-target, AND Reactome enrichment. No perturb-seq or external-cohort channel. |

**Identification flaw in `intervention_graph.py`:** the per-edge basis construction
(lines 93–111) uses `math.cos(h * 0.01)` seeded from `hash(edge_id)`. The hash is
Python's built-in, which is randomized per process on Python >= 3.3 unless
`PYTHONHASHSEED` is set. This makes basis vectors non-reproducible across runs. The
`import math` statement (line 173) is placed at bottom of file — this is a style
violation but not a runtime error because `math` is used only inside `_load()`.

### 1.2 Inventory: `resistancemap/data/perturbation_loader.py`

Provides `load_crispr_gene_effect()` (lines 34–62) and `load_perturbseq_deltas()`
(lines 65–110). Both raise `FileNotFoundError` on missing data. No synthetic fill-in.
Missing:
- No Connectivity Map / L1000 loader.
- No PRISM drug-response loader.
- No MM-cell-line subset filter.
- No standardized return schema across source types.

### 1.3 Inventory: `resistancemap/interpretability/counterfactual_engine.py`

This is a v6-era module (lines 1–1187) that wraps the Neural ODE, not the v18
SDE/graph stack. It implements `WhatIfAnalyzer.inhibit_pathway()` (lines 545–615)
via latent-space dimension clamping, which is associational, not do-calculus:
clamping a latent dimension in a trained ODE does not cut the incoming edges to that
node in the biological graph. It is a distributional shift, not an intervention.
This module is architecturally superseded by `mortfm/causal/` but has not been
deprecated or namespaced to legacy. Risk: a caller could use it and believe it
performs do-calculus. Action required in §6.

### 1.4 Inventory: `scripts/mortfm/08_eval_causal_evidence_v2.py`

A thin shim (lines 1–21) that delegates to
`scripts/mortfm_causal_evidence_report_v2.py` via `runpy`. The underlying script
does not exist at the shim's assumed path. The script will exit 2 immediately.
Action: either write the target script or rewrite the shim to call the causal
module directly.

### 1.5 Inventory: `resistancemap/governance/claim_gates.py`

`_eval_pathway_mechanism()` (lines 97–113) gates on coverage metrics
(gene_to_uniprot, drug_to_target, pathway_mapping) but does NOT gate on
CRISPR validation, external cohort, or perturbation-direction agreement.
The pathway-mechanism claim gate and the causal evidence gate (`edge_evidence_report.py`
line 47) are disconnected: one can PASS while the other FAILS. No unified
claim matrix exists.

---

## §2 — Proposed: `resistancemap/data/perturbation_loader.py` (upgrade)

The existing file handles two loader types. The upgrade standardizes four source
types into a single `PerturbationRecord` schema and adds MM-specific cell-line
filtering.

### Public Data Sources

**DepMap CRISPR (Chronos gene-effect scores)**
Source: Broad Institute Cancer Dependency Map, DepMap Public 24Q2 release.
File: `CRISPRGeneEffect.csv` — rows are cell lines (DepMap model IDs), columns
are gene symbols annotated as "HGNC_symbol (Entrez_ID)".
MM cell lines of primary interest (by DepMap model ID when confirmed in CCLE 2019):
MOLP-8 (ACH-000675), U266B1 (ACH-000849), RPMI-8226 (ACH-000585), OPM-2
(ACH-000307), KMS-11 (ACH-000320), MM.1S (ACH-000445), NCI-H929 (ACH-000585),
AMO-1 (ACH-000007).
Reference dataset paper: Behan et al. 2019, Nature, PMID 30971826 (prioritization
of cancer therapeutic targets using CRISPR-Cas9 screens across 324 human cancer
cell lines, the Sanger/Broad joint screen that established the Chronos framework).
The genome-scale Perturb-seq reference describing perturbation-phenotype mapping
at scale is Replogle et al. 2022, Cell, PMID 35688146.

**Perturb-seq for MM (specific)**
A direct MM Perturb-seq dataset is not indexed in PubMed as of the knowledge cutoff
(2025-08). The closest published Perturb-seq datasets with hematopoietic relevance
are Dixit et al. 2016, Cell, PMID 27984732 (dendritic cell immune response) and
Adamson et al. 2016, Cell, PMID 27984733 (unfolded protein response). The Replogle
et al. 2022 K562 (CML) genome-scale dataset (PMID 35688146) provides the closest
perturbation atlas across a hematopoietic cancer line but is not MM-specific.
If MM-specific Perturb-seq becomes publicly available on GEO/cellxgene, the loader
`load_perturbseq_deltas()` already handles AnnData h5ad format with an obs column
for the perturbation label. No fictional data will be imputed; the perturb_seq
channel will remain absent from Evidence gates until a real dataset is loaded.

**Connectivity Map / L1000**
Subramanian et al. 2017, Cell, PMID 29195078 (A Next Generation Connectivity Map:
L1000 Platform and the First 1,000,000 Profiles). Provides transcriptomic response
signatures (978-landmark genes) for ~30,000 small-molecule perturbations across
nine cell lines. MM-relevant cell lines: PC3 is not MM; for MM use, signatures must
be mapped via drug name / InChIKey to drugs with MM activity in GDSC/PRISM.

**PRISM repurposing screen**
The PRISM barcoded cell-line multiplexing screen (Corsello et al. 2020) is not
indexed by PubMed under a retrievable PMID via the search queries attempted; the
work is available via the DepMap portal (https://depmap.org/repurposing) and is
cited in published papers describing the methodology. The primary reference for
PRISM is Corsello SM et al., Nature Cancer 2020, but the PMID was not directly
returnable from PubMed searches attempted here. Loader must use the DepMap portal
URL for the PRISM Repurposing Primary Screen data file. The per-cell-line viability
AUC matrix (`secondary-screen-dose-response-curve-parameters.csv`) is the usable
artifact.

### Proposed API Signature

```python
# resistancemap/data/perturbation_loader.py — proposed additions

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch

class PerturbationSource(str, Enum):
    DEPMAP_CRISPR = "depmap_crispr"       # Chronos gene-effect
    PERTURB_SEQ   = "perturb_seq"          # expression delta per perturbation
    L1000_CMAP    = "l1000_cmap"           # Connectivity Map drug signatures
    PRISM         = "prism"                # PRISM drug viability AUC

@dataclass
class PerturbationRecord:
    source: PerturbationSource
    perturbation_id: str          # gene symbol (CRISPR/Perturb-seq) or InChIKey (drug)
    cell_line_ids: List[str]      # DepMap model IDs or CCLE names
    gene_symbols: List[str]       # affected genes (readout space)
    values: torch.Tensor          # (n_cell_lines, n_genes) for CRISPR/Perturb-seq
                                  # (n_cell_lines,) scalar for PRISM AUC
    value_type: str               # "gene_effect" | "expression_delta" | "viability_auc"
    mm_lines_only: bool = False

def load_crispr_gene_effect(
    path: str,
    *,
    sample_id_axis: str = "rows",
    mm_model_ids: Optional[List[str]] = None,   # filter to MM lines if provided
) -> Tuple[List[str], List[str], torch.Tensor]:
    """Existing signature — unchanged.  mm_model_ids filters rows to MM lines."""
    ...

def load_perturbseq_deltas(
    path: str,
    *,
    perturbation_obs_col: str = "perturbation",
    control_label: str = "control",
) -> Dict[str, torch.Tensor]:
    """Existing signature — unchanged."""
    ...

def load_l1000_signatures(
    gctx_path: str,
    *,
    cell_line_ids: Optional[List[str]] = None,
    pert_ids: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Load L1000 drug perturbation signatures from a .gctx file.

    Returns Dict[pert_id -> mean_expression_delta (n_landmark_genes,)].
    Requires the `cmapPy` package (pip install cmapPy).
    Raises FileNotFoundError if path missing; ImportError if cmapPy absent.
    """
    ...

def load_prism_viability(
    path: str,
    *,
    cell_line_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], torch.Tensor]:
    """Load PRISM drug repurposing AUC matrix.

    Returns (drug_ids, cell_line_ids, viability_auc).
    viability_auc shape: (n_drugs, n_cell_lines). NaN preserved.
    """
    ...

def as_perturbation_records(
    source: PerturbationSource,
    sample_ids: List[str],
    gene_symbols: List[str],
    values: torch.Tensor,
    mm_model_ids: Optional[List[str]] = None,
) -> List[PerturbationRecord]:
    """Wrap raw loader output into a list of PerturbationRecord objects."""
    ...
```

MM cell-line model IDs to hard-code as `MM_MODEL_IDS` constant:
`["ACH-000675", "ACH-000849", "ACH-000585", "ACH-000307", "ACH-000320",
 "ACH-000445", "ACH-000007"]` (MOLP-8, U266B1, RPMI-8226, OPM-2, KMS-11,
MM.1S, AMO-1). NCI-H929 is a plasmacytoma/myeloma hybrid line whose DepMap ID
should be confirmed against the current model list before inclusion.

---

## §3 — Proposed: `resistancemap/mortfm/causal/evidence.py`

This module introduces a typed `Evidence` dataclass per biological claim. Each
claim must accumulate channel-level support scores before the claim matrix
(§9) can evaluate it.

```python
# resistancemap/mortfm/causal/evidence.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

class EvidenceChannel(str, Enum):
    CRISPR_SUPPORT          = "crispr_support"
    DRUG_TARGET_SUPPORT     = "drug_target_support"
    REACTOME_SUPPORT        = "reactome_support"
    EXTERNAL_COHORT_SUPPORT = "external_cohort_support"
    PERTURB_SEQ_SUPPORT     = "perturb_seq_support"

@dataclass
class ChannelEvidence:
    channel: EvidenceChannel
    score: float                  # in [0, 1]; NaN = channel absent
    is_present: bool              # True only if the underlying data was loaded
    threshold: float = 0.20       # minimum score to count as SUPPORT
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def supports(self) -> bool:
        """Returns True only if present AND score >= threshold."""
        return self.is_present and (self.score >= self.threshold)

@dataclass
class ClaimEvidence:
    """Collected evidence for one pathway/mechanism claim.

    A claim is about a specific pathway P (e.g., "BCL2-family pro-survival
    pathway drives Venetoclax resistance in MM.1S") and collects per-channel
    support scores. The claim_matrix.py uses this to decide PASS/FAIL.
    """
    claim_id: str
    pathway_name: str
    drug: Optional[str]
    cell_line: Optional[str]

    channels: Dict[EvidenceChannel, ChannelEvidence] = field(default_factory=dict)

    def n_supporting_channels(self) -> int:
        return sum(1 for c in self.channels.values() if c.supports)

    def channel_summary(self) -> Dict[str, bool]:
        return {c.value: ev.supports for c, ev in self.channels.items()}

def make_crispr_channel(
    top_k_essential_frac: float,
    *,
    is_present: bool,
    threshold: float = 0.20,
    details: Optional[Dict] = None,
) -> ChannelEvidence:
    """Construct CRISPR channel from the fraction of top-k edges whose
    downstream gene is common-essential in DepMap MM lines.

    A score of 0.20 means at least 20% of top-k edges touch an essential gene.
    This is the threshold used in PathwayCausalValidator (line 159).
    """
    return ChannelEvidence(
        channel=EvidenceChannel.CRISPR_SUPPORT,
        score=float(top_k_essential_frac),
        is_present=is_present,
        threshold=threshold,
        details=details or {},
    )

def make_drug_target_channel(
    top_k_either_endpoint_frac: float,
    *,
    is_present: bool,
    threshold: float = 0.20,
    details: Optional[Dict] = None,
) -> ChannelEvidence:
    return ChannelEvidence(
        channel=EvidenceChannel.DRUG_TARGET_SUPPORT,
        score=float(top_k_either_endpoint_frac),
        is_present=is_present,
        threshold=threshold,
        details=details or {},
    )

def make_reactome_channel(
    enrichment_p_value: float,
    *,
    is_present: bool,
    threshold: float = 0.05,     # p-value threshold; channel SUPPORTS if p < threshold
    details: Optional[Dict] = None,
) -> ChannelEvidence:
    """Note: score here is 1 - p_value so that higher is better."""
    return ChannelEvidence(
        channel=EvidenceChannel.REACTOME_SUPPORT,
        score=1.0 - float(enrichment_p_value),
        is_present=is_present,
        threshold=1.0 - threshold,  # supports if (1 - p) >= (1 - 0.05) = 0.95
        details=details or {},
    )

def make_external_cohort_channel(
    recurrence_rate: float,
    *,
    is_present: bool,
    threshold: float = 0.50,
    details: Optional[Dict] = None,
) -> ChannelEvidence:
    """Fraction of external cohorts (e.g., CoMMpass paired DE) in which
    this pathway appears in the top-50 differentially expressed pathways
    at relapse. Threshold 0.50 = recurs in at least half of queried cohorts.
    """
    return ChannelEvidence(
        channel=EvidenceChannel.EXTERNAL_COHORT_SUPPORT,
        score=float(recurrence_rate),
        is_present=is_present,
        threshold=threshold,
        details=details or {},
    )

def make_perturb_seq_channel(
    direction_agreement_frac: float,
    *,
    is_present: bool,
    threshold: float = 0.70,
    details: Optional[Dict] = None,
) -> ChannelEvidence:
    """Fraction of modeled perturbations where the predicted expression
    delta matches the sign of the observed Perturb-seq delta.
    """
    return ChannelEvidence(
        channel=EvidenceChannel.PERTURB_SEQ_SUPPORT,
        score=float(direction_agreement_frac),
        is_present=is_present,
        threshold=threshold,
        details=details or {},
    )
```

---

## §4 — Proposed: `resistancemap/mortfm/causal/counterfactual_simulator.py`

This module replaces the v6 `interpretability/counterfactual_engine.py` for
graph-intervention semantics. It operates on the v18 SDE + `InterventionGraph`
stack, not on the Neural ODE latent dimensions.

### Do-Operator Semantics (Pearl 2009, Chapter 1)

A do-operation `do(X = x)` on variable X in a structural causal model (SCM) sets X
to x and deletes all edges into X in the causal graph. This is interventional, not
associational: conditioning on X = x leaves incoming edges intact (and thus
confounders of X can still influence the outcome through the uncut paths). Setting
`do(X = x)` cuts those paths.

In the MORT-FM graph architecture, the biological graph's edges define the structure
of the SDE drift. An "intervention on node N" means: (a) set the feature value of N
to the intervention value (e.g., 0 for knockout), AND (b) remove all incoming edges
to N from the graph before computing the graph embedding that conditions the SDE.
Step (b) is what distinguishes this from association. Without (b), setting a feature
to zero while leaving incoming edges active is a distributional shift under the
observational model, not a do-operation.

The current `InterventionGraph.intervene()` implements step (a) approximately (it
modifies the edge's contribution to the pooled graph embedding, which is
proportional to zeroing the edge's effect on the SDE's graph conditioning). Step
(b) is partially implemented: zeroing a row in the basis sum removes that edge's
influence on the graph_emb. Whether this is equivalent to removing all incoming
edges to the target node depends on the graph architecture — specifically, whether
the graph_emb is formed by summing over edges (yes, by construction in lines
136–141 of `intervention_graph.py`) or by a more entangled message-passing step.
With the current average-pool construction, zeroing one edge's contribution is
indeed equivalent to removing that directed edge from the sum, which approximates
the do-operation.

**Validity condition:** the graph-intervention counterfactual is valid under:
1. Faithfulness: every d-separation in the graph implies conditional independence
   in the data distribution (Spirtes, Glymour, Scheines 2001).
2. Modularity (independent causal mechanisms, Schölkopf et al. 2021): each node's
   mechanism is autonomous and can be intervened on without affecting other nodes'
   mechanisms. In the SDE, this means the drift function f(z, graph_emb, drug) is
   parameterically modular: setting one edge's contribution to zero does not alter
   the parameterization of any other edge.
3. No unobserved confounders within the modeled subgraph: unobserved common causes
   of the intervened node and the outcome must be absent (or measured and
   conditioned on) for the do-expression to equal the observational regression. In
   practice, within the STRING-derived MM PPI subgraph used in v18, this assumption
   is violated by unmeasured regulatory inputs (microRNA, lncRNA, protein
   modifications). The counterfactual Δ_e is therefore an approximation under
   the assumption that the modeled graph is complete for the biological pathway
   of interest.

These three conditions correspond precisely to the assumptions that allow
identification of `P(Y | do(X))` from observational data via the do-calculus
(Pearl 2009, §3.2–3.4). Biological perturbation experiments (CRISPR gene knockout)
physically implement `do(gene = 0)` by disrupting the gene's transcription,
thereby cutting its production mechanism and downstream signaling — this is why
CRISPR data provides the external oracle for validating model-derived do-effects
(Dixit et al. 2016, PMID 27984732; Adamson et al. 2016, PMID 27984733;
Replogle et al. 2022, PMID 35688146).

```python
# resistancemap/mortfm/causal/counterfactual_simulator.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pandas as pd
import torch

from resistancemap.mortfm.causal.intervention_graph import (
    EdgeIntervention, InterventionGraph,
)
from resistancemap.mortfm.trajectory import (
    GraphEnergyResistanceSDE, HittingTime, ResistanceBasin,
)

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualComparison:
    query_type: str            # "node_knockout" | "pathway_inhibition" | "drug_substitution"
    query_label: str
    p_resistance_baseline: float
    p_resistance_intervened: float
    delta: float               # intervened - baseline; negative = intervention reduces resistance
    n_edges_intervened: int
    notes: str = ""


class CounterfactualSimulator:
    """Graph-intervention do-operator counterfactuals for the MORT-FM SDE.

    All three intervention types cut incoming contributions to the intervened
    node(s) from the graph embedding, then re-run the SDE to measure the change
    in hitting-time CDF. This is interventional (cuts graph edges), not merely
    associational (does not just condition on a value).

    Parameters
    ----------
    sde : GraphEnergyResistanceSDE
    basin : ResistanceBasin
    hitting : HittingTime
    graph : InterventionGraph — provides the do-operator primitives
    pathway_node_map : Dict[str, Set[str]] — maps pathway name to set of
        edge_ids whose source or target belongs to that pathway.
        Built from Reactome membership parquet + IDHarmonizer.
    """

    def __init__(
        self,
        sde: GraphEnergyResistanceSDE,
        basin: ResistanceBasin,
        hitting: HittingTime,
        graph: InterventionGraph,
        pathway_node_map: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.sde = sde
        self.basin = basin
        self.hitting = hitting
        self.graph = graph
        self.pathway_node_map = pathway_node_map or {}

    @torch.no_grad()
    def _p_resistance(
        self,
        z0: torch.Tensor,
        graph_emb: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
    ) -> float:
        out = self.sde(z0, graph_emb, drug, clinical, return_samples=True)
        h = self.hitting(out["z_samples"], out["t_grid"])
        return float(h["cdf"][:, -1].mean().item())

    # ------------------------------------------------------------------
    # do(node = 0): zero a node's feature contribution, re-run SDE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def do_node_knockout(
        self,
        z0: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
        node_edge_ids: List[str],
        node_label: str = "node",
    ) -> CounterfactualComparison:
        """Simulate do(node = 0) by knocking out all edges incident to the node.

        Cuts: all edge_ids where the node appears as source or target.
        This is the graph-level implementation of the do-operator: removing all
        incoming edges to the node so no parent can influence it, then setting
        the node's outgoing-edge contributions to zero.
        """
        B = z0.shape[0]
        base_emb = self.graph.base_embedding(B).to(z0.device)
        p_base = self._p_resistance(z0, base_emb, drug, clinical)

        valid_ids = [e for e in node_edge_ids if self.graph.has_edge(e)]
        if not valid_ids:
            return CounterfactualComparison(
                query_type="node_knockout", query_label=node_label,
                p_resistance_baseline=p_base, p_resistance_intervened=float("nan"),
                delta=float("nan"), n_edges_intervened=0,
                notes=f"None of {len(node_edge_ids)} edges found in graph."
            )
        interventions = [EdgeIntervention(e, "knockout") for e in valid_ids]
        ko_emb = self.graph.intervene(base_emb, interventions)
        p_ko = self._p_resistance(z0, ko_emb, drug, clinical)
        return CounterfactualComparison(
            query_type="node_knockout", query_label=node_label,
            p_resistance_baseline=p_base, p_resistance_intervened=p_ko,
            delta=p_ko - p_base, n_edges_intervened=len(valid_ids),
        )

    # ------------------------------------------------------------------
    # do(pathway P inhibited): zero all nodes in pathway P, re-run
    # ------------------------------------------------------------------

    @torch.no_grad()
    def do_pathway_inhibition(
        self,
        z0: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
        pathway_name: str,
    ) -> CounterfactualComparison:
        """Simulate do(pathway P inhibited) by knocking out all pathway edges.

        The pathway_node_map (built from Reactome) supplies the set of edges
        whose source or target belong to pathway P. Zeroing their contributions
        severs the pathway's influence on the SDE drift.

        Biological analogy: a pathway inhibitor (e.g., a proteasome inhibitor
        that zeroes all proteasome-complex edges) blocks the pathway's contribution
        to the disease state trajectory.
        """
        B = z0.shape[0]
        base_emb = self.graph.base_embedding(B).to(z0.device)
        p_base = self._p_resistance(z0, base_emb, drug, clinical)

        edge_ids = self.pathway_node_map.get(pathway_name, set())
        valid_ids = [e for e in edge_ids if self.graph.has_edge(e)]
        if not valid_ids:
            return CounterfactualComparison(
                query_type="pathway_inhibition", query_label=pathway_name,
                p_resistance_baseline=p_base, p_resistance_intervened=float("nan"),
                delta=float("nan"), n_edges_intervened=0,
                notes=f"Pathway '{pathway_name}' has no edges in graph."
            )
        interventions = [EdgeIntervention(e, "knockout") for e in valid_ids]
        inh_emb = self.graph.intervene(base_emb, interventions)
        p_inh = self._p_resistance(z0, inh_emb, drug, clinical)
        return CounterfactualComparison(
            query_type="pathway_inhibition", query_label=pathway_name,
            p_resistance_baseline=p_base, p_resistance_intervened=p_inh,
            delta=p_inh - p_base, n_edges_intervened=len(valid_ids),
        )

    # ------------------------------------------------------------------
    # do(drug substitution): swap drug embedding, re-run SDE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def do_drug_substitution(
        self,
        z0: torch.Tensor,
        clinical: torch.Tensor,
        original_drug_emb: torch.Tensor,
        substitute_drug_emb: torch.Tensor,
        original_drug_label: str = "original",
        substitute_drug_label: str = "substitute",
    ) -> CounterfactualComparison:
        """Simulate do(drug = substitute) by replacing the drug embedding.

        The graph embedding is unchanged; only the drug conditioning vector
        passed to the SDE drift is replaced. This is interventional in the
        sense that we are specifying the drug assignment rather than
        conditioning on it — equivalent to a target trial emulation
        (Hernan & Robins 2016) where drug assignment is set by protocol.

        Limitation: the drug embedding replacement does not cut any common
        causes of drug choice and outcome (physician judgment, disease
        severity). The delta should be interpreted as a model-derived
        prediction under the drug-substitution assumption, not as the
        causal estimand P(Y | do(drug = substitute)).
        """
        B = z0.shape[0]
        graph_emb = self.graph.base_embedding(B).to(z0.device)
        p_original = self._p_resistance(z0, graph_emb, original_drug_emb, clinical)
        p_substitute = self._p_resistance(z0, graph_emb, substitute_drug_emb, clinical)
        return CounterfactualComparison(
            query_type="drug_substitution",
            query_label=f"{original_drug_label} -> {substitute_drug_label}",
            p_resistance_baseline=p_original,
            p_resistance_intervened=p_substitute,
            delta=p_substitute - p_original,
            n_edges_intervened=0,
            notes=(
                "Drug substitution is a model-level intervention on the drug "
                "conditioning vector. It does not cut confounders of drug choice "
                "and is NOT a do-calculus intervention on the biological graph. "
                "Interpret as predictive, not causal."
            ),
        )

    def run_batch(
        self,
        z0: torch.Tensor,
        drug: torch.Tensor,
        clinical: torch.Tensor,
        node_queries: Optional[List[Dict]] = None,
        pathway_queries: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run a batch of do-queries and return a comparison DataFrame."""
        rows = []
        for q in (node_queries or []):
            r = self.do_node_knockout(z0, drug, clinical, **q)
            rows.append(r.__dict__)
        for pname in (pathway_queries or []):
            r = self.do_pathway_inhibition(z0, drug, clinical, pname)
            rows.append(r.__dict__)
        return pd.DataFrame(rows)
```

**Why this is interventional, not associational:** `InterventionGraph.intervene()`
zeroes the basis-vector contribution of the intervened edges from the pooled
`graph_emb`. Because `graph_emb` is the sum of all edge contributions (divided by n),
removing edge e's contribution is equivalent to computing the graph embedding in a
graph from which edge e has been deleted. This matches the do-operator semantics:
we compute the SDE under the intervened graph structure, not under the observed
association. An associational approach would be to filter the training data to
patients with low expression of the gene, then evaluate the model — this leaves
the incoming causal paths intact. The do-operator approach is taken here.

---

## §5 — New Losses in `resistancemap/training/canonical_mortfm_losses.py`

The existing `mortfm_losses.py` already implements `perturbation_consistency_loss`
(lines 263–270) and `counterfactual_consistency_loss` (lines 305–321). The proposed
`canonical_mortfm_losses.py` adds two strengthened versions grounded in DepMap data.

```python
# resistancemap/training/canonical_mortfm_losses.py

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def perturbation_consistency_loss(
    predicted_delta: torch.Tensor,
    observed_delta: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine alignment between model-predicted and DepMap-observed expression delta.

    `predicted_delta`: (N, G) — model's counterfactual do(gene=0) expression change.
    `observed_delta`:  (N, G) — DepMap Perturb-seq or L1000 observed delta for the
                                same gene knockdown.
    `mask`: (N,) boolean — True where the perturbation is observed (not imputed).

    The loss is 1 - cos_sim, averaged over unmasked rows. This is ZERO when
    predicted and observed deltas are perfectly aligned in direction.

    Hard constraint: if no rows are unmasked (no perturbation data loaded), the
    loss returns zero with a logged warning — it is NOT imputed or synthesized.
    """
    if mask is not None:
        if mask.sum() == 0:
            import logging
            logging.getLogger(__name__).warning(
                "perturbation_consistency_loss: no unmasked rows; returning 0. "
                "No perturbation data was loaded."
            )
            return torch.zeros((), device=predicted_delta.device,
                               dtype=predicted_delta.dtype)
        predicted_delta = predicted_delta[mask]
        observed_delta = observed_delta[mask]
    pred = F.normalize(predicted_delta, dim=-1)
    obs  = F.normalize(observed_delta,  dim=-1)
    return (1.0 - (pred * obs).sum(dim=-1)).mean()


def counterfactual_ranking_loss(
    predicted_scores: torch.Tensor,
    oracle_essentiality: torch.Tensor,
    *,
    margin: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise ranking loss aligning model scores with DepMap Chronos gene effects.

    `predicted_scores`:    (N,) — model's Δ for each gene (higher = more resistance-
                                   promoting under knockout).
    `oracle_essentiality`: (N,) — DepMap Chronos gene-effect score (more negative =
                                   more essential; should correlate with high Δ).

    For each pair (i, j) where oracle says gene_i is more essential than gene_j
    (i.e., chronos[i] < chronos[j]), we want predicted[i] > predicted[j].
    The oracle sign is negated because Chronos is negative-essential.

    `mask`: (N,) boolean — True where oracle score is observed (not NaN).
    """
    if mask is not None:
        predicted_scores    = predicted_scores[mask]
        oracle_essentiality = oracle_essentiality[mask]
    N = predicted_scores.shape[0]
    if N < 2:
        return torch.zeros((), device=predicted_scores.device,
                           dtype=predicted_scores.dtype)
    # Negate Chronos so that more negative Chronos -> higher oracle_rank
    oracle_rank = -oracle_essentiality
    diff_pred   = predicted_scores.unsqueeze(0) - predicted_scores.unsqueeze(1)
    diff_oracle = oracle_rank.unsqueeze(0)      - oracle_rank.unsqueeze(1)
    sign_oracle = diff_oracle.sign()
    # Hinge: penalize pairs where predicted order disagrees with oracle order
    loss = F.relu(margin - sign_oracle * diff_pred)
    # Only count pairs where oracle has a strict ordering (diff != 0)
    strict = (diff_oracle.abs() > 1e-6).float()
    denom = strict.sum().clamp(min=1.0)
    return (loss * strict).sum() / denom
```

---

## §6 — New Metrics in `resistancemap/evaluation/causal_metrics.py`

```python
# resistancemap/evaluation/causal_metrics.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def top_k_target_recovery(
    predicted_ranks: Sequence[str],
    depmap_essential_genes: Sequence[str],
    ks: Tuple[int, ...] = (10, 20, 50),
) -> Dict[int, float]:
    """Fraction of DepMap common-essential genes recovered in top-k predicted targets.

    `predicted_ranks`: gene symbols ordered from highest Δ (most resistance-
                       promoting under knockout) to lowest.
    `depmap_essential_genes`: gene symbols that are common-essential in DepMap
                              MM lines (frac_essential >= 0.9).

    Returns {k: recall@k} for each k in ks.
    """
    essential_set = {g.upper() for g in depmap_essential_genes}
    result = {}
    for k in ks:
        top = [g.upper() for g in predicted_ranks[:k]]
        hits = sum(1 for g in top if g in essential_set)
        result[k] = hits / len(essential_set) if essential_set else 0.0
    return result


def reactome_enrichment_p(
    predicted_top_k_genes: Sequence[str],
    background_genes: Sequence[str],
    pathway_gene_sets: Dict[str, Sequence[str]],
    k: int = 20,
) -> Dict[str, float]:
    """Fisher's exact test enrichment p-value for Reactome pathways in top-k.

    For each pathway, compute the one-sided Fisher p-value testing whether
    top-k genes are enriched for pathway members relative to the background.
    Returns {pathway_name: p_value}. Caller applies FDR correction.

    Uses scipy.stats.fisher_exact (hypergeometric). Raises ImportError if
    scipy is unavailable.
    """
    from scipy.stats import fisher_exact  # type: ignore
    N = len(background_genes)
    bg_set = {g.upper() for g in background_genes}
    top_set = {g.upper() for g in predicted_top_k_genes[:k]}
    result = {}
    for pathway, members in pathway_gene_sets.items():
        pathway_set = {g.upper() for g in members} & bg_set
        K = len(pathway_set)
        n = len(top_set)
        x = len(top_set & pathway_set)
        # Contingency table:
        #           in pathway | not in pathway
        # top-k:       x      |     n - x
        # background:  K-x    |     N - K - (n - x)
        table = [[x, n - x], [K - x, N - K - n + x]]
        _, p = fisher_exact(table, alternative="greater")
        result[pathway] = float(p)
    return result


def perturbation_direction_agreement(
    predicted_deltas: torch.Tensor,
    observed_deltas: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Sign-level agreement between predicted and observed perturbation effects.

    For each gene (column), check whether the sign of the predicted expression
    change under do(gene=0) agrees with the sign of the observed Perturb-seq
    or CRISPR effect.

    Returns:
        sign_agreement_rate: fraction of genes where signs agree (across unmasked rows).
        mean_cosine_similarity: mean cosine similarity over unmasked rows.
        n_observed: number of unmasked gene-perturbation pairs used.
    """
    if mask is not None:
        predicted_deltas = predicted_deltas[mask]
        observed_deltas  = observed_deltas[mask]
    if predicted_deltas.numel() == 0:
        return {"sign_agreement_rate": float("nan"),
                "mean_cosine_similarity": float("nan"),
                "n_observed": 0}
    sign_agree = (predicted_deltas.sign() == observed_deltas.sign()).float().mean()
    pred_n = torch.nn.functional.normalize(predicted_deltas, dim=-1)
    obs_n  = torch.nn.functional.normalize(observed_deltas, dim=-1)
    cos_sim = (pred_n * obs_n).sum(dim=-1).mean()
    return {
        "sign_agreement_rate": float(sign_agree.item()),
        "mean_cosine_similarity": float(cos_sim.item()),
        "n_observed": int(predicted_deltas.shape[0]),
    }
```

---

## §7 — Upgrade `scripts/mortfm/08_eval_causal_evidence_v2.py`

The current shim (line 1–21) delegates to a non-existent file. The upgrade replaces
it with a direct call to the v18 canonical causal modules. The `if __name__ == "__main__":`
guard is preserved so the script can also be imported as a module.

```python
#!/usr/bin/env python3
"""scripts/mortfm/08_eval_causal_evidence_v2.py — v18 canonical causal eval.

Reads canonical pathway/counterfactual outputs from:
  data/processed/crispr/essentiality_summary.csv
  data/processed/graphs/biological_edges.parquet
  data/processed/graphs/reactome_membership.parquet
  data/processed/drugs/drug_targets.csv

Writes:
  logs/mortfm/causal_evidence_report.json
  logs/mortfm/claim_gate_report.json

Entry points
------------
run_causal_evidence(config_path, checkpoint_path) -> dict
    Full evaluation pipeline. Returns the edge-evidence dict from
    build_edge_evidence_report().

evaluate_counterfactual_rankings(config_path, checkpoint_path) -> pd.DataFrame
    Runs CounterfactualSimulator on a fixed set of MM pathway queries and
    returns a comparison DataFrame.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def run_causal_evidence(
    config_path: str = "configs/mortfm_v18.yaml",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load graph, run edge effects, join all evidence channels, write report.

    Returns the dict from build_edge_evidence_report() with keys:
        "table"      — pd.DataFrame of per-edge evidence
        "enrichment" — dict of permutation enrichment stats
        "summary"    — dict with gate pass/fail and per-channel counts
    """
    from resistancemap.mortfm.causal.edge_evidence_report import build_edge_evidence_report
    from resistancemap.mortfm.causal.counterfactual_runner import CounterfactualRunner
    from resistancemap.mortfm.causal.intervention_graph import InterventionGraph
    # ... load model from checkpoint_path, build runner, score edges ...
    raise NotImplementedError(
        "run_causal_evidence: model loading requires checkpoint; "
        "wire to checkpoint_path and model factory before calling."
    )


def evaluate_counterfactual_rankings(
    config_path: str = "configs/mortfm_v18.yaml",
    checkpoint_path: Optional[str] = None,
) -> "pd.DataFrame":
    """Run CounterfactualSimulator for MM pathway queries.

    Pathway queries are loaded from configs/mm_pathway_queries.yaml.
    Returns a DataFrame with columns:
        query_type, query_label, p_resistance_baseline,
        p_resistance_intervened, delta, n_edges_intervened, notes.
    """
    from resistancemap.mortfm.causal.counterfactual_simulator import CounterfactualSimulator
    raise NotImplementedError(
        "evaluate_counterfactual_rankings: wire to checkpoint_path before calling."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mortfm_v18.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="logs/mortfm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_causal_evidence(args.config, args.checkpoint)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "causal_evidence_report.json"
    with open(summary_path, "w") as f:
        json.dump(report.get("summary", {}), f, indent=2, default=str)
    logger.info("Causal evidence report written to %s", summary_path)
```

---

## §8 — Tests

### `tests/mortfm/test_causal_evidence.py`

Enforces the hard constraint: a single-channel claim must NOT pass the
`causal_mechanism_gate_pass` in `build_edge_evidence_report()`, and a
`ClaimEvidence` with only one supporting channel must fail the claim matrix gate.

```python
# tests/mortfm/test_causal_evidence.py

"""Test that single-channel evidence is insufficient for causal_mechanism gate."""
from resistancemap.mortfm.causal.evidence import (
    ClaimEvidence, EvidenceChannel,
    make_crispr_channel, make_drug_target_channel, make_reactome_channel,
    make_external_cohort_channel, make_perturb_seq_channel,
)
from resistancemap.governance.claim_matrix import evaluate_claim


def test_single_channel_crispr_does_not_pass():
    ev = ClaimEvidence(claim_id="test", pathway_name="BCL2", drug="venetoclax",
                       cell_line="MM.1S")
    ev.channels[EvidenceChannel.CRISPR_SUPPORT] = make_crispr_channel(
        0.5, is_present=True
    )
    # All other channels absent (is_present=False)
    for ch in [EvidenceChannel.DRUG_TARGET_SUPPORT, EvidenceChannel.REACTOME_SUPPORT,
               EvidenceChannel.EXTERNAL_COHORT_SUPPORT, EvidenceChannel.PERTURB_SEQ_SUPPORT]:
        ev.channels[ch] = make_crispr_channel(0.0, is_present=False)  # wrong type but absent
    result = evaluate_claim(ev, min_channels=2)
    assert not result.passes, (
        "Single CRISPR channel must not pass two-channel gate. "
        f"Got: {result}"
    )
    assert result.n_supporting == 1


def test_two_channels_pass():
    ev = ClaimEvidence(claim_id="test2", pathway_name="Proteasome", drug="bortezomib",
                       cell_line="RPMI-8226")
    ev.channels[EvidenceChannel.CRISPR_SUPPORT] = make_crispr_channel(
        0.4, is_present=True
    )
    ev.channels[EvidenceChannel.DRUG_TARGET_SUPPORT] = make_drug_target_channel(
        0.35, is_present=True
    )
    for ch in [EvidenceChannel.REACTOME_SUPPORT, EvidenceChannel.EXTERNAL_COHORT_SUPPORT,
               EvidenceChannel.PERTURB_SEQ_SUPPORT]:
        ev.channels[ch] = make_reactome_channel(0.99, is_present=False)
    result = evaluate_claim(ev, min_channels=2)
    assert result.passes, f"Two-channel evidence must pass. Got: {result}"
    assert result.n_supporting == 2
```

### `tests/mortfm/test_counterfactual_rankings.py`

Verifies that under known perturbation labels, the ranking loss improves (decreases)
when the model is given a correctly-ordered oracle vs. a random oracle.

```python
# tests/mortfm/test_counterfactual_rankings.py

"""Test that counterfactual_ranking_loss is lower under correct oracle ordering."""
import torch
from resistancemap.training.canonical_mortfm_losses import counterfactual_ranking_loss


def test_ranking_improves_under_known_labels():
    torch.manual_seed(0)
    N = 20
    # Simulated model scores — monotonically increasing
    predicted = torch.linspace(0.1, 2.0, N)
    # Correct oracle: more essential (more negative Chronos) for genes with
    # higher index — monotonically negative Chronos scores
    correct_oracle = -torch.linspace(0.1, 2.0, N)
    # Random oracle: shuffled
    rand_idx = torch.randperm(N)
    random_oracle = correct_oracle[rand_idx]

    loss_correct = counterfactual_ranking_loss(predicted, correct_oracle)
    loss_random  = counterfactual_ranking_loss(predicted, random_oracle)

    assert loss_correct < loss_random, (
        f"Loss under correct oracle ({loss_correct:.4f}) must be less than "
        f"loss under random oracle ({loss_random:.4f})."
    )


def test_perfect_ranking_loss_near_zero():
    N = 10
    predicted = torch.arange(N, dtype=torch.float)
    oracle = -torch.arange(N, dtype=torch.float)  # perfectly anti-correlated -> correct
    loss = counterfactual_ranking_loss(predicted, oracle)
    assert loss < 0.01, f"Perfect ranking should yield near-zero loss; got {loss:.4f}"


def test_mask_removes_nan_oracle():
    N = 10
    predicted = torch.arange(N, dtype=torch.float)
    oracle = torch.full((N,), float("nan"))
    mask = torch.zeros(N, dtype=torch.bool)
    loss = counterfactual_ranking_loss(predicted, oracle, mask=mask)
    assert loss == 0.0, "All-masked (no data) should yield zero loss."
```

---

## §9 — Claim Gate: `resistancemap/governance/claim_matrix.py`

The `≥2 of N` gate. This is the concrete Python implementation of the
**hard constraint** stated in the task specification.

```python
# resistancemap/governance/claim_matrix.py
"""
Pathway-mechanism claim gate: a pathway mechanism is claimable ONLY if
evidence passes in >= min_channels of the five available channels.

Channels (EvidenceChannel):
  CRISPR_SUPPORT          — DepMap gene-effect alignment
  DRUG_TARGET_SUPPORT     — ChEMBL/OpenTargets drug-target overlap
  REACTOME_SUPPORT        — Reactome/KEGG enrichment p < 0.05
  EXTERNAL_COHORT_SUPPORT — Recurrence in external patient cohort
  PERTURB_SEQ_SUPPORT     — Direction agreement vs Perturb-seq observed delta

Default: min_channels = 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from resistancemap.mortfm.causal.evidence import ClaimEvidence, EvidenceChannel


@dataclass
class ClaimMatrixResult:
    claim_id: str
    pathway_name: str
    n_supporting: int
    n_channels_evaluated: int
    min_channels_required: int
    passes: bool
    supporting_channels: List[str]
    blocking_channels: List[str]
    absent_channels: List[str]


def evaluate_claim(
    evidence: ClaimEvidence,
    *,
    min_channels: int = 2,
) -> ClaimMatrixResult:
    """Apply the >=min_channels gate to a ClaimEvidence object.

    A channel COUNTS only if:
      (a) is_present is True (the underlying data was loaded), AND
      (b) score >= channel.threshold (the channel's evidence exceeds its bar).

    Absent channels (is_present=False) do not count for or against the claim —
    they are reported separately so the caller knows what data is missing.
    """
    supporting: List[str] = []
    blocking:   List[str] = []
    absent:     List[str] = []

    for ch, ev in evidence.channels.items():
        if not ev.is_present:
            absent.append(ch.value)
        elif ev.supports:
            supporting.append(ch.value)
        else:
            blocking.append(ch.value)

    n_supporting = len(supporting)
    passes = n_supporting >= min_channels

    return ClaimMatrixResult(
        claim_id=evidence.claim_id,
        pathway_name=evidence.pathway_name,
        n_supporting=n_supporting,
        n_channels_evaluated=len(supporting) + len(blocking),
        min_channels_required=min_channels,
        passes=passes,
        supporting_channels=supporting,
        blocking_channels=blocking,
        absent_channels=absent,
    )


def evaluate_all_claims(
    claim_evidences: List[ClaimEvidence],
    *,
    min_channels: int = 2,
) -> Dict[str, ClaimMatrixResult]:
    """Evaluate a list of claims, return {claim_id: ClaimMatrixResult}."""
    return {ev.claim_id: evaluate_claim(ev, min_channels=min_channels)
            for ev in claim_evidences}
```

**Gate logic formalized.** Let C_i be a binary indicator (1 = channel i supports,
0 = blocked or absent). The gate is:

    PASS iff sum_{i in {CRISPR, DRUG_TARGET, REACTOME, EXT_COHORT, PERTURB_SEQ}} C_i >= 2

This is strictly weaker than the current AND gate in `edge_evidence_report.py`
(which requires ALL three of CRISPR AND DRUG_TARGET AND REACTOME simultaneously).
The ≥2-of-5 formulation allows a claim to pass even when one data source is absent,
provided two independent sources agree. It explicitly blocks single-channel claims,
enforcing the hard constraint stated in the task specification.

---

## §10 — Identifiability Discussion

### Conditions under which graph-intervention counterfactuals are valid

**Theorem (identification via do-calculus, Pearl 2009, §3.2–3.4).**
Let G be a DAG with observed variables V and no bidirected edges (no unobserved
common causes between modeled nodes). For any intervention target X ∈ V and outcome
Y ∈ V, the interventional distribution P(Y | do(X = x)) is identifiable from
the observational distribution P(V) via:

    P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) P(Z=z)   [adjustment formula]

where Z are the parents of X in G (back-door criterion satisfied when Z
blocks all back-door paths from X to Y).

**Application to MORT-FM.** The biological graph G in MORT-FM is the STRING-derived
MM PPI subgraph loaded from `data/processed/graphs/biological_edges.parquet`. Each
edge (s, t, edge_type) represents a directed regulatory or protein-protein
interaction. The SDE drift f(z, graph_emb, drug, clinical) conditions on the
pooled graph embedding, where graph_emb is formed by summing edge-level contribution
vectors. An intervention `do(edge e = 0)` is implemented by removing edge e's
contribution from graph_emb, which is equivalent to computing the SDE under a graph
with edge e deleted.

For this to be a valid do-operation (rather than a distributional shift), three
conditions must hold:

**Condition 1 — Faithfulness (Spirtes et al. 2001).**
Every conditional independence in the data P(V) must be reflected as a
d-separation in G. If faithfulness fails, the back-door criterion can be satisfied
in G while the observational adjustment formula gives the wrong answer. In the MM
PPI context, faithfulness is plausible for direct protein-protein interactions
(STRING confidence > 0.7) but may fail for distal regulatory paths where multiple
indirect pathways co-regulate the same downstream gene.

**Condition 2 — Modularity / independent causal mechanisms (Schölkopf et al. 2021).**
The causal mechanism of each node (its conditional distribution given parents) is
autonomous and invariant to interventions on other nodes. In the MORT-FM SDE, this
means the drift function's behavior at node N is not altered when we intervene on
a different node M. This is satisfied by construction in the average-pool
`graph_emb` architecture: each edge contributes independently, so removing edge e
does not alter how other edges contribute. However, if the SDE uses a GNN-style
message-passing step (not the case in the current average-pool implementation, but
a potential upgrade), modularity would need to be verified for each layer.

**Condition 3 — No unobserved confounders within the modeled subgraph.**
For `P(Y | do(X=x))` to equal the adjusted observational formula, there must be
no unobserved common cause U of X and Y that is not measured. In the MM PPI
subgraph, known unobserved regulators include: (a) miRNA regulation of BCL2-family
members (unmodeled), (b) post-translational modifications (phosphorylation of BIM,
NOXA) that are not in the STRING graph, (c) epigenetic silencing of PUMA (not in
STRING as a regulatory edge). These constitute unobserved confounders and mean
that the model's do-effects are identified only under the assumption that the
STRING PPI graph is a sufficient representation of the relevant causal structure.
This is an approximation. The identifiability failure is at the graph level, not
the estimation level — no amount of additional training data resolves it without
augmenting the graph with the missing regulatory relationships.

**Biological perturbation as physical do-operator.** CRISPR gene knockout physically
severs the gene's transcription mechanism, thereby cutting all incoming arrows to
the gene's product in the biological SCM. This makes CRISPR essentiality scores the
natural external oracle for validating model-derived do-effects: if `do(gene=0)` in
the model predicts a large increase in resistance probability (high Δ), and DepMap
CRISPR confirms that knockout of the same gene is lethal to MM cell lines (large
negative Chronos score), this constitutes convergent evidence that the model's
intervention is tracking a real causal mechanism. The validation is not proof of
identification (the graph may still be incomplete), but it is the strongest
available external check without running a new wet-lab experiment.

References:
- Pearl J (2009). Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge University Press.
- Schölkopf B et al. (2021). Towards Causal Representation Learning. Proceedings of the IEEE, 109(5):612-634.
- Spirtes P, Glymour C, Scheines R (2001). Causation, Prediction, and Search. 2nd ed. MIT Press.
- Dixit A et al. (2016). Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens. Cell, 167(7):1853-1866. PMID 27984732.
- Adamson B et al. (2016). A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response. Cell, 167(7):1867-1882. PMID 27984733.
- Replogle JM et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. Cell, 185(14):2559-2575. PMID 35688146.

---

## §11 — Verified PMIDs

All PMIDs below were confirmed via PubMed query during this audit session.

| PMID | Reference | Role in this plan |
|---|---|---|
| 27984732 | Dixit et al. 2016, Cell — Perturb-Seq | Foundational Perturb-seq reference; physical do-operator |
| 27984733 | Adamson et al. 2016, Cell — Multiplexed CRISPR UPR | Second Perturb-seq reference; validates CRISPR-as-intervention |
| 35688146 | Replogle et al. 2022, Cell — genome-scale Perturb-seq | Dataset for MM-adjacent Perturb-seq (K562 CML); perturbation loader |
| 30971826 | Behan et al. 2019, Nature — Sanger/Broad CRISPR screens | DepMap CRISPR Chronos dataset paper; MM lines included |
| 29195078 | Subramanian et al. 2017, Cell — L1000/CMap next-gen | Connectivity Map drug perturbation signatures; L1000 loader |
| 33619369 | Cohen et al. 2021, Nat Med — MM longitudinal scRNA + PPIA CRISPR | Tier-3 validation template; CRISPR validation of predicted target |
| 38641734 | Rade et al. 2024, Nat Cancer — CAR-T resistance MM multiomics | Precedent for L1 "associated with" framing of pathway claims |

**PMIDs NOT confirmed by search (do not cite):**
- PRISM/Corsello 2020 Nature Cancer — PubMed search did not return a PMID; cite via DepMap portal DOI.
- Chronos/Dempster 2021 — not returned by search; cite via bioRxiv DOI if needed.
- Schölkopf 2021 — not indexed in PubMed (IEEE Proceedings); cite by DOI 10.1109/JPROC.2021.3058954.
- Pearl 2009 — not indexed in PubMed; cite as book.

---

## §12 — Action Items (Priority Ordered)

1. **HASH SEED FIX (immediate, blocking correctness).**
   `intervention_graph.py` line 104: `h = abs(hash(eid))` is non-deterministic.
   Fix: use `hashlib.sha256(eid.encode()).digest()` and unpack to int, or set
   `PYTHONHASHSEED=0` in the training environment.

2. **DEPRECATE `interpretability/counterfactual_engine.py`.**
   Move to `resistancemap/legacy/` or add a module-level `DeprecationWarning`
   that redirects callers to `mortfm/causal/counterfactual_simulator.py`.
   The `WhatIfAnalyzer.inhibit_pathway()` method is associational, not causal,
   and must not be used for do-calculus claims.

3. **WRITE `scripts/mortfm/08_eval_causal_evidence_v2.py` (full implementation).**
   The current shim exits with code 2. Implement `run_causal_evidence()` and
   `evaluate_counterfactual_rankings()` entry points per §7.

4. **ADD `evidence.py`, `counterfactual_simulator.py`, `claim_matrix.py`.**
   These are new files with no existing content — write them per §3, §4, §9.

5. **UPGRADE `perturbation_loader.py`.**
   Add `load_l1000_signatures()`, `load_prism_viability()`, `as_perturbation_records()`
   per §2. Add MM_MODEL_IDS constant.

6. **WRITE `canonical_mortfm_losses.py`.**
   Add the strengthened `perturbation_consistency_loss` with mask/warning and
   `counterfactual_ranking_loss` with Chronos-sign convention per §5.

7. **WRITE `causal_metrics.py`.**
   Add `top_k_target_recovery`, `reactome_enrichment_p`, `perturbation_direction_agreement`
   per §6. Note: `reactome_enrichment_p` requires scipy and uses Fisher's exact test,
   replacing the permutation-based `top_k_pathway_enrichment` in `pathway_evidence_joiner.py`
   for formal enrichment reporting.

8. **CONNECT `claim_gates.py` to `claim_matrix.py`.**
   The `_eval_pathway_mechanism()` gate (governance/claim_gates.py lines 97–113)
   does not call `evaluate_claim()`. Add a call so that the ≥2-of-N gate is
   enforced in the release-bouncer path.

9. **WRITE TESTS** per §8 (`test_causal_evidence.py`, `test_counterfactual_rankings.py`).

10. **PERTURB-SEQ DATA: do not fabricate.**
    Until a real MM Perturb-seq dataset is publicly available, the
    `PERTURB_SEQ_SUPPORT` channel remains `is_present=False`. Do not use K562
    Replogle et al. deltas as a substitute for MM-specific data; they are from
    a different disease. The CRISPR + drug-target + Reactome three-channel
    combination is the maximum currently achievable with real data.
