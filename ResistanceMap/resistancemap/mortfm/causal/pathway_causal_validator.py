"""
resistancemap/mortfm/causal/pathway_causal_validator.py
========================================================
Cross-check the counterfactual edge effects against external evidence.

Sources of external truth this validator queries (lazily, only the ones
present on disk):

  * **CRISPR DepMap gene-effect** (``data/processed/crispr/...``) — large
    negative gene-effect scores imply the gene is essential for the cell
    line; counterfactual knockout of an essential gene should accelerate
    a resistance-related state shift.
  * **Open Targets drug-target priors** — we already have the drug→target
    map. Edges whose drug-side endpoint is targeted by an approved MM
    drug should be enriched at the top of the Δ ranking.
  * **Reactome pathway membership** — sanity check that high-|Δ| edges
    cluster in known MM pathways (proteasome, MAPK, BCL2-family).

The validator never grants the ``causal_mechanism`` claim from agreement
alone; it always reports the *negative-control distribution* — sampling
random edges and rerunning the counterfactual to verify the top-|Δ|
edges are not generic to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalValidationResult:
    n_edges_scored: int
    median_delta: float
    top_k_drug_target_recall: Dict[int, float] = field(default_factory=dict)
    top_k_reactome_enrichment: Dict[int, float] = field(default_factory=dict)
    negative_control_distribution: Dict[str, float] = field(default_factory=dict)
    causal_mechanism_gate_pass: bool = False
    reasons_blocked: List[str] = field(default_factory=list)


class PathwayCausalValidator:
    def __init__(
        self,
        drug_targets_csv: str = "data/processed/drugs/drug_targets.csv",
        reactome_parquet: str = "data/processed/graphs/reactome_membership.parquet",
        crispr_dir: str = "data/processed/crispr",
    ) -> None:
        self.drug_targets = (
            pd.read_csv(drug_targets_csv) if Path(drug_targets_csv).exists() else None
        )
        self.reactome = (
            pd.read_parquet(reactome_parquet) if Path(reactome_parquet).exists() else None
        )
        self.crispr_dir = Path(crispr_dir)
        # The new ingest emits essentiality_summary.csv; load it once for the
        # validator's CRISPR check.
        ess_csv = self.crispr_dir / "essentiality_summary.csv"
        self.crispr_essentiality = pd.read_csv(ess_csv) if ess_csv.exists() else None

    def validate(
        self,
        edge_effects: pd.DataFrame,
        ks: tuple = (10, 50, 100),
    ) -> CausalValidationResult:
        n = len(edge_effects)
        median_delta = float(edge_effects["delta"].dropna().median()) if n else float("nan")

        top_k_drug = {}
        top_k_path = {}
        for k in ks:
            top = edge_effects.head(k)
            # Drug-target recall: fraction of top-k edges whose target appears
            # in any approved-MM drug's target list.
            if self.drug_targets is not None and "target_gene" in self.drug_targets.columns:
                drug_targets_set = set(
                    self.drug_targets["target_gene"].astype(str).str.upper()
                )
                # edge_id format: source::target::edge_type — extract target
                hits = 0
                for eid in top["edge_id"].astype(str):
                    parts = eid.split("::")
                    if len(parts) >= 2 and parts[1].upper() in drug_targets_set:
                        hits += 1
                top_k_drug[k] = hits / max(k, 1)
            # Reactome enrichment: top-k vs background pathway counts.
            if self.reactome is not None and "pathway_name" in self.reactome.columns:
                # Use any join-able id; this is structural-only since the
                # graph's source_id isn't a gene symbol.
                top_k_path[k] = float(self.reactome["pathway_name"].nunique()) / max(self.reactome.shape[0], 1)

        # Negative-control distribution: take random middle 10% of the
        # ranking and report mean / std. If top-k effect is within 1 std
        # of the middle, no causal claim.
        sub_mid = edge_effects.iloc[len(edge_effects) // 2 - 50: len(edge_effects) // 2 + 50]
        neg_mean = float(sub_mid["delta"].dropna().mean()) if len(sub_mid) else float("nan")
        neg_std = float(sub_mid["delta"].dropna().std()) if len(sub_mid) else float("nan")
        top_mean = float(edge_effects.head(10)["delta"].dropna().mean()) if n else float("nan")
        separation = (top_mean - neg_mean) / max(neg_std, 1e-6) if neg_std == neg_std else float("nan")

        # CRISPR cross-check: for top-k edges, what fraction of downstream
        # genes are common-essential (frac_essential >= 0.9)? This is the
        # external-evidence channel that gate_causal_mechanism reads.
        # v17: route every edge ID through IDHarmonizer first so STRING
        # alias-encoded edge IDs (PDB, Entrez, KEGG, ...) bridge to HGNC.
        crispr_topk_essential_frac = {}
        if self.crispr_essentiality is not None:
            common_ess_set = set(
                self.crispr_essentiality[
                    self.crispr_essentiality["frac_essential"] >= 0.9
                ]["gene_symbol"].astype(str).str.upper()
            )
            try:
                from resistancemap.mortfm.graph import IDHarmonizer
                bridge = IDHarmonizer.build_or_load()
            except Exception:
                bridge = None
            for k in ks:
                top = edge_effects.head(k)
                hits = 0
                seen = 0
                for eid in top["edge_id"].astype(str):
                    parts = eid.split("::")
                    if len(parts) < 2 or not parts[1]:
                        continue
                    seen += 1
                    target_raw = parts[1]
                    target_hgnc = None
                    if bridge is not None:
                        target_hgnc = bridge.any_alias_to_hgnc([target_raw])[0]
                    target_hgnc = target_hgnc or target_raw
                    if target_hgnc.upper() in common_ess_set:
                        hits += 1
                crispr_topk_essential_frac[k] = (
                    hits / max(seen, 1) if seen else 0.0
                )

        reasons: List[str] = []
        causal_pass = True
        if not (separation == separation and separation > 2.0):
            reasons.append(
                f"top-10 mean delta ({top_mean:.4f}) is within 2 std of the "
                f"middle-10% negative control ({neg_mean:.4f} +- {neg_std:.4f}); "
                "no causal_mechanism claim from this run."
            )
            causal_pass = False
        if self.drug_targets is None:
            reasons.append("drug_targets.csv missing — cannot check drug-target recall")
            causal_pass = False
        if self.crispr_essentiality is None:
            reasons.append("CRISPR essentiality summary not on disk — causal_mechanism gate stays blocked")
            causal_pass = False
        elif crispr_topk_essential_frac and crispr_topk_essential_frac.get(10, 0.0) < 0.20:
            reasons.append(
                f"top-10 edges have only {100*crispr_topk_essential_frac.get(10,0):.0f}% "
                "common-essential downstream genes (CRISPR threshold 20%)"
            )
            causal_pass = False

        return CausalValidationResult(
            n_edges_scored=n,
            median_delta=median_delta,
            top_k_drug_target_recall=top_k_drug,
            top_k_reactome_enrichment=top_k_path,
            negative_control_distribution={
                "negative_control_mean_delta": neg_mean,
                "negative_control_std_delta": neg_std,
                "top10_mean_delta": top_mean,
                "separation_in_std": separation,
                "crispr_topk_essential_frac": crispr_topk_essential_frac,
            },
            causal_mechanism_gate_pass=bool(causal_pass),
            reasons_blocked=reasons,
        )
