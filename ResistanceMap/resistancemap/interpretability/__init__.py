"""
ResistanceMap v6 -- Mechanistic Interpretability Engine
======================================================

A comprehensive interpretability suite for the ResistanceMap Neural ODE model
of drug resistance in multiple myeloma. Designed for ICML/ICLR-quality
transparency: pathway attributions, counterfactual trajectories, temporal
explanations, and uncertainty-aware predictions.

Modules:
    pathway_attribution: Temporal Integrated Gradients and gene interaction recovery.
    counterfactual_engine: Counterfactual trajectory generation and causal effects.
    temporal_explanations: Critical windows, trajectory decomposition, phase transitions.
    uncertainty_explanations: Uncertainty decomposition, conformal calibration.
    visualization: Publication-quality figures (Waddington landscape, Circos, heatmaps).

Quick start::

    from resistancemap.interpretability import explain_patient

    explanation = explain_patient(model, patient_data, config)
    print(explanation["significant_pathways"])
    print(explanation["critical_window"])
    explanation["figures"]["patient_journey"].savefig("patient_journey.pdf")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

from .pathway_attribution import (
    PathwayAttributionConfig,
    PathwayAttributionResult,
    PathwayAttributor,
    GeneInteractionExtractor,
    InteractionResult,
    load_pathway_database,
)

from .counterfactual_engine import (
    CounterfactualConfig,
    CounterfactualResult,
    CounterfactualTrajectoryGenerator,
    WhatIfAnalyzer,
    CausalEffectEstimator,
    CausalEffectResult,
)

from .temporal_explanations import (
    TemporalConfig,
    CriticalWindowDetector,
    CriticalWindowResult,
    TrajectoryDecomposer,
    TrajectoryDecomposition,
    PhaseTransitionAnalyzer,
    PhaseTransitionResult,
)

from .uncertainty_explanations import (
    UncertaintyConfig,
    UncertaintyDecomposer,
    UncertaintyDecomposition,
    ConfidenceCalibrator,
    CalibrationResult,
    UncertaintyVisualizer,
)

from .visualization import (
    WaddingtonLandscapePlot,
    PathwayCircosPlot,
    CounterfactualTrajectoryPlot,
    TemporalAttributionHeatmap,
    PatientJourneyPlot,
    apply_nature_style,
)

__all__ = [
    # pathway_attribution
    "PathwayAttributionConfig",
    "PathwayAttributionResult",
    "PathwayAttributor",
    "GeneInteractionExtractor",
    "InteractionResult",
    "load_pathway_database",
    # counterfactual_engine
    "CounterfactualConfig",
    "CounterfactualResult",
    "CounterfactualTrajectoryGenerator",
    "WhatIfAnalyzer",
    "CausalEffectEstimator",
    "CausalEffectResult",
    # temporal_explanations
    "TemporalConfig",
    "CriticalWindowDetector",
    "CriticalWindowResult",
    "TrajectoryDecomposer",
    "TrajectoryDecomposition",
    "PhaseTransitionAnalyzer",
    "PhaseTransitionResult",
    # uncertainty_explanations
    "UncertaintyConfig",
    "UncertaintyDecomposer",
    "UncertaintyDecomposition",
    "ConfidenceCalibrator",
    "CalibrationResult",
    "UncertaintyVisualizer",
    # visualization
    "WaddingtonLandscapePlot",
    "PathwayCircosPlot",
    "CounterfactualTrajectoryPlot",
    "TemporalAttributionHeatmap",
    "PatientJourneyPlot",
    "apply_nature_style",
    # convenience
    "explain_patient",
]


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def explain_patient(
    model: torch.nn.Module,
    patient_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full interpretability pipeline for a single patient.

    This is the main entry point for generating a complete mechanistic
    explanation of a drug resistance prediction. It runs:
    1. Pathway attribution (Temporal Integrated Gradients)
    2. Counterfactual trajectory generation
    3. Temporal critical window detection
    4. Trajectory decomposition into biological programs
    5. Phase transition analysis
    6. Uncertainty decomposition
    7. Publication-quality figure generation

    Args:
        model: Trained ResistanceMap Neural ODE model. Must expose:
            - model.ode_func(t, x): ODE right-hand side
            - model.solve(x0, t_eval): ODE integrator
            - model.predict(x): resistance score from final state
            - model.classifier (optional): linear classification head
        patient_data: Dict containing:
            - 'x0': Tensor of initial state (n_genes,) or (1, n_genes)
            - 't_span': Tensor of evaluation timepoints
            - 'gene_names': List[str] of gene names
            - 'program_names': List[str] of biological program names
            - 'pathway_db': Dict[str, List[str]] pathway -> gene members
            - 'drug_history': (optional) List of drug intervention dicts
            - 'data_mask': (optional) Tensor of observed-feature indicators
        config: Optional dict of config overrides. Keys:
            - 'pathway': PathwayAttributionConfig overrides
            - 'counterfactual': CounterfactualConfig overrides
            - 'temporal': TemporalConfig overrides
            - 'uncertainty': UncertaintyConfig overrides

    Returns:
        Dict with keys:
            - 'pathway_attribution': PathwayAttributionResult
            - 'significant_pathways': List[str] significant pathway names
            - 'counterfactual': CounterfactualResult (if drug_history provided)
            - 'critical_window': CriticalWindowResult
            - 'trajectory_decomposition': TrajectoryDecomposition
            - 'phase_transition': PhaseTransitionResult
            - 'uncertainty': UncertaintyDecomposition
            - 'figures': Dict of matplotlib Figure objects
            - 'summary': Human-readable summary string
    """
    config = config or {}

    # Extract patient data
    x0 = patient_data["x0"]
    t_span = patient_data["t_span"]
    gene_names = patient_data.get("gene_names", [f"gene_{i}" for i in range(x0.shape[-1])])
    program_names = patient_data.get(
        "program_names",
        ["stemness", "drug_efflux", "dna_damage", "immune_evasion", "metabolic"],
    )
    pathway_db = patient_data.get("pathway_db", load_pathway_database("hallmark"))
    drug_history = patient_data.get("drug_history", None)
    data_mask = patient_data.get("data_mask", None)

    result: Dict[str, Any] = {}

    # ---- 1. Pathway Attribution ----
    pw_config = PathwayAttributionConfig(**config.get("pathway", {}))
    attributor = PathwayAttributor(model, gene_names, pathway_db, pw_config)
    pw_result = attributor.attribute(x0, t_span)
    result["pathway_attribution"] = pw_result
    result["significant_pathways"] = pw_result.significant_pathways

    # ---- 2. Critical Window Detection ----
    temp_config = TemporalConfig(**config.get("temporal", {}))
    cwd = CriticalWindowDetector(model, program_names, temp_config)
    cw_result = cwd.detect_critical_windows(x0, t_span)
    result["critical_window"] = cw_result

    # ---- 3. Trajectory Decomposition ----
    decomposer = TrajectoryDecomposer(model, program_names, config=temp_config)
    decomp = decomposer.decompose(x0, t_span)
    result["trajectory_decomposition"] = decomp

    # ---- 4. Phase Transition Analysis ----
    pta = PhaseTransitionAnalyzer(model, program_names, temp_config)
    pt_result = pta.analyze(x0, t_span)
    result["phase_transition"] = pt_result

    # ---- 5. Uncertainty Decomposition ----
    unc_config = UncertaintyConfig(**config.get("uncertainty", {}))
    unc_decomposer = UncertaintyDecomposer(model, config=unc_config)
    unc_result = unc_decomposer.decompose(
        x0, t_span,
        data_mask=data_mask,
        program_names=program_names,
    )
    result["uncertainty"] = unc_result

    # ---- 6. Counterfactual (if drug history provided) ----
    if drug_history is not None and len(drug_history) > 0:
        cf_config = CounterfactualConfig(**config.get("counterfactual", {}))
        cf_gen = CounterfactualTrajectoryGenerator(model, config=cf_config)

        # Simulate no-treatment counterfactual
        no_treatment = cf_gen.simulate_trajectory(x0, t_span)
        cf_result = CounterfactualResult(
            factual_trajectory=decomp.trajectory,
            counterfactual_trajectories={"No treatment": no_treatment.numpy()},
            factual_outcome=float(
                model.predict(torch.tensor(decomp.trajectory[-1:]).float()).mean()
                if hasattr(model, "predict")
                else np.linalg.norm(decomp.trajectory[-1])
            ),
            counterfactual_outcomes={
                "No treatment": float(
                    model.predict(no_treatment[-1:].float()).mean()
                    if hasattr(model, "predict")
                    else np.linalg.norm(no_treatment[-1].numpy())
                )
            },
            intervention_details={"drug_history": drug_history},
        )
        result["counterfactual"] = cf_result

    # ---- 7. Generate Figures ----
    figures = {}
    try:
        import matplotlib
        matplotlib.use("Agg")

        # Temporal attribution heatmap
        if pw_result.pathway_scores:
            pw_names = list(pw_result.pathway_scores.keys())
            attr_matrix = np.stack([pw_result.pathway_scores[n] for n in pw_names])
            figures["temporal_heatmap"] = TemporalAttributionHeatmap.plot(
                t_span.cpu().numpy() if isinstance(t_span, Tensor) else t_span,
                pw_names, attr_matrix,
            )

        # Patient journey
        figures["patient_journey"] = PatientJourneyPlot.plot(
            decomp.timepoints,
            decomp.trajectory,
            decomp.program_contributions,
            decomp.program_names,
            drug_history=drug_history,
            patient_id=patient_data.get("patient_id", "Patient"),
        )

    except Exception as e:
        result["figure_error"] = str(e)

    result["figures"] = figures

    # ---- 8. Summary ----
    summary_lines = [
        f"=== Interpretability Report ===",
        f"Significant pathways ({len(pw_result.significant_pathways)}):",
    ]
    for pw in pw_result.significant_pathways[:5]:
        summary_lines.append(f"  - {pw} (p={pw_result.pathway_pvalues.get(pw, 'N/A'):.4f})")

    if cw_result.critical_windows:
        w = cw_result.critical_windows[0]
        summary_lines.append(f"Primary critical window: t=[{w[0]:.1f}, {w[1]:.1f}]")

    if decomp.handoff_times:
        summary_lines.append(f"Program handoffs at: {[f'{t:.1f}' for t in decomp.handoff_times]}")

    if pt_result.bifurcation_times:
        summary_lines.append(
            f"Bifurcation at t={pt_result.bifurcation_times[0]:.1f}, "
            f"order parameter: {pt_result.order_parameter[1]}"
        )

    summary_lines.append(
        f"Total uncertainty: {unc_result.total_uncertainty:.4f} "
        f"(aleatoric: {unc_result.aleatoric:.4f}, "
        f"epistemic: {unc_result.epistemic:.4f})"
    )

    result["summary"] = "\n".join(summary_lines)

    return result
