"""Ablation framework.

An ablation is a deliberate, falsifiable removal/replacement of a single
architectural component, paired with a hypothesis about what it should
break and a metric to measure the breakage. This module:

1. Defines :class:`AblationSpec`, a frozen contract for one ablation.
2. Pre-populates the canonical ablations for ResistanceMap.
3. Provides :class:`AblationRunner`, which can either *describe* the
   experiment as a JSON manifest (no model required) or *run* it via a
   user-supplied model factory + bootstrap CIs on the metric delta.

The runner does NOT execute training itself and does NOT fabricate metric
values. Without a real model factory it raises a clear error rather than
return synthetic numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AblationSpec:
    """A pre-registered ablation experiment.

    Attributes:
        name: Short identifier.
        component: Architectural component being ablated.
        scientific_hypothesis: What we believe the component contributes.
        metric: Quantitative measurement used to detect the contribution.
        falsifier: Empirical condition under which the hypothesis is
            considered FALSIFIED (i.e. the ablation does NOT break what we
            said it would, so the component is NOT pulling the weight we
            attributed to it).
        expected_effect_direction: One of {"decrease", "increase"} -- which
            way the metric should move when the component is removed.
        expected_effect_magnitude: Minimum effect size the ablation must
            produce to be considered confirmatory.
    """

    name: str
    component: str
    scientific_hypothesis: str
    metric: str
    falsifier: str
    expected_effect_direction: str
    expected_effect_magnitude: float


# Canonical ResistanceMap ablations.
DEFAULT_ABLATIONS: list[AblationSpec] = [
    AblationSpec(
        name="ode_layer_removed",
        component="ODE trajectory layer",
        scientific_hypothesis=(
            "the ODE layer enforces temporal smoothness in the latent "
            "trajectory; removing it should make latent trajectories jagged "
            "and less Lipschitz"
        ),
        metric="trajectory_lipschitz_estimate",
        falsifier=(
            "ablated model achieves the same or smaller Lipschitz constant; "
            "this would mean the ODE layer was not actually enforcing "
            "smoothness and is functionally inert"
        ),
        expected_effect_direction="increase",
        expected_effect_magnitude=0.20,
    ),
    AblationSpec(
        name="gnn_replaced_with_mlp",
        component="GNN-on-PPI protein interaction module",
        scientific_hypothesis=(
            "the GNN encodes biological priors from STRING edges; replacing "
            "it with a permutation-invariant MLP destroys edge structure and "
            "should reduce stability of pathway attribution under random "
            "edge dropout"
        ),
        metric="edge_dropout_stability",
        falsifier=(
            "MLP variant matches GNN edge_dropout_stability; this means the "
            "PPI graph was not being used and the GNN was acting as a plain "
            "feature mixer"
        ),
        expected_effect_direction="decrease",
        expected_effect_magnitude=0.15,
    ),
    AblationSpec(
        name="vae_latent_dim_halved",
        component="VAE latent dimension",
        scientific_hypothesis=(
            "halving the latent dim reduces representation capacity and "
            "should worsen calibration of state predictions (ECE up)"
        ),
        metric="expected_calibration_error",
        falsifier=(
            "ECE does not change or improves when latent_dim is halved; "
            "this means the original latent dim was over-parameterized and "
            "the extra capacity was wasted"
        ),
        expected_effect_direction="increase",
        expected_effect_magnitude=0.05,
    ),
    AblationSpec(
        name="fusion_replaced_with_concat",
        component="Cross-attention multimodal fusion",
        scientific_hypothesis=(
            "cross-attention learns interaction terms between proteomics, "
            "epigenomics and scRNA modalities; replacing it with naive "
            "concatenation should reduce multimodal discriminability "
            "(macro-F1 across resistance phenotypes)"
        ),
        metric="macro_f1",
        falsifier=(
            "concatenation matches cross-attention macro-F1; this means "
            "the cross-attention was not learning useful cross-modal "
            "interactions"
        ),
        expected_effect_direction="decrease",
        expected_effect_magnitude=0.05,
    ),
]


@dataclass
class AblationResult:
    """Outcome of running one ablation."""

    spec: AblationSpec
    main_metric: Optional[float] = None
    ablated_metric: Optional[float] = None
    delta: Optional[float] = None
    bootstrap_ci: Optional[tuple[float, float]] = None
    confirmed: Optional[bool] = None
    falsified: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec"] = asdict(self.spec)
        d["bootstrap_ci"] = (
            list(self.bootstrap_ci) if self.bootstrap_ci is not None else None
        )
        return d


class AblationRunner:
    """Pre-registers, runs, and reports ablations.

    Usage:

        runner = AblationRunner()  # uses DEFAULT_ABLATIONS
        manifest = runner.manifest()  # JSON-serializable description

        # To actually run, supply a model factory:
        results = runner.run(
            model_factory=my_factory,
            evaluator=my_evaluator,
        )
    """

    def __init__(
        self,
        specs: Optional[list[AblationSpec]] = None,
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05,
    ) -> None:
        self.specs = specs if specs is not None else list(DEFAULT_ABLATIONS)
        self.n_bootstrap = int(n_bootstrap)
        self.ci_alpha = float(ci_alpha)

    # ------------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest describing the ablation suite.

        The manifest schema:

        {
          "framework_version": str,
          "n_ablations": int,
          "n_bootstrap": int,
          "ci_alpha": float,
          "ablations": [
            {
              "name": str,
              "component": str,
              "scientific_hypothesis": str,
              "metric": str,
              "falsifier": str,
              "expected_effect_direction": "increase" | "decrease",
              "expected_effect_magnitude": float,
              "pass_criterion": str,
              "fail_criterion": str
            },
            ...
          ]
        }
        """
        ablations = []
        for s in self.specs:
            ablations.append(
                {
                    **asdict(s),
                    "pass_criterion": (
                        f"metric '{s.metric}' moves in direction "
                        f"'{s.expected_effect_direction}' by at least "
                        f"{s.expected_effect_magnitude:.3f} with bootstrap "
                        f"{int((1 - self.ci_alpha) * 100)}% CI excluding 0"
                    ),
                    "fail_criterion": s.falsifier,
                }
            )
        return {
            "framework_version": "1.0",
            "n_ablations": len(self.specs),
            "n_bootstrap": self.n_bootstrap,
            "ci_alpha": self.ci_alpha,
            "ablations": ablations,
        }

    def write_manifest(self, path: Path) -> Path:
        """Write the manifest to disk and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.manifest(), f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Execution.
    # ------------------------------------------------------------------
    def run(
        self,
        model_factory: Optional[
            Callable[[AblationSpec, str], Any]
        ] = None,
        evaluator: Optional[
            Callable[[Any, AblationSpec], np.ndarray]
        ] = None,
    ) -> list[AblationResult]:
        """Execute every ablation.

        Args:
            model_factory: callable ``(spec, variant) -> trained_model``.
                ``variant`` is either ``"main"`` or ``"ablated"``. The
                factory is responsible for using REAL data; the runner
                does not pass training data and does not fabricate any.
            evaluator: callable ``(trained_model, spec) -> np.ndarray`` of
                per-example metric values (so the runner can bootstrap).

        Raises:
            RuntimeError: if model_factory or evaluator is None. The runner
                refuses to invent metric values; it only reports real runs.
        """
        if model_factory is None or evaluator is None:
            raise RuntimeError(
                "AblationRunner.run() requires a real model_factory and "
                "evaluator that operate on REAL data; the runner refuses "
                "to fabricate ablation results. Use AblationRunner.manifest() "
                "if you only want the experiment description."
            )

        rng = np.random.default_rng(0)
        out: list[AblationResult] = []
        for spec in self.specs:
            try:
                main_model = model_factory(spec, "main")
                ablated_model = model_factory(spec, "ablated")
                main_vals = np.asarray(evaluator(main_model, spec)).ravel()
                ablated_vals = np.asarray(evaluator(ablated_model, spec)).ravel()
                if main_vals.size == 0 or ablated_vals.size == 0:
                    raise ValueError("evaluator returned empty array")
                main_metric = float(np.mean(main_vals))
                ablated_metric = float(np.mean(ablated_vals))
                delta = ablated_metric - main_metric

                # Paired bootstrap on delta.
                n = min(main_vals.size, ablated_vals.size)
                deltas = np.empty(self.n_bootstrap, dtype=float)
                for b in range(self.n_bootstrap):
                    idx = rng.integers(0, n, size=n)
                    deltas[b] = float(
                        np.mean(ablated_vals[idx]) - np.mean(main_vals[idx])
                    )
                lo = float(np.quantile(deltas, self.ci_alpha / 2))
                hi = float(np.quantile(deltas, 1 - self.ci_alpha / 2))

                expected_sign = (
                    1.0 if spec.expected_effect_direction == "increase" else -1.0
                )
                magnitude = abs(delta)
                directional = (delta * expected_sign) > 0
                excludes_zero = lo > 0 or hi < 0
                confirmed = bool(
                    directional
                    and magnitude >= spec.expected_effect_magnitude
                    and excludes_zero
                )
                falsified = bool(
                    not directional
                    or magnitude < spec.expected_effect_magnitude / 2
                )

                out.append(
                    AblationResult(
                        spec=spec,
                        main_metric=main_metric,
                        ablated_metric=ablated_metric,
                        delta=delta,
                        bootstrap_ci=(lo, hi),
                        confirmed=confirmed,
                        falsified=falsified,
                        notes=(
                            "real run; bootstrap CI from per-example values"
                        ),
                    )
                )
            except Exception as exc:
                logger.exception("Ablation %s failed", spec.name)
                out.append(
                    AblationResult(
                        spec=spec,
                        notes=f"run failed: {exc}",
                    )
                )
        return out
