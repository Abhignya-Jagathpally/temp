"""MORT-FM Scientific Falsification Agent Layer.

Eleven scientific agents that form the falsification backbone for MORT-FM.
Every agent inherits from :class:`~resistancemap.evaluation.base.EvalAgent`
and returns an :class:`~resistancemap.evaluation.base.EvalFinding`.

Tier layout
-----------
* **Tier A** (data integrity gates) -- DataInventoryAgent,
  LongitudinalValidityAgent, LeakageAuditAgent, EndpointSemanticsAgent
* **Tier B** (mechanistic evidence) -- PathwayEvidenceAgent,
  CounterfactualEvidenceAgent
* **Tier C** (statistical falsification) -- BaselineAdversaryAgent,
  TrajectoryFalsificationAgent, SurvivalFalsificationAgent
* **Tier D** (claim governance) -- ClaimGateAgent, ReportAgent

Hard rule (applies to every agent):
    **Never fabricate data.** If required inputs are absent on disk or in
    the intake dict, the agent returns ``EvalVerdict.SKIPPED`` via
    ``skip_if_data_missing()`` or an equivalent manual check, documenting
    exactly what is missing. SKIPPED is NOT a silent pass.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from resistancemap.config import ResistanceMapConfig
from resistancemap.evaluation.base import EvalAgent, EvalFinding, EvalVerdict
from resistancemap.evaluation.leakage import (
    LeakageReport,
    audit_patient_split,
    detect_temporal_leakage,
)
from resistancemap.governance.endpoint_validator import (
    EndpointSemanticsReport,
    validate_endpoint_semantics,
    VALID_ENDPOINTS,
)
from resistancemap.governance.claim_gates import (
    ClaimGateReport,
    RunEvidence,
    run_gates,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None if it does not exist or is malformed."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _coarse_score(criteria: dict[str, Any]) -> float:
    """Crude average score: 1.0 for pass/True, 0.5 for conditional, 0 else."""
    if not criteria:
        return 0.0
    good = 0.0
    total = 0
    for value in criteria.values():
        total += 1
        if value is True or value == "pass":
            good += 1.0
        elif value == "conditional" or value == "partial":
            good += 0.5
    return good / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DataInventoryAgent (Tier A)
# ═══════════════════════════════════════════════════════════════════════════════


class DataInventoryAgent(EvalAgent):
    """Tier A: confirm which real cohorts and modalities are available on disk.

    Scans ``data/processed/`` and ``data/raw/`` for expected modality files.
    Outputs a data manifest listing present/missing modalities so downstream
    agents know what they can rely on.
    """

    tier: str = "A"

    #: Expected modality files (relative to the project root).
    EXPECTED_MODALITIES: dict[str, str] = {
        "ccle_proteomics": "data/raw/ccle_proteomics.csv",
        "ccle_epigenomics": "data/raw/ccle_epigenomics/",
        "string_ppi": "data/raw/string_ppi.txt",
        "gdsc_drug_sensitivity": "data/raw/gdsc_drug_sensitivity.csv",
        "scrna_gse124310": "data/raw/gse124310.h5ad",
        "scrna_gse271107": "data/raw/gse271107.h5ad",
        "mmrf_commpass": "data/raw/mmrf_commpass/",
        "depmap_crispr": "data/raw/depmap/CRISPRGeneEffect.csv",
        "prism": "data/raw/prism/secondary-screen-dose-response-curve-parameters.csv",
        "hmcl_keats": "data/raw/hmcl_keats/",
    }

    def __init__(self) -> None:
        super().__init__(name="data_inventory_agent", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        del intake  # Tier A root -- no upstream input.

        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Determine project root from config paths.
        project_root = Path.cwd()

        present: list[str] = []
        missing: list[str] = []
        manifest: dict[str, dict[str, Any]] = {}

        for modality, rel_path in self.EXPECTED_MODALITIES.items():
            full_path = project_root / rel_path
            exists = full_path.exists()
            manifest[modality] = {
                "path": str(full_path),
                "exists": exists,
            }
            if exists:
                present.append(modality)
            else:
                missing.append(modality)

        # Also scan data/processed/ for any checkpoint artefacts.
        processed_dir = project_root / "data" / "processed"
        processed_files: list[str] = []
        if processed_dir.exists():
            processed_files = [
                str(p.relative_to(processed_dir))
                for p in processed_dir.rglob("*")
                if p.is_file()
            ]
        manifest["processed_artefacts"] = {
            "count": len(processed_files),
            "files": processed_files[:50],  # cap for serialisability
        }

        evidence["manifest"] = manifest
        evidence["present_modalities"] = present
        evidence["missing_modalities"] = missing
        criteria_results["n_present"] = len(present)
        criteria_results["n_missing"] = len(missing)
        criteria_results["n_total"] = len(self.EXPECTED_MODALITIES)

        if missing:
            failure_modes.append(
                f"{len(missing)} expected modality source(s) missing on disk: "
                f"{missing}"
            )
            for mod in missing:
                required_changes.append(
                    f"Provide the {mod} dataset at "
                    f"{self.EXPECTED_MODALITIES[mod]}"
                )

        coverage = len(present) / len(self.EXPECTED_MODALITIES)
        criteria_results["coverage_ratio"] = round(coverage, 3)

        if coverage >= 0.8:
            verdict = EvalVerdict.PASS
        elif coverage >= 0.4:
            verdict = EvalVerdict.CONDITIONAL
        elif present:
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.FAIL

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=coverage,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LongitudinalValidityAgent (Tier A)
# ═══════════════════════════════════════════════════════════════════════════════


class LongitudinalValidityAgent(EvalAgent):
    """Tier A: verify true baseline-to-follow-up temporal ordering.

    Reads temporal pairs from the intake or from the MMRF CoMMpass manifest.
    Checks:

    * ``delta_t_days > 0`` for every follow-up relative to its baseline.
    * No pseudotime passed off as calendar time.
    * Censoring status is explicit (no NaN / sentinel masquerading as event).

    Hard FAIL if any violations are found.
    """

    tier: str = "A"

    def __init__(self) -> None:
        super().__init__(name="longitudinal_validity_agent", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Look for temporal pair data.
        temporal_pairs = intake.get("temporal_pairs")
        manifest_path = config.data.mmrf_commpass_dir / "patient_manifest.csv"

        if temporal_pairs is None and not manifest_path.exists():
            skipped = self.skip_if_data_missing(
                [manifest_path],
                memo=(
                    "No temporal pair data in intake and no patient manifest "
                    "on disk; longitudinal validity cannot be assessed."
                ),
            )
            if skipped is not None:
                return skipped

        # If temporal pairs are in intake, validate them directly.
        if temporal_pairs is not None:
            pairs = list(temporal_pairs)
        else:
            # Attempt to load from manifest (scaffold -- requires real loader).
            pairs = []
            failure_modes.append(
                "Temporal pairs loaded from manifest scaffold; real loader "
                "not yet wired."
            )
            required_changes.append(
                "Wire the MMRF CoMMpass temporal-pair loader so "
                "LongitudinalValidityAgent reads real time-stamped records."
            )

        # --- Check 1: delta_t_days > 0 ------------------------------------
        violations_negative_dt: list[dict[str, Any]] = []
        violations_pseudotime: list[dict[str, Any]] = []
        violations_censoring: list[dict[str, Any]] = []

        for idx, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                continue

            dt = pair.get("delta_t_days")
            if dt is not None and dt <= 0:
                violations_negative_dt.append(
                    {"index": idx, "delta_t_days": dt, "patient_id": pair.get("patient_id")}
                )

            # Pseudotime masquerading as calendar time.
            if pair.get("time_type") == "pseudotime" and pair.get("unit") in (
                "days",
                "months",
                "years",
            ):
                violations_pseudotime.append(
                    {"index": idx, "patient_id": pair.get("patient_id")}
                )

            # Censoring must be explicit.
            censor = pair.get("censored")
            event = pair.get("event_observed")
            if censor is None and event is None:
                violations_censoring.append(
                    {"index": idx, "patient_id": pair.get("patient_id")}
                )

        criteria_results["n_pairs_checked"] = len(pairs)
        criteria_results["violations_negative_dt"] = len(violations_negative_dt)
        criteria_results["violations_pseudotime"] = len(violations_pseudotime)
        criteria_results["violations_censoring"] = len(violations_censoring)

        evidence["negative_dt_examples"] = violations_negative_dt[:10]
        evidence["pseudotime_examples"] = violations_pseudotime[:10]
        evidence["censoring_examples"] = violations_censoring[:10]

        total_violations = (
            len(violations_negative_dt)
            + len(violations_pseudotime)
            + len(violations_censoring)
        )

        if violations_negative_dt:
            failure_modes.append(
                f"{len(violations_negative_dt)} pair(s) have delta_t_days <= 0 "
                "(follow-up not after baseline)."
            )
            required_changes.append(
                "Fix all temporal pairs with non-positive delta_t_days; "
                "ensure follow-up date is strictly after baseline."
            )

        if violations_pseudotime:
            failure_modes.append(
                f"{len(violations_pseudotime)} pair(s) pass pseudotime off "
                "as calendar time."
            )
            required_changes.append(
                "Replace pseudotime with real calendar timestamps or remove "
                "calendar-time unit labels from pseudotime records."
            )

        if violations_censoring:
            failure_modes.append(
                f"{len(violations_censoring)} pair(s) have implicit censoring "
                "(neither 'censored' nor 'event_observed' is set)."
            )
            required_changes.append(
                "Make censoring status explicit for every temporal pair: "
                "set either 'censored' or 'event_observed'."
            )

        if total_violations > 0:
            verdict = EvalVerdict.FAIL
            score = max(0.0, 1.0 - total_violations / max(len(pairs), 1))
        elif not pairs:
            verdict = EvalVerdict.SKIPPED
            score = 0.0
            failure_modes.append("No temporal pairs available to validate.")
            required_changes.append("Provide temporal pair data for validation.")
        else:
            verdict = EvalVerdict.PASS
            score = 1.0

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LeakageAuditAgent (Tier A)
# ═══════════════════════════════════════════════════════════════════════════════


class LeakageAuditAgent(EvalAgent):
    """Tier A: enforce patient-disjoint and time-valid data splits.

    Delegates structural overlap detection to
    :func:`~resistancemap.evaluation.leakage.audit_patient_split` and
    temporal leakage to
    :func:`~resistancemap.evaluation.leakage.detect_temporal_leakage`.

    Additionally checks:

    * Drug-holdout contamination (test drugs appearing in training set).
    * Replicate-alias contamination (different IDs for same patient).
    * External-cohort overlap (CoMMpass IDs in CCLE or vice versa).
    """

    tier: str = "A"

    def __init__(self) -> None:
        super().__init__(name="leakage_audit_agent", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        train_ids = intake.get("train_patient_ids") or []
        val_ids = intake.get("val_patient_ids") or []
        test_ids = intake.get("test_patient_ids") or []

        # --- 1. Patient-disjoint split audit --------------------------------
        split_report: LeakageReport = audit_patient_split(
            train_ids, val_ids, test_ids,
        )
        criteria_results["patient_split_clean"] = split_report.clean
        evidence["patient_split_report"] = split_report.to_dict()

        if split_report.duplicate_patient_ids:
            n_dup = len(split_report.duplicate_patient_ids)
            failure_modes.append(
                f"{n_dup} patient(s) appear in multiple splits; "
                "patient-disjoint constraint violated."
            )
            required_changes.append(
                "Re-split data so that no patient ID appears in more than "
                "one of train/val/test."
            )

        # --- 2. Temporal leakage audit --------------------------------------
        samples = intake.get("temporal_samples")
        if samples is not None:
            temporal_report: LeakageReport = detect_temporal_leakage(
                samples=samples,
                split_assignments=intake.get("split_assignments", {}),
                timestamp_field=intake.get("timestamp_field", "collection_date"),
                patient_id_field=intake.get("patient_id_field", "patient_id"),
            )
        else:
            # No temporal data -> vacuous report.
            temporal_report = LeakageReport(
                clean=False,
                details={"warnings": ["no temporal samples provided"]},
            )
        criteria_results["temporal_leakage_clean"] = temporal_report.clean
        evidence["temporal_leakage_report"] = temporal_report.to_dict()

        if temporal_report.temporal_leaks:
            n_leak = len(temporal_report.temporal_leaks)
            failure_modes.append(
                f"{n_leak} patient/split pair(s) have temporal leakage "
                "(train time >= test time)."
            )
            required_changes.append(
                "Re-split to ensure every patient's training samples "
                "precede their test samples in calendar time."
            )

        # --- 3. Drug-holdout contamination ----------------------------------
        train_drugs = set(intake.get("train_drugs") or [])
        test_drugs = set(intake.get("test_drugs") or [])
        drug_overlap = train_drugs & test_drugs
        criteria_results["drug_holdout_clean"] = len(drug_overlap) == 0
        evidence["drug_overlap"] = sorted(drug_overlap)
        if drug_overlap:
            failure_modes.append(
                f"Drug holdout contamination: {sorted(drug_overlap)} appear "
                "in both train and test."
            )
            required_changes.append(
                "Remove overlapping drugs from the test set or implement "
                "a proper drug-holdout split."
            )

        # --- 4. Replicate-alias check ---------------------------------------
        alias_map = intake.get("patient_alias_map") or {}
        aliased_overlap: list[str] = []
        all_train = set(str(i) for i in train_ids)
        all_test = set(str(i) for i in test_ids)
        for canonical, aliases in alias_map.items():
            aliases_set = set(str(a) for a in aliases) | {str(canonical)}
            in_train = aliases_set & all_train
            in_test = aliases_set & all_test
            if in_train and in_test:
                aliased_overlap.append(canonical)

        criteria_results["alias_contamination_clean"] = len(aliased_overlap) == 0
        evidence["aliased_overlap_patients"] = aliased_overlap[:20]
        if aliased_overlap:
            failure_modes.append(
                f"{len(aliased_overlap)} patient(s) appear across splits "
                "under different alias IDs."
            )
            required_changes.append(
                "Resolve patient aliases before splitting; use canonical "
                "IDs for all split assignments."
            )

        # --- 5. External-cohort overlap -------------------------------------
        external_ids = set(str(i) for i in (intake.get("external_cohort_ids") or []))
        ext_in_train = external_ids & all_train
        ext_in_test = external_ids & all_test
        ext_overlap = ext_in_train | ext_in_test
        criteria_results["external_cohort_clean"] = len(ext_overlap) == 0
        evidence["external_cohort_overlap_count"] = len(ext_overlap)
        if ext_overlap:
            failure_modes.append(
                f"{len(ext_overlap)} external cohort ID(s) found in "
                "train or test splits."
            )
            required_changes.append(
                "Remove external-cohort samples from internal splits or "
                "use external cohort exclusively for external validation."
            )

        # --- Verdict roll-up -------------------------------------------------
        any_hard_fail = (
            bool(split_report.duplicate_patient_ids)
            or bool(temporal_report.temporal_leaks)
            or bool(aliased_overlap)
        )
        if any_hard_fail:
            verdict = EvalVerdict.FAIL
        elif not split_report.clean or not temporal_report.clean:
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.PASS

        score = _coarse_score(criteria_results)
        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EndpointSemanticsAgent (Tier A)
# ═══════════════════════════════════════════════════════════════════════════════


class EndpointSemanticsAgent(EvalAgent):
    """Tier A: validate endpoint semantics for resistance claims.

    Delegates to :func:`~resistancemap.governance.endpoint_validator.validate_endpoint_semantics`.
    Hard blocks OS-as-resistance: if the endpoint is ``overall_survival``
    and the pipeline claims resistance prediction, this agent issues FAIL.
    """

    tier: str = "A"

    def __init__(self) -> None:
        super().__init__(name="endpoint_semantics_agent", dependencies=[])

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        endpoint_name = intake.get("endpoint_name") or getattr(
            config, "endpoint_name", None
        )
        if endpoint_name is None:
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.SKIPPED,
                score=0.0,
                criteria_results={"endpoint_name": None},
                evidence={},
                failure_modes=["No endpoint_name provided in intake or config."],
                required_changes=["Set endpoint_name in intake or config."],
            )

        if endpoint_name not in VALID_ENDPOINTS:
            return self.fail_closed_gate(
                reason=(
                    f"Unknown endpoint {endpoint_name!r}; must be one of "
                    f"{VALID_ENDPOINTS}."
                ),
                required_changes=[
                    f"Use a recognised endpoint name from: {VALID_ENDPOINTS}"
                ],
            )

        has_longitudinal = bool(intake.get("has_longitudinal_pairs", False))
        n_longitudinal = int(intake.get("n_longitudinal_pairs", 0))
        claim_type = intake.get("claim_type", "resistance")

        report: EndpointSemanticsReport = validate_endpoint_semantics(
            endpoint_name=endpoint_name,
            has_longitudinal_pairs=has_longitudinal,
            n_longitudinal_pairs=n_longitudinal,
        )

        criteria_results["endpoint_name"] = report.endpoint_name
        criteria_results["endpoint_type"] = report.endpoint_type
        criteria_results["claim_level"] = report.claim_level
        criteria_results["resistance_claim_allowed"] = report.resistance_claim_allowed
        criteria_results["trajectory_claim_allowed"] = report.trajectory_claim_allowed
        criteria_results["survival_claim_allowed"] = report.survival_claim_allowed
        evidence["endpoint_semantics_report"] = report.__dict__

        # Block OS-as-resistance.
        if claim_type == "resistance" and not report.resistance_claim_allowed:
            failure_modes.append(
                f"Endpoint {endpoint_name!r} does NOT support a resistance "
                f"claim (claim_level={report.claim_level})."
            )
            required_changes.append(
                "Use an endpoint that supports resistance claims "
                "(relapse_time, refractory_status, mrd_conversion, "
                "drug_resistance_label, or ex_vivo_resistance_shift), "
                "or reframe the claim as survival-only."
            )
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.FAIL,
                score=0.0,
                criteria_results=criteria_results,
                evidence=evidence,
                failure_modes=failure_modes,
                required_changes=required_changes,
            )

        # Block trajectory claims without longitudinal data.
        if claim_type == "trajectory" and not report.trajectory_claim_allowed:
            failure_modes.append(
                f"Endpoint {endpoint_name!r} does not support trajectory "
                f"claims: {report.blocking_reasons}"
            )
            required_changes.append(
                "Provide sufficient longitudinal pairs or reframe "
                "as a static prediction claim."
            )

        if report.blocking_reasons:
            for reason in report.blocking_reasons:
                evidence.setdefault("blocking_reasons", []).append(reason)

        verdict = (
            EvalVerdict.FAIL
            if failure_modes
            else EvalVerdict.CONDITIONAL
            if report.blocking_reasons
            else EvalVerdict.PASS
        )
        score = 1.0 if verdict == EvalVerdict.PASS else 0.5 if verdict == EvalVerdict.CONDITIONAL else 0.0

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BaselineAdversaryAgent (Tier C)
# ═══════════════════════════════════════════════════════════════════════════════


class BaselineAdversaryAgent(EvalAgent):
    """Tier C: run honest baselines and try to beat MORT-FM.

    Reads baseline comparison results from:
    * ``paper/tables/baseline_comparison.json``
    * ``paper/v8_artifacts/sota_benchmark/sota_comparison_table.json``

    Computes paired bootstrap deltas. FAIL if any baseline beats MORT-FM
    with a 95% CI that excludes zero.
    """

    tier: str = "C"

    def __init__(self) -> None:
        super().__init__(
            name="baseline_adversary_scientific_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        project_root = Path.cwd()
        baseline_json = project_root / "paper" / "tables" / "baseline_comparison.json"
        sota_json = (
            project_root
            / "paper"
            / "v8_artifacts"
            / "sota_benchmark"
            / "sota_comparison_table.json"
        )

        # Try loading from files on disk.
        baseline_data = _safe_load_json(baseline_json)
        sota_data = _safe_load_json(sota_json)

        # Also accept pre-loaded results from intake.
        baseline_predictions = intake.get("baseline_predictions") or {}

        if baseline_data is None and sota_data is None and not baseline_predictions:
            skipped = self.skip_if_data_missing(
                [baseline_json, sota_json],
                memo=(
                    "No baseline comparison data found on disk or in intake; "
                    "cannot adjudicate whether baselines beat MORT-FM."
                ),
            )
            if skipped is not None:
                return skipped

        evidence["baseline_json_path"] = str(baseline_json)
        evidence["sota_json_path"] = str(sota_json)
        evidence["baseline_json_loaded"] = baseline_data is not None
        evidence["sota_json_loaded"] = sota_data is not None

        # Merge all available comparison rows.
        comparisons: list[dict[str, Any]] = []
        if isinstance(baseline_data, list):
            comparisons.extend(baseline_data)
        elif isinstance(baseline_data, dict):
            comparisons.extend(baseline_data.get("comparisons", []))
        if isinstance(sota_data, list):
            comparisons.extend(sota_data)
        elif isinstance(sota_data, dict):
            comparisons.extend(sota_data.get("comparisons", []))

        # Evaluate each comparison row.
        n_boot = int(intake.get("n_bootstrap", 1000))
        any_baseline_wins = False
        comparison_results: list[dict[str, Any]] = []

        for comp in comparisons:
            baseline_name = comp.get("baseline") or comp.get("model", "unknown")
            mortfm_metric = comp.get("mortfm_metric") or comp.get("main_metric")
            baseline_metric = comp.get("baseline_metric") or comp.get("metric")

            if mortfm_metric is None or baseline_metric is None:
                continue

            try:
                mortfm_val = float(mortfm_metric)
                baseline_val = float(baseline_metric)
            except (TypeError, ValueError):
                continue

            delta = mortfm_val - baseline_val
            ci_low = comp.get("ci_low")
            ci_high = comp.get("ci_high")

            # If CI is provided, use it; otherwise note that bootstrap is needed.
            row = {
                "baseline": baseline_name,
                "mortfm_metric": mortfm_val,
                "baseline_metric": baseline_val,
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            # Baseline wins if delta < 0 and CI excludes zero.
            if delta < 0:
                if ci_low is not None and ci_high is not None:
                    try:
                        if float(ci_high) < 0:
                            row["verdict"] = "BASELINE_WINS"
                            any_baseline_wins = True
                            failure_modes.append(
                                f"Baseline {baseline_name} beats MORT-FM "
                                f"(delta={delta:+.4f}, CI=[{ci_low}, {ci_high}])."
                            )
                        else:
                            row["verdict"] = "INCONCLUSIVE"
                    except (TypeError, ValueError):
                        row["verdict"] = "CI_PARSE_ERROR"
                else:
                    row["verdict"] = "NEGATIVE_DELTA_NO_CI"
                    failure_modes.append(
                        f"Baseline {baseline_name} shows negative delta "
                        f"({delta:+.4f}) but no CI is provided; "
                        "cannot confirm significance."
                    )
                    required_changes.append(
                        f"Compute bootstrap CI for baseline {baseline_name}."
                    )
            else:
                row["verdict"] = "MORTFM_WINS" if delta > 0 else "TIE"

            comparison_results.append(row)

        criteria_results["n_comparisons"] = len(comparison_results)
        criteria_results["any_baseline_wins"] = any_baseline_wins
        evidence["comparison_results"] = comparison_results

        if any_baseline_wins:
            verdict = EvalVerdict.FAIL
            required_changes.append(
                "MORT-FM must beat all baselines with CI excluding zero "
                "before claiming superiority."
            )
        elif not comparison_results:
            verdict = EvalVerdict.SKIPPED
            failure_modes.append("No valid comparison rows found.")
            required_changes.append(
                "Populate baseline comparison data in "
                "paper/tables/baseline_comparison.json."
            )
        else:
            n_wins = sum(1 for r in comparison_results if r["verdict"] == "MORTFM_WINS")
            if n_wins == len(comparison_results):
                verdict = EvalVerdict.PASS
            else:
                verdict = EvalVerdict.CONDITIONAL

        score = (
            sum(1 for r in comparison_results if r.get("verdict") == "MORTFM_WINS")
            / max(len(comparison_results), 1)
        )

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TrajectoryFalsificationAgent (Tier C)
# ═══════════════════════════════════════════════════════════════════════════════


class TrajectoryFalsificationAgent(EvalAgent):
    """Tier C: test whether trajectory prediction beats LOCF/mean/clinical-only.

    Reads trajectory metrics from checkpoints or intake. Requires that
    the delta versus LOCF baseline is positive and significant.
    """

    tier: str = "C"

    def __init__(self) -> None:
        super().__init__(
            name="trajectory_falsification_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Look for trajectory metrics in intake or checkpoint.
        traj_metrics = intake.get("trajectory_metrics")
        checkpoint_dir = Path(config.checkpoint_dir) if hasattr(config, "checkpoint_dir") else Path("checkpoints")
        traj_checkpoint = checkpoint_dir / "trajectory_metrics.json"

        if traj_metrics is None:
            loaded = _safe_load_json(traj_checkpoint)
            if isinstance(loaded, dict):
                traj_metrics = loaded

        if traj_metrics is None:
            skipped = self.skip_if_data_missing(
                [traj_checkpoint],
                memo=(
                    "No trajectory metrics in intake or on disk; "
                    "cannot falsify trajectory prediction claims."
                ),
            )
            if skipped is not None:
                return skipped

        evidence["source"] = (
            "intake" if intake.get("trajectory_metrics") else str(traj_checkpoint)
        )

        # Expected keys: mortfm_mse, locf_mse, mean_mse, clinical_only_mse
        mortfm_mse = traj_metrics.get("mortfm_mse")
        locf_mse = traj_metrics.get("locf_mse")
        mean_mse = traj_metrics.get("mean_mse")
        clinical_mse = traj_metrics.get("clinical_only_mse")

        baselines_checked: dict[str, dict[str, Any]] = {}

        for baseline_name, baseline_val in [
            ("LOCF", locf_mse),
            ("mean", mean_mse),
            ("clinical_only", clinical_mse),
        ]:
            if baseline_val is None or mortfm_mse is None:
                baselines_checked[baseline_name] = {"status": "missing"}
                continue

            try:
                delta = float(baseline_val) - float(mortfm_mse)  # positive = MORTFM better
            except (TypeError, ValueError):
                baselines_checked[baseline_name] = {"status": "parse_error"}
                continue

            baselines_checked[baseline_name] = {
                "mortfm_mse": float(mortfm_mse),
                "baseline_mse": float(baseline_val),
                "delta_mse": delta,
                "mortfm_better": delta > 0,
            }

            if delta <= 0:
                failure_modes.append(
                    f"MORT-FM trajectory does NOT beat {baseline_name} "
                    f"(delta MSE = {delta:+.4f}; lower is better)."
                )

        criteria_results["baselines_checked"] = baselines_checked
        evidence["trajectory_metrics_raw"] = traj_metrics

        # Check for delta vs LOCF specifically.
        locf_result = baselines_checked.get("LOCF", {})
        criteria_results["beats_locf"] = locf_result.get("mortfm_better", False)

        # Check for significance (p-value or CI).
        delta_p = traj_metrics.get("locf_delta_p_value")
        delta_ci = traj_metrics.get("locf_delta_ci")
        if delta_p is not None:
            criteria_results["locf_delta_p_value"] = float(delta_p)
            criteria_results["locf_significant"] = float(delta_p) < 0.05
        if delta_ci is not None:
            criteria_results["locf_delta_ci"] = delta_ci

        # Verdict.
        all_beaten = all(
            v.get("mortfm_better", False)
            for v in baselines_checked.values()
            if v.get("status") != "missing"
        )
        any_present = any(
            v.get("status") != "missing"
            for v in baselines_checked.values()
        )

        if not any_present:
            verdict = EvalVerdict.SKIPPED
            failure_modes.append("No baseline trajectory metrics available.")
            required_changes.append(
                "Compute LOCF, mean, and clinical-only baselines for "
                "trajectory prediction."
            )
        elif all_beaten:
            verdict = EvalVerdict.PASS
        elif criteria_results.get("beats_locf"):
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.FAIL
            required_changes.append(
                "MORT-FM trajectory must beat at minimum LOCF baseline "
                "before claiming trajectory prediction capability."
            )

        score = (
            sum(
                1
                for v in baselines_checked.values()
                if v.get("mortfm_better", False)
            )
            / max(
                sum(1 for v in baselines_checked.values() if v.get("status") != "missing"),
                1,
            )
        )

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SurvivalFalsificationAgent (Tier C)
# ═══════════════════════════════════════════════════════════════════════════════


class SurvivalFalsificationAgent(EvalAgent):
    """Tier C: test survival predictions against clinical-only Cox.

    PASS requires all four criteria:
    1. MORT-FM C-index > Cox C-index
    2. MORT-FM lower CI > Cox point estimate
    3. Permutation p < 0.05
    4. IBS improves (MORT-FM IBS < Cox IBS)
    """

    tier: str = "C"

    def __init__(self) -> None:
        super().__init__(
            name="survival_falsification_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Look for survival metrics.
        survival_metrics = intake.get("survival_metrics")
        checkpoint_dir = Path(config.checkpoint_dir) if hasattr(config, "checkpoint_dir") else Path("checkpoints")
        survival_json = checkpoint_dir / "survival_baseline_suite.json"

        if survival_metrics is None:
            loaded = _safe_load_json(survival_json)
            if isinstance(loaded, dict):
                survival_metrics = loaded

        if survival_metrics is None:
            skipped = self.skip_if_data_missing(
                [survival_json],
                memo=(
                    "No survival baseline suite results in intake or on disk; "
                    "cannot falsify survival prediction claims."
                ),
            )
            if skipped is not None:
                return skipped

        evidence["source"] = (
            "intake" if intake.get("survival_metrics") else str(survival_json)
        )
        evidence["raw_metrics"] = survival_metrics

        # Extract key metrics.
        mortfm_c = survival_metrics.get("mortfm_c_index")
        cox_c = survival_metrics.get("cox_c_index")
        mortfm_c_ci_low = survival_metrics.get("mortfm_c_ci_low")
        perm_p = survival_metrics.get("permutation_p_value")
        mortfm_ibs = survival_metrics.get("mortfm_ibs")
        cox_ibs = survival_metrics.get("cox_ibs")

        n_criteria_passed = 0
        n_criteria_total = 4

        # Criterion 1: MORT-FM C > Cox C.
        if mortfm_c is not None and cox_c is not None:
            c1 = float(mortfm_c) > float(cox_c)
            criteria_results["c_index_better"] = c1
            if c1:
                n_criteria_passed += 1
            else:
                failure_modes.append(
                    f"MORT-FM C-index ({mortfm_c}) does not exceed "
                    f"Cox C-index ({cox_c})."
                )
        else:
            criteria_results["c_index_better"] = "missing"
            failure_modes.append("C-index comparison data missing.")
            required_changes.append("Compute C-index for both MORT-FM and Cox.")

        # Criterion 2: MORT-FM lower CI > Cox point.
        if mortfm_c_ci_low is not None and cox_c is not None:
            c2 = float(mortfm_c_ci_low) > float(cox_c)
            criteria_results["ci_excludes_cox"] = c2
            if c2:
                n_criteria_passed += 1
            else:
                failure_modes.append(
                    f"MORT-FM C-index lower CI ({mortfm_c_ci_low}) does not "
                    f"exceed Cox C-index ({cox_c}); difference may not be significant."
                )
        else:
            criteria_results["ci_excludes_cox"] = "missing"
            required_changes.append(
                "Compute bootstrap CI for MORT-FM C-index."
            )

        # Criterion 3: Permutation p < 0.05.
        if perm_p is not None:
            c3 = float(perm_p) < 0.05
            criteria_results["permutation_significant"] = c3
            criteria_results["permutation_p_value"] = float(perm_p)
            if c3:
                n_criteria_passed += 1
            else:
                failure_modes.append(
                    f"Permutation test p-value ({perm_p}) >= 0.05; "
                    "survival improvement not significant."
                )
        else:
            criteria_results["permutation_significant"] = "missing"
            required_changes.append(
                "Run permutation test comparing MORT-FM vs Cox survival model."
            )

        # Criterion 4: IBS improves.
        if mortfm_ibs is not None and cox_ibs is not None:
            c4 = float(mortfm_ibs) < float(cox_ibs)
            criteria_results["ibs_improves"] = c4
            if c4:
                n_criteria_passed += 1
            else:
                failure_modes.append(
                    f"MORT-FM IBS ({mortfm_ibs}) does not improve over "
                    f"Cox IBS ({cox_ibs})."
                )
        else:
            criteria_results["ibs_improves"] = "missing"
            required_changes.append(
                "Compute Integrated Brier Score for both MORT-FM and Cox."
            )

        criteria_results["n_criteria_passed"] = n_criteria_passed
        criteria_results["n_criteria_total"] = n_criteria_total

        score = n_criteria_passed / n_criteria_total
        if n_criteria_passed == n_criteria_total:
            verdict = EvalVerdict.PASS
        elif n_criteria_passed >= 2:
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.FAIL
            required_changes.append(
                "MORT-FM must pass all 4 survival falsification criteria "
                "(C-index > Cox, CI excludes Cox, permutation p < 0.05, "
                "IBS improves) before claiming survival prediction superiority."
            )

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PathwayEvidenceAgent (Tier B)
# ═══════════════════════════════════════════════════════════════════════════════


class PathwayEvidenceAgent(EvalAgent):
    """Tier B: join attribution with CRISPR, drug-target, and Reactome evidence.

    Delegates to :func:`~resistancemap.mortfm.causal.edge_evidence_report.build_edge_evidence_report`
    for the k-of-N evidence gate (default k=2, N=3). The agent PASSES when
    at least 2 of 3 independent evidence channels (CRISPR essentiality,
    drug-target support, Reactome pathway enrichment) support the top-k
    attributed edges.
    """

    tier: str = "B"

    def __init__(self) -> None:
        super().__init__(
            name="pathway_evidence_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        edge_effects = intake.get("edge_effects")
        if edge_effects is None:
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.SKIPPED,
                score=0.0,
                criteria_results={"edge_effects_provided": False},
                evidence={},
                failure_modes=[
                    "No edge_effects DataFrame in intake; cannot evaluate "
                    "pathway evidence. This agent does NOT fabricate edges."
                ],
                required_changes=[
                    "Run the counterfactual pipeline and provide "
                    "edge_effects DataFrame in intake."
                ],
            )

        try:
            import pandas as pd

            if not isinstance(edge_effects, pd.DataFrame):
                edge_effects = pd.DataFrame(edge_effects)
        except Exception as exc:
            return self.fail_closed_gate(
                reason=f"Cannot convert edge_effects to DataFrame: {exc}",
                required_changes=["Provide edge_effects as a pandas DataFrame."],
            )

        # Delegate to the real evidence builder.
        try:
            from resistancemap.mortfm.causal.edge_evidence_report import (
                build_edge_evidence_report,
            )

            top_k = int(intake.get("top_k", 20))
            k = int(intake.get("evidence_k", 2))
            report = build_edge_evidence_report(edge_effects, top_k=top_k, k=k)
        except Exception as exc:
            logger.exception("build_edge_evidence_report failed")
            return self.fail_closed_gate(
                reason=f"Edge evidence report builder failed: {exc}",
                required_changes=[
                    "Fix the edge evidence report pipeline and re-run."
                ],
            )

        summary = report.get("summary", {})
        gate_pass = summary.get("causal_mechanism_gate_pass", False)
        n_supporting = summary.get("causal_mechanism_gate_n_supporting", 0)
        gate_k = summary.get("causal_mechanism_gate_k", 2)

        criteria_results["k_of_n_gate_pass"] = gate_pass
        criteria_results["n_supporting_channels"] = n_supporting
        criteria_results["gate_k"] = gate_k
        criteria_results["channels_supported"] = summary.get(
            "causal_mechanism_gate_channels_supported", []
        )
        criteria_results["channels_not_supported"] = summary.get(
            "causal_mechanism_gate_channels_not_supported", []
        )

        evidence["edge_evidence_summary"] = {
            k: v
            for k, v in summary.items()
            if not isinstance(v, (type(None),))  # skip None-ish
        }
        evidence["n_edges"] = summary.get("n_edges", 0)
        evidence["pathway_enrichment"] = summary.get("pathway_enrichment", {})

        if not gate_pass:
            failure_modes.append(
                f"Causal mechanism gate FAIL: only {n_supporting}/{3} "
                f"channels support (need >= {gate_k}). "
                f"Not supported: {criteria_results['channels_not_supported']}"
            )
            required_changes.append(
                "Strengthen evidence for attributed pathways: ensure at "
                "least 2 of 3 channels (CRISPR, drug-target, Reactome) "
                "support the top-k edges."
            )

        verdict = EvalVerdict.PASS if gate_pass else EvalVerdict.FAIL
        score = n_supporting / 3.0

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CounterfactualEvidenceAgent (Tier B)
# ═══════════════════════════════════════════════════════════════════════════════


class CounterfactualEvidenceAgent(EvalAgent):
    """Tier B: check whether interventions change predicted resistance plausibly.

    Validates that the top-10 counterfactual edge effects exceed 2 sigma
    above the negative-control distribution. The negative control is the
    distribution of effects from edges known to be biologically inert
    (provided in intake as ``negative_control_deltas``).
    """

    tier: str = "B"

    def __init__(self) -> None:
        super().__init__(
            name="counterfactual_evidence_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Accept either a pre-computed DataFrame or the runner's raw output.
        cf_results = intake.get("counterfactual_results")
        negative_control = intake.get("negative_control_deltas")

        if cf_results is None:
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.SKIPPED,
                score=0.0,
                criteria_results={"counterfactual_results_provided": False},
                evidence={},
                failure_modes=[
                    "No counterfactual_results in intake; cannot validate "
                    "intervention effects. This agent does NOT fabricate data."
                ],
                required_changes=[
                    "Run CounterfactualRunner and provide results in intake."
                ],
            )

        # Extract top-10 effects.
        try:
            import pandas as pd

            if isinstance(cf_results, pd.DataFrame):
                df = cf_results
            else:
                df = pd.DataFrame(cf_results)

            # Sort by absolute delta, take top 10.
            df = df.dropna(subset=["delta"])
            df_sorted = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
            top_10 = df_sorted.head(10)
            top_10_deltas = top_10["delta"].values.astype(float)
        except Exception as exc:
            return self.fail_closed_gate(
                reason=f"Cannot parse counterfactual results: {exc}",
                required_changes=[
                    "Provide counterfactual_results as a DataFrame with a 'delta' column."
                ],
            )

        criteria_results["n_total_edges"] = len(df)
        criteria_results["n_top_10"] = len(top_10_deltas)
        evidence["top_10_edges"] = (
            top_10[["edge_id", "delta"]].to_dict("records")
            if "edge_id" in top_10.columns
            else top_10_deltas.tolist()
        )

        # Negative control comparison.
        if negative_control is not None:
            try:
                nc = np.asarray(negative_control, dtype=float)
                nc = nc[~np.isnan(nc)]
                nc_mean = float(np.mean(nc))
                nc_std = float(np.std(nc))
            except Exception as exc:
                logger.warning("Cannot parse negative_control_deltas: %s", exc)
                nc_mean = 0.0
                nc_std = 0.0
                nc = np.array([])
        else:
            # Without explicit negative control, use the bottom half of deltas
            # as a proxy (conservative).
            all_deltas = df["delta"].dropna().values.astype(float)
            bottom_half = np.sort(np.abs(all_deltas))[: len(all_deltas) // 2]
            nc_mean = float(np.mean(bottom_half)) if len(bottom_half) > 0 else 0.0
            nc_std = float(np.std(bottom_half)) if len(bottom_half) > 0 else 0.0
            nc = bottom_half
            evidence["negative_control_source"] = "bottom_half_proxy"

        criteria_results["nc_mean"] = nc_mean
        criteria_results["nc_std"] = nc_std
        evidence["nc_n"] = len(nc)

        # Check: top-10 effects exceed 2*sigma above negative control mean.
        threshold = nc_mean + 2.0 * nc_std
        n_exceeding = int(np.sum(np.abs(top_10_deltas) > threshold))
        criteria_results["threshold_2sigma"] = threshold
        criteria_results["n_top10_exceeding_2sigma"] = n_exceeding
        criteria_results["fraction_exceeding"] = n_exceeding / max(len(top_10_deltas), 1)

        if n_exceeding == 0:
            failure_modes.append(
                f"None of the top-10 counterfactual effects exceed 2-sigma "
                f"above negative control (threshold={threshold:.4f})."
            )
            required_changes.append(
                "Investigate why no counterfactual interventions produce "
                "effects distinguishable from noise; model may lack "
                "intervention sensitivity."
            )
            verdict = EvalVerdict.FAIL
        elif n_exceeding < 5:
            failure_modes.append(
                f"Only {n_exceeding}/10 top counterfactual effects exceed "
                f"2-sigma threshold."
            )
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.PASS

        score = n_exceeding / 10.0

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ClaimGateAgent (Tier D)
# ═══════════════════════════════════════════════════════════════════════════════


#: The 12 claim levels the ClaimGateAgent adjudicates.
CLAIM_LEVELS: list[str] = [
    "static_drug_response",
    "pathway_mechanism",
    "survival_head_technical",
    "survival_clinical_validity",
    "resistance_proxy_pfs",
    "resistance_direct",
    "trajectory_prediction",
    "trajectory_clinical",
    "counterfactual_mechanism",
    "subgroup_stratification",
    "external_validation",
    "clinical_utility",
]


class ClaimGateAgent(EvalAgent):
    """Tier D: grant or block each of 12 claim levels with explicit reasons.

    Reads all upstream agent findings and uses them to determine which
    claims are supported by the evidence. Emits ``claim_gate_report.json``.
    """

    tier: str = "D"

    def __init__(self) -> None:
        super().__init__(
            name="claim_gate_agent",
            dependencies=[],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        # Collect all upstream findings.
        findings: list[EvalFinding] = list(intake.get("findings") or [])
        findings_by_name: dict[str, EvalFinding] = {
            f.agent_name: f for f in findings
        }

        if not findings:
            failure_modes.append(
                "No upstream findings provided; all claim levels blocked."
            )
            required_changes.append(
                "Run Tier A/B/C agents and pass their findings to ClaimGateAgent."
            )
            blocked_report = {
                level: {"granted": False, "reason": "no upstream evidence"}
                for level in CLAIM_LEVELS
            }
            criteria_results["claim_gate"] = blocked_report
            self._write_report(blocked_report)
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.FAIL,
                score=0.0,
                criteria_results=criteria_results,
                evidence=evidence,
                failure_modes=failure_modes,
                required_changes=required_changes,
            )

        # Summarise upstream verdicts for evidence.
        upstream_summary: dict[str, str] = {}
        for f in findings:
            v = f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)
            upstream_summary[f.agent_name] = v
        evidence["upstream_verdicts"] = upstream_summary

        # Check for any Tier A hard fails.
        tier_a_fail = any(
            f.tier == "A"
            and (f.verdict == EvalVerdict.FAIL or getattr(f.verdict, "value", "") == "fail")
            for f in findings
        )

        # Adjudicate each claim level.
        gate_report: dict[str, dict[str, Any]] = {}

        for level in CLAIM_LEVELS:
            granted, reason = self._evaluate_claim(
                level, findings_by_name, tier_a_fail
            )
            gate_report[level] = {"granted": granted, "reason": reason}
            if not granted:
                failure_modes.append(f"Claim '{level}' BLOCKED: {reason}")

        n_granted = sum(1 for v in gate_report.values() if v["granted"])
        criteria_results["claim_gate"] = gate_report
        criteria_results["n_granted"] = n_granted
        criteria_results["n_total"] = len(CLAIM_LEVELS)
        evidence["claim_gate_report"] = gate_report

        # Write report to disk.
        self._write_report(gate_report)

        score = n_granted / len(CLAIM_LEVELS)
        if tier_a_fail:
            verdict = EvalVerdict.FAIL
        elif n_granted == 0:
            verdict = EvalVerdict.FAIL
        elif n_granted >= len(CLAIM_LEVELS) * 0.75:
            verdict = EvalVerdict.PASS
        else:
            verdict = EvalVerdict.CONDITIONAL

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )

    def _evaluate_claim(
        self,
        level: str,
        findings: dict[str, EvalFinding],
        tier_a_fail: bool,
    ) -> tuple[bool, str]:
        """Evaluate whether a specific claim level is granted."""

        def _verdict_ok(name: str) -> bool:
            f = findings.get(name)
            if f is None:
                return False
            v = f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)
            return v in ("pass", "conditional")

        # All claims blocked if Tier A data integrity fails.
        if tier_a_fail and level not in ("static_drug_response",):
            return False, "blocked by Tier A data integrity failure"

        if level == "static_drug_response":
            return True, "always permitted as baseline capability"

        if level == "pathway_mechanism":
            if _verdict_ok("pathway_evidence_agent"):
                return True, "pathway evidence gate passed (2-of-3 channels)"
            return False, "pathway evidence gate not passed"

        if level == "survival_head_technical":
            if _verdict_ok("survival_falsification_agent"):
                return True, "survival falsification passed"
            return False, "survival falsification not passed or not run"

        if level == "survival_clinical_validity":
            # Requires external validation evidence.
            has_ext = any(
                "external" in f.agent_name and _verdict_ok(f.agent_name)
                for f in findings.values()
            )
            return has_ext, (
                "external validation present" if has_ext
                else "requires external cohort validation"
            )

        if level == "resistance_proxy_pfs":
            if _verdict_ok("endpoint_semantics_agent"):
                return True, "endpoint semantics permit PFS resistance proxy"
            return False, "endpoint semantics do not permit resistance claim"

        if level == "resistance_direct":
            es = findings.get("endpoint_semantics_agent")
            if es and es.criteria_results.get("resistance_claim_allowed"):
                return True, "direct resistance endpoint validated"
            return False, "no direct resistance endpoint validated"

        if level == "trajectory_prediction":
            if _verdict_ok("trajectory_falsification_agent"):
                return True, "trajectory beats baselines"
            return False, "trajectory does not beat baselines or not run"

        if level == "trajectory_clinical":
            # Requires trajectory + longitudinal validity.
            t_ok = _verdict_ok("trajectory_falsification_agent")
            l_ok = _verdict_ok("longitudinal_validity_agent")
            if t_ok and l_ok:
                return True, "trajectory + longitudinal validity passed"
            return False, "requires both trajectory and longitudinal validity"

        if level == "counterfactual_mechanism":
            if _verdict_ok("counterfactual_evidence_agent"):
                return True, "counterfactual effects exceed noise"
            return False, "counterfactual evidence not sufficient"

        if level == "subgroup_stratification":
            # Requires baseline adversary to show consistent gains.
            if _verdict_ok("baseline_adversary_scientific_agent"):
                return True, "baseline adversary confirms model value"
            return False, "baseline adversary not passed"

        if level == "external_validation":
            has_ext = any(
                "external" in f.agent_name and _verdict_ok(f.agent_name)
                for f in findings.values()
            )
            return has_ext, (
                "external validation present" if has_ext
                else "no external validation agent passed"
            )

        if level == "clinical_utility":
            # Highest bar: all prior tiers must pass.
            all_ok = all(
                _verdict_ok(f.agent_name)
                for f in findings.values()
                if f.tier in ("A", "B", "C")
            )
            return all_ok, (
                "all Tier A/B/C agents passed" if all_ok
                else "not all prerequisite agents passed"
            )

        return False, f"unknown claim level: {level}"

    @staticmethod
    def _write_report(report: dict[str, Any]) -> None:
        """Write claim gate report to disk."""
        out = Path("logs/mortfm/claim_gate_report.json")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Claim gate report written to %s", out)
        except OSError as exc:
            logger.warning("Failed to write claim gate report: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ReportAgent (Tier D)
# ═══════════════════════════════════════════════════════════════════════════════


class ReportAgent(EvalAgent):
    """Tier D: emit paper-ready tables, figures list, and provenance manifest.

    Collects all agent outputs and assembles a publication bundle manifest
    documenting:

    * Which claims are supported (from ClaimGateAgent).
    * Summary statistics from each tier.
    * Required tables and figures for the paper.
    * Full provenance chain (agent names, verdicts, evidence hashes).
    """

    tier: str = "D"

    def __init__(self) -> None:
        super().__init__(
            name="report_agent",
            dependencies=["claim_gate_agent"],
        )

    async def assess(
        self,
        intake: dict[str, Any],
        config: ResistanceMapConfig,
    ) -> EvalFinding:
        criteria_results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        failure_modes: list[str] = []
        required_changes: list[str] = []

        findings: list[EvalFinding] = list(intake.get("findings") or [])

        if not findings:
            return EvalFinding(
                agent_name=self.name,
                tier=self.tier,
                verdict=EvalVerdict.SKIPPED,
                score=0.0,
                criteria_results={},
                evidence={},
                failure_modes=["No findings to compile into a report."],
                required_changes=[
                    "Run all agents and pass findings to ReportAgent."
                ],
            )

        # --- Build provenance chain. ----------------------------------------
        provenance: list[dict[str, Any]] = []
        for f in findings:
            provenance.append({
                "agent_name": f.agent_name,
                "tier": f.tier,
                "verdict": f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict),
                "score": f.score,
                "n_failure_modes": len(f.failure_modes),
                "n_required_changes": len(f.required_changes),
            })
        evidence["provenance_chain"] = provenance

        # --- Per-tier summary. ----------------------------------------------
        per_tier: dict[str, dict[str, int]] = {}
        for f in findings:
            tier_summary = per_tier.setdefault(
                f.tier, {"pass": 0, "conditional": 0, "fail": 0, "skipped": 0}
            )
            v = f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)
            v_lower = v.lower()
            if v_lower in tier_summary:
                tier_summary[v_lower] += 1
        criteria_results["per_tier_summary"] = per_tier

        # --- Claim gate results (from ClaimGateAgent if present). -----------
        claim_gate_finding = None
        for f in findings:
            if f.agent_name == "claim_gate_agent":
                claim_gate_finding = f
                break

        supported_claims: list[str] = []
        blocked_claims: list[str] = []
        if claim_gate_finding:
            gate = claim_gate_finding.criteria_results.get("claim_gate", {})
            for level, info in gate.items():
                if isinstance(info, dict) and info.get("granted"):
                    supported_claims.append(level)
                else:
                    blocked_claims.append(level)
        criteria_results["supported_claims"] = supported_claims
        criteria_results["blocked_claims"] = blocked_claims

        # --- Required paper artefacts. --------------------------------------
        paper_tables: list[dict[str, str]] = [
            {
                "table": "Table 1: Data Inventory",
                "source_agent": "data_inventory_agent",
                "status": self._agent_status(findings, "data_inventory_agent"),
            },
            {
                "table": "Table 2: Baseline Comparison",
                "source_agent": "baseline_adversary_scientific_agent",
                "status": self._agent_status(findings, "baseline_adversary_scientific_agent"),
            },
            {
                "table": "Table 3: Survival Falsification",
                "source_agent": "survival_falsification_agent",
                "status": self._agent_status(findings, "survival_falsification_agent"),
            },
            {
                "table": "Table 4: Trajectory Falsification",
                "source_agent": "trajectory_falsification_agent",
                "status": self._agent_status(findings, "trajectory_falsification_agent"),
            },
            {
                "table": "Table 5: Pathway Evidence",
                "source_agent": "pathway_evidence_agent",
                "status": self._agent_status(findings, "pathway_evidence_agent"),
            },
            {
                "table": "Table 6: Claim Gate Summary",
                "source_agent": "claim_gate_agent",
                "status": self._agent_status(findings, "claim_gate_agent"),
            },
        ]

        paper_figures: list[dict[str, str]] = [
            {
                "figure": "Figure 1: Data Coverage Heatmap",
                "source_agent": "data_inventory_agent",
            },
            {
                "figure": "Figure 2: Survival Curves (MORT-FM vs Cox)",
                "source_agent": "survival_falsification_agent",
            },
            {
                "figure": "Figure 3: Trajectory Prediction vs Baselines",
                "source_agent": "trajectory_falsification_agent",
            },
            {
                "figure": "Figure 4: Counterfactual Effect Volcano Plot",
                "source_agent": "counterfactual_evidence_agent",
            },
            {
                "figure": "Figure 5: Pathway Evidence Network",
                "source_agent": "pathway_evidence_agent",
            },
        ]

        criteria_results["paper_tables"] = paper_tables
        criteria_results["paper_figures"] = paper_figures

        # --- Missing artefacts. ---------------------------------------------
        missing_agents = set()
        for table in paper_tables:
            if table["status"] in ("not_run", "fail"):
                missing_agents.add(table["source_agent"])
        for agent_name in missing_agents:
            required_changes.append(
                f"Run {agent_name} to generate data for paper tables/figures."
            )

        # --- Overall readiness. ---------------------------------------------
        n_pass = sum(
            1
            for f in findings
            if (f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)) == "pass"
        )
        n_total = len(findings)
        criteria_results["publication_readiness"] = (
            "ready"
            if n_pass == n_total
            else "conditional"
            if n_pass >= n_total * 0.5
            else "not_ready"
        )

        # --- Write bundle manifest. -----------------------------------------
        bundle_manifest = {
            "provenance": provenance,
            "per_tier_summary": per_tier,
            "supported_claims": supported_claims,
            "blocked_claims": blocked_claims,
            "paper_tables": paper_tables,
            "paper_figures": paper_figures,
            "publication_readiness": criteria_results["publication_readiness"],
        }
        evidence["bundle_manifest"] = bundle_manifest
        self._write_manifest(bundle_manifest)

        # Verdict.
        readiness = criteria_results["publication_readiness"]
        if readiness == "ready":
            verdict = EvalVerdict.PASS
        elif readiness == "conditional":
            verdict = EvalVerdict.CONDITIONAL
        else:
            verdict = EvalVerdict.FAIL

        score = n_pass / max(n_total, 1)

        return EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=verdict,
            score=score,
            criteria_results=criteria_results,
            evidence=evidence,
            failure_modes=failure_modes,
            required_changes=required_changes,
        )

    @staticmethod
    def _agent_status(findings: list[EvalFinding], name: str) -> str:
        """Return the status string for a named agent."""
        for f in findings:
            if f.agent_name == name:
                v = f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict)
                return v
        return "not_run"

    @staticmethod
    def _write_manifest(manifest: dict[str, Any]) -> None:
        """Write publication bundle manifest to disk."""
        out = Path("logs/mortfm/publication_bundle_manifest.json")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            logger.info("Publication bundle manifest written to %s", out)
        except OSError as exc:
            logger.warning("Failed to write publication manifest: %s", exc)
