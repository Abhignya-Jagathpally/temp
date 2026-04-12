# Evaluation Governance for ResistanceMap

## Purpose

The evaluation governance layer is a multi-agent review system that runs *orthogonally* to the existing ResistanceMap training DAG. The training orchestrator (`resistancemap.agents.orchestrator.Orchestrator`, wired via `build_agent_dag()`) is responsible for producing artifacts: VAE checkpoints, trajectory ensembles, the protein interaction GNN, the fusion model, the resistance landscape predictor, and validation metrics. The governance layer is responsible for *adjudicating* whether those artifacts justify the scientific claims that ResistanceMap intends to make about myeloma drug resistance.

The governance layer consumes the same `ResistanceMapConfig` and reads training outputs (checkpoints, metric tables, attribution maps, frozen evaluation datasets) but must not write back into the training DAG. It does not reimplement training. It does not silently mutate config files. It produces an `EvalReport` with per-tier verdicts, a verification hash chain, a list of required changes, and timing — all of which feed back to the user as a closed loop. Re-running governance after the punch-list is closed is the only path to a passing verdict.

## Testable claims and epistemic limits

ResistanceMap makes three falsifiable scientific claims. Anything outside these three is out of scope for the governance charter.

| Claim | Operational meaning | What would falsify it |
|---|---|---|
| **state** | A discrete or continuous label assignable to a patient at time *t* describing the resistance phenotype, predicted from multi-omics measurements. | Calibration error larger than the worst-case naive predictor, or accuracy indistinguishable from class-prior baseline under patient-clustered cross-validation with bootstrap CIs. |
| **when** | A time-to-event or per-horizon distribution over the next resistance event, evaluated at `forecast_horizons: [3, 6, 12]` months. | Concordance index, time-dependent AUC, or pinball loss at the configured horizons no better than last-observation-carried-forward and a Kaplan-Meier-style parametric baseline, after multiple-testing correction. |
| **pathway** | An attributed mechanism over the STRING PPI context (gene set, sub-graph, or knockout direction) explaining a predicted state transition. | Attribution that is unstable under PPI edge perturbation (low top-k overlap across perturbed runs), or attributions that fail to generalize to a pathway-level holdout (held-out perturbation cohort). |

A note on the language **"before it happens"**: a strict prospective claim is only demonstrable with prospective or strictly time-ordered data. Cross-sectional snapshots — even when paired across patients — bound what can be claimed to "consistent with prior longitudinal patterns" rather than "predicts the future event in this patient." The governance charter encodes this asymmetry in the `epistemic_limits` field of every claim, and the Tier C forecasting agent must reject any "before it happens" wording that is unsupported by the underlying split structure.

## 10-agent roster

### Tier A — gating

- **Data adequacy.** Audits sample size, modality coverage, label availability, and missingness patterns against the operational meaning of each claim. Rejects evaluation runs where the cohort cannot in principle support the claim being tested (e.g. a "when" claim on a cohort with no follow-up).
- **Measurement & integration.** Audits the harmonization step: protein quantification pipelines, ATAC peak calling, single-cell QC thresholds (`scrna_min_genes`, `scrna_max_mt`), and cross-modality patient-ID joins. Rejects implicit modality dropouts and unjoined samples masquerading as full-modality cases.
- **Bias / fairness / shift.** Checks demographic balance, batch confounders, and distribution shift between training, validation, and any external cohort using maximum mean discrepancy and per-subgroup metric decomposition.

### Tier B — model substance

- **Architecture auditor.** Reads the training DAG and the chosen architecture (VAE latent geometry, ODE solver, GNN convolution type, fusion rule). Asks whether each component is necessary: ablations must be either present in the checkpoint set or in the punch list.
- **Cell-state & trajectory specialist.** Adjudicates whether the latent dynamics are interpretable as cell-state transitions versus arbitrary embedding drift. Requires alignment with at least one external single-cell perturbation cohort.
- **Pathway & PPI reasoning.** Checks that pathway attributions are PPI-consistent, sparse, and stable under graph perturbation; checks that drug-target lists used at inference time match `target_drugs` in the config.

### Tier C — outcome substance

- **Forecasting & uncertainty.** Verifies that uncertainty estimates are calibrated (per-horizon ECE, integrated Brier score), that pinball loss is reported at every configured horizon, and that the model is not silently degenerate (e.g. always predicting the prior).
- **Baseline adversary.** Owns the reference predictors: clinical-only, transcriptome-only, last-observation-carried-forward, and a simple parametric survival model. Adversary holds the burden of refutation: ResistanceMap is retained only if it beats every baseline by a statistically significant margin under patient-clustered nested CV.
- **Clinical translation & safety.** Translates each retained claim into a clinical decision context, identifies failure modes whose cost is asymmetric (a missed early relapse is not a missed routine visit), and writes a model card with intended use, contraindications, and known failure modes.

### Tier D — synthesis

- **Principal integrator / chair.** Reads all nine findings, runs the adversarial round (architecture auditor versus baseline adversary), applies the resolution rules, writes the synthesis, freezes the verification chain, and produces the final `EvalReport`.

### Optional 11th agent

- **CLIA-adjacent regulatory.** Engaged whenever an evaluation run touches actual patient decisions (live deployment, pre-clinical reporting, or external validation cohorts with PHI). Reviews data provenance, software-as-a-medical-device classification, and audit log retention. Skipped for purely retrospective benchmarking.

## Orchestration pattern

```
intake package (config + checkpoints + frozen eval dataset)
        │
        ▼
Tier A  ──parallel──>  data_adequacy  measurement_integration  bias_fairness
        │
        │  hard stop: if ANY Tier A verdict == FAIL, all later
        │  agents return Verdict.BLOCKED and the chair short-circuits.
        ▼
Tier B  ──parallel──>  architecture_auditor  cell_state_specialist  pathway_ppi
Tier C  ──parallel──>  forecasting_uncertainty  baseline_adversary  clinical_safety
        │
        ▼
adversarial round
  - architecture_auditor presents necessity ablations
  - baseline_adversary presents refutation evidence
  - resolution rules: only peer-reviewed-style evidence or
    pre-registered experiments count; opinion does not.
        │
        ▼
Tier D chair synthesis  ──>  EvalReport
        │
        ▼
required_changes punch list  ──>  re-run after closure (closed loop)
```

This loop is *orthogonal* to `build_agent_dag()`. The training DAG runs once per checkpoint set; the evaluation DAG runs once per claim review. They share configs and artifacts but never share an `Orchestrator` instance.

## Deterministic factors table

| Factor | Scientific hypothesis | Metric | Ablation | Why simpler alternatives fail |
|---|---|---|---|---|
| Latent geometry (VAE) | A low-dimensional manifold separates resistant from sensitive states across modalities. | Reconstruction error, latent class separability, downstream calibration. | Replace conditional VAE with PCA at the same latent dim. | PCA is linear and ignores epigenome conditioning; cannot represent the proteome→epigenome map. |
| ODE dynamics | State trajectories are governed by a smooth dynamical system that admits forecasting at fixed horizons. | Time-dependent AUC, per-horizon pinball loss, calibration. | Replace ODE block with a discrete-step LSTM. | Discrete steppers do not enforce smoothness or solver-controlled error and cannot translate "ODE time" to calendar time. |
| Fusion rules | Cross-attention between modalities outperforms early concat or late ensemble. | Held-out fusion metric vs. ablation cohort. | concat / gated / no-fusion. | Concat collapses modality-specific scales; late ensemble cannot exploit cross-modality interactions. |
| PPI graph (GNN) | Predictions explained on a known interaction graph generalize across cohorts. | Top-k pathway overlap, edge-perturbation stability. | MLP with the same parameter budget. | MLP discards graph structure and yields attributions with no biological prior. |
| Drug-target list | Predictions and attributions are consistent with the configured `target_drugs`. | Target enrichment in attribution, drug-response calibration. | Random drug list. | A random list would test the null and is the negative control, not an alternative. |

## Metric families aligned to the pipeline

### State prediction

- Multiclass AUROC and macro-F1 for the discrete resistance label.
- Expected calibration error (ECE).
- Patient-clustered cross-validation: splits by `patient_id`, never by sample.
- Per-subgroup decomposition for the fairness agent.

### When (time-to-event)

- Time-dependent AUC at `forecast_horizons: [3, 6, 12]`.
- Integrated Brier score over the same horizons.
- Pinball loss at multiple quantiles, reported per horizon.
- Per-horizon calibration plots, with reliability diagrams archived as evidence.

### Pathway

- Top-k pathway overlap against held-out perturbation knowledge bases.
- Stability under PPI edge perturbation: top-k overlap across N edge-dropout runs.
- In-silico pathway knockout tests: predicted-effect direction must match the literature when a pathway is masked from the GNN.

## "Why other ways would not contribute as much" — structured reasoning requirement

Every Tier B and Tier C agent must, for each claim it touches, fill the following template before the chair will accept its finding:

| Field | Content |
|---|---|
| Alternative A | The named simpler or competing approach. |
| What it optimizes | The loss / objective the alternative actually minimizes. |
| What biological structure it discards | The structure (graph, ordering, modality, prior) that the alternative ignores. |
| Empirical criterion | The pre-registered statistical test that decides "kept" versus "discarded." |
| Conclusion | The chosen architecture is retained *only if* the gap is significant under nested CV with bootstrap confidence intervals and multiple-testing correction. |

The chair rejects findings that skip this template. "We chose X because it works" is not a finding.

## Required changes (punch list)

### Data and labels

- Longitudinal samples with documented sampling cadence per patient.
- Resistance ontology: a single, versioned vocabulary linking discrete labels and continuous scores.
- Harmonized patient IDs across proteomics, epigenomics, single-cell, and clinical modalities.
- At least one external validation cohort with prospective follow-up.

### Modeling

- Explicitly map ODE integration time to calendar time, or drop all calendar-time claims.
- Honest uncertainty for rare modes: low-prior subgroups must not be reported with the same confidence as well-sampled subgroups.
- Pathway outputs must be tied to PPI-consistent explanations; free-form gene sets are not accepted.

### Evaluation infrastructure

- Patient-level splits, never sample-level.
- Nested hyperparameter search inside every CV fold.
- Leakage audits (`audit_patient_split`, `detect_temporal_leakage`) must run on every freshly produced split before any metric is computed.
- A prospective protocol document for any "before event" claim.
- Model cards and data sheets generated automatically from the evaluation report.

### Governance

- Pre-registration of benchmark protocols on a versioned, append-only store.
- Versioned datasets and checkpoints with cryptographic hashes.
- Extended audit trail beyond the existing `verification.audit_log_path`, covering every governance run.

## Connection to the codebase

| Training stage in `build_agent_dag()` | Output consumed by | Eval tier |
|---|---|---|
| `data_validation` | data adequacy, measurement & integration | A |
| `data_prep` | measurement & integration, bias / fairness / shift | A |
| `vae_pretrain`, `vae_finetune` | architecture auditor, cell-state specialist | B |
| `esm2_embed`, `protein_net` | architecture auditor, pathway & PPI | B |
| `trajectory` | cell-state specialist, forecasting & uncertainty | B, C |
| `fusion` | architecture auditor, forecasting & uncertainty | B, C |
| `landscape` | clinical translation & safety, pathway & PPI | B, C |
| `validation` | baseline adversary, chair | C, D |

## Practical launch checklist

1. **Charter document.** A frozen YAML/JSON file enumerating the three claims, their non-goals, the falsifiers, the epistemic limits, and the stop rules (Tier A hard stop, adversarial-round resolution rules, chair veto).
2. **Rubric spreadsheet.** Agents x criteria x weights x evidence type, persisted as `rubric.yaml` and round-trippable through `Rubric.to_yaml()` / `Rubric.from_yaml()`.
3. **Frozen evaluation dataset.** A versioned, hash-pinned copy of the held-out cohort, with leaderboard rules (no peeking, single submission per protocol revision).
4. **First dry run.** Execute the full evaluation DAG on a small subset with deliberately weak checkpoints to confirm that Tier A actually hard-stops, that the chair actually fires, and that the verification chain has the expected length.

## How to invoke

```
python -m resistancemap eval --config configs/default.yaml
```

This command is a placeholder pending the `main.py` wiring step, which is owned by a separate pull request. The evaluation governance layer must remain runnable as `python -m resistancemap.evaluation` for offline review even after the `main.py` integration lands.
