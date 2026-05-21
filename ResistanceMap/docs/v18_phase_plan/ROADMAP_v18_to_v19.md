# ROADMAP — MORT-FM v18 → v19 (Unified)

**Branch:** `v18-clean-canonical-mortfm`
**Working dir:** `/home/aj0486@students.ad.unt.edu/pipeline3`
**Author:** senior integrator (Wave-1 synthesis of 10 phase docs)
**Date:** 2026-05-20
**Status:** design-locked roadmap; no code committed in this document.

This document flattens the ten Wave-1 phase docs
(`phase_0_canonical_trainer.md`, `phase_1_6_7_trajectory.md`,
`phase_2_9_multiomics.md`, `phase_3_drug_encoding.md`,
`phase_4_5_graph_heads.md`, `phase_8_counterfactual.md`,
`phase_10_baselines.md`, `phase_11_visualizations.md`,
`phase_12_13_15_infra.md`, `phase_14_claim_gates.md`) into a single
priority-ordered execution plan. Every numeric or PMID assertion in this doc
is anchored to one of those source phase docs (cited inline) — no PMID has
been introduced here that is not already verified in a Wave-1 doc.

---

## 1. Executive verdict (≤200 words)

**v19 IS paper-ready for the falsification framework as the headline
contribution; v19 is NOT paper-ready for a "MORT-FM beats SOTA on PFS
C-index" claim.** Project memory is unambiguous: v11.5 ties mmSYGNAL on PFS
C-index, v12 fell *below* v11.5 on the same task, and trivial per-drug-mean
already wins drug-MSE
(`project_resistancemap_floor_evidence.md`). The single most consequential
change is **Phase 0 — make `MORTFM.forward()` the path the trainer actually
exercises**: today every patient-level stage still routes through
`forward_legacy(...)` at `trainer.py:227`, so the canonical LENS dict
surface (`hazard`, `survival_curve`, `cif_per_event`, `basin_probs`,
`hitting_cdf`, `hitting_mean_tau`) is unreachable from any real training
run. Phase 0 is a strict blocker for Phases 1, 6, 7, 8 — and therefore
for every clinical claim above `multiomic_foundation` in the Phase 14
matrix. **Most likely failure mode** of executing this plan: scope creep
inside Phase 4 (HGT backbone on the full STRING + Reactome graph) and
identifiability-limit failure of Phase 1 trajectory supervision at the
known N_paired = 29 ceiling (`project_v10_sprint7_outcome.md`). The
falsification framework is what we should pitch; the metric beats are not.

---

## 2. Bugs already found by Wave 1 (highest-leverage near-term fixes)

These are verbatim from Wave-1 docs + verified by file-grep against the
v18 branch (`trainer.py:227`, `trainer.py:289`, `trainer.py:302/307`,
`intervention_graph.py:105`, `mortfm_dataset.py:243-246`).

| # | Bug | File:line | Source phase doc | Severity |
|---|---|---|---|---|
| B1 | Trainer routes patient-level stages through `forward_legacy`, canonical LENS dict surface unreachable | `mortfm/trainer.py:227-232` | Phase 0 §1.1 | **P0 blocker** |
| B2 | `prediction.trajectory_mean[-1]` indexes the batch axis instead of the time axis (silent semantic bug — loss minimises a basis-dependent slice) | `mortfm/trainer.py:289` | Phase 1 §1.1 | P0 |
| B3 | SDE rolls on `linspace(0, 1.0, 25)` (LENS internal) but survival head bins on `linspace(0, 12.0, 13)` (months) — directional consistency invariant is *unevaluable* | `mortfm/trajectory/graph_energy_sde.py` vs `mortfm/trainer.py:302,307` and `model.py:151` | Phase 7 §7.1 | **P0** (loss currently impossible) |
| B4 | `abs(hash(eid))` is per-process randomised on Python ≥ 3.3 unless `PYTHONHASHSEED` is set — basis vectors non-reproducible across runs | `mortfm/causal/intervention_graph.py:105-107` | Phase 8 §1.1 | P1 (reproducibility) |
| B5 | `WhatIfAnalyzer.inhibit_pathway()` clamps an ODE latent dim ≠ do-calculus; it is associational, not interventional, but a caller can mistake it for the latter | `interpretability/counterfactual_engine.py:545-615` | Phase 8 §1.3 | P1 (semantics) |
| B6 | `scripts/mortfm/08_eval_causal_evidence_v2.py` is a shim that delegates to `scripts/mortfm_causal_evidence_report_v2.py` which does not exist — exits 2 immediately | `scripts/mortfm/08_eval_causal_evidence_v2.py:1-21` | Phase 8 §1.4 | P1 (broken entry-point) |
| B7 | `mort_collate()` synthesises `future_state` from raw RNA mean of the follow-up snapshot, then the trainer truncates to the first `d_latent` genes — MMD compares apples (SDE latent) to oranges (alphabetically-first-128-gene RNA) | `data/mortfm_dataset.py:235-246` + `mortfm/trainer.py:287-296` | Phase 1 §1.1 | **P0** |
| B8 | `configs/mortfm.yaml` enables only 4/8 modalities; ATAC/methylation/histone-PTM/phospho loaders are 59–87-line stubs never consumed by `mortfm_dataset.py` | `configs/mortfm.yaml:19-26` + four stub loaders | Phase 2 §2.1 | P1 |
| B9 | `_stack_drug()` packs only `[dose, start_time, end_time]` (shape `(N, 3)`) — `drug_name`, `drug_class`, `target_genes`, `chembl_id`, `molecular_embedding` from `DrugContext` never reach the SDE → drug identity lost | `data/mortfm_dataset.py:113-200` | Phase 3 §1 | **P0** for mechanistic claims |
| B10 | `LatentToGraphProjector` is an MLP whose own docstring states "It is NOT a graph neural network" — `forward(z0)` returns `(B, 16)` without touching graph topology | `mortfm/trajectory/graph_projector.py:1-52` | Phase 4 §4.1 | P0 for graph claims |
| B11 | `generate_paper_figures.py` uses `np.random` in most panels (project memory: `project_figure_fabrication_risk.md`); Phase 11 fabrication-firewall is the mitigation | `paper/generate_paper_figures.py` | Phase 11 §0 + memory | **P0** for paper |
| B12 | `claim_gate_registry.py` declares `forbidden_endpoint_types=["overall_survival"]` for `resistance_emergence` but it is **not enforced at runtime** — registry is documentation only | `mortfm/registries/claim_gate_registry.py` | Phase 14 §1b | P1 (governance) |
| B13 | `edge_evidence_report.py:47` `causal_mechanism_gate_pass` is a hard AND on exactly 3 channels (CRISPR + drug-target + Reactome) — single-channel absence blocks the whole claim even if four other channels agree; needs K-of-N reformulation | `mortfm/causal/edge_evidence_report.py:24-52` | Phase 8 §1.1 | P1 |
| B14 | `MORTBatch.validate_for_loss(...)` already exists (`mortfm/schemas.py:464-496`) and `training/loss_router.py` already exists (`58-131`) — neither is called anywhere from `trainer.py` today | `mortfm/schemas.py`, `training/loss_router.py` | Phase 0 §1.4 | P0 (infra already there) |
| B15 | Stage-F survival branch (`trainer.py:299-312`) re-runs `model.encode + rollout + survival_head` but the head it calls is the v15 legacy `TimeToResistanceHead`, not the canonical `CompetingRiskHead` allocated at `model.py:165-169` | `mortfm/trainer.py:299-312` | Phase 0 §1.1 | P0 |
| B16 | Stage A only runs RNA recon as a "representative"; the trainer comment itself flags this as a placeholder (`trainer.py:236`) — ATAC/methylation/histone-PTM/phospho recon code paths exist in their encoders but are never called | `mortfm/trainer.py:234-246` | Phase 9 §9.1 | P1 |
| B17 | Stage B "masked-modality" loss feeds the masked modality back to its own encoder — it measures fusion-token / encoder-token agreement, not cross-modal prediction | `mortfm/trainer.py:247-256` | Phase 9 §9.1 | P1 |
| B18 | Stage B contrastive aligns only the first two modality embeddings (`embs[0]` vs `embs[1]`) — the other six modalities are ignored | `mortfm/trainer.py:257-263` | Phase 9 §9.1 | P1 |
| B19 | Phase 4 graph-encoder dim mismatch — `GraphEnergyResistanceSDE(d_graph=16)` is hard-coded but the new HGT encoder emits `d_model=64`; SDE must be re-instantiated with `d_graph=config.d_graph` | `mortfm/model.py:162-164,297` | Phase 4 §4.7 | P1 (silent at instantiation, runtime shape error) |

Bugs **the user did not flag** that Wave-1 surfaced (added by this synthesis):
B12, B13, B14, B15, B16, B17, B18, B19. The most consequential of these
is **B14** — the strict-supervision infrastructure is already on disk and
the v18.2 commit shipped `MissingSupervisionError`; the trainer just
doesn't call it. Phase 0 wires it in for free, which is why Phase 0 is the
single highest-leverage change in the entire 15-phase plan.

---

## 3. Priority-ordered roadmap (P0 → P6) and dependency DAG

The 15 phases collapse into a 7-tier dependency DAG. Each tier's nodes
can be implemented in parallel; cross-tier edges are strict prerequisites.

```
P0 (BLOCKER — single sprint)
└── Phase 0   Canonical trainer + canonical_mortfm_losses.py + LossRouter wiring
              (fixes B1, B14, B15)

P1 (unblocked by P0, parallelisable within tier)
├── Phase 1   Trajectory target — future_snapshot_batch + latent_future_state_loss
│             (fixes B2, B7)
├── Phase 7   Grid alignment — single canonical_time_grid for SDE + CompetingRiskHead
│             (fixes B3; consistency loss becomes evaluable)
├── Phase 2   Real multi-omic activation — FeatureRegistry + 4 loader rewrites + 3 configs
│             (fixes B8)
└── Phase 3   Drug ontology — MM_DRUG_MANIFEST (19 ChEMBL-verified entries) + DrugBatch
              (fixes B9)

P2 (depends on P1)
├── Phase 9   Foundation pretraining — per-modality decoders + MaskedModalityTask
│             (depends on Phase 2; fixes B16, B17, B18)
├── Phase 6   Waddington — basin_label_builder + barrier/attractor/monotonicity losses
│             (depends on Phase 1 + Phase 7)
└── Phase 4   PatientConditionedGraphEncoder (HGT) — graph_builder, string_debiasing,
              reactome_loader, uniprot_loader, graph_encoder.py
              (depends on Phase 3 drug-target edges; fixes B10, B19)

P3 (depends on P2)
├── Phase 5   Canonical V2 heads — PathwayRouteHeadV2, DrugRiskHeadV2,
│             UncertaintyHeadV2, CounterfactualSimulator (depends on Phase 4)
└── Phase 8   Counterfactual engine + evidence module — Evidence dataclass,
              perturbation_loader upgrade (DepMap/Perturb-seq/L1000/PRISM),
              K-of-N evidence gating, ClaimMatrix governance
              (depends on Phase 5; fixes B5, B6, B13; partial B4 by removing the
              hash basis in favour of an explicit per-edge basis tensor)

P4 (depends on P3)
├── Phase 10  Baseline registry — registry.py + static_ml/deep_static/sequence/
│             survival/graph; results parquet writer; CLI runner
└── Phase 14  Clinical claim-gate matrix — claim_matrix.py, IMWG validators
              (depends on Phase 8 ClaimMatrix surface; anchors IMWG PMID 27210871,
              MAIA 31141632, CASSIOPEIA 31171419)

P5 (depends on P4)
├── Phase 15  Reproducibility — RunManifest schema, run_manifest.py writer,
│             bundle.py exporter, verify_bundle (depends on Phase 14 ClaimGate dataclass)
└── Phase 12  Agent DAG — 17 MORT-FM agents, ClaimCriticAgent at L9 (depends on Phase 10
              for BaselineSuiteAgent and Phase 14 for ClaimCriticAgent)

P6 (depends on P5)
├── Phase 13  TaskBackend — Local/Ray/Dask/SLURM (depends on Phase 12; needed by
│             AblationAgent, HPO, BaselineSuiteAgent for parallel fan-out)
└── Phase 11  Visualisations — 9 figures with fabrication firewall
              (depends on Phase 15 RunManifest as the *only* allowed metric source —
              fixes B11)
```

ASCII dependency graph:

```
                              ┌────────────┐
                              │  Phase 0   │   P0
                              └──────┬─────┘
                ┌────────────┬──────┴──────┬────────────┐
                v            v             v            v
            ┌───────┐    ┌────────┐    ┌────────┐    ┌───────┐
            │Phase 1│    │Phase 7 │    │Phase 2 │    │Phase 3│   P1
            └───┬───┘    └───┬────┘    └───┬────┘    └───┬───┘
                │            │              │            │
                ├────────────┤              v            v
                v            v          ┌────────┐   ┌────────┐
            ┌───────┐    ┌────────┐    │Phase 9 │   │Phase 4 │
            │Phase 6│    │ (used  │    └────────┘   └───┬────┘ P2
            └───────┘    │ by  6) │                     v
                         └────────┘                 ┌────────┐
                                                    │Phase 5 │
                                                    └───┬────┘
                                                        v
                                                    ┌────────┐
                                                    │Phase 8 │   P3
                                                    └───┬────┘
                                ┌───────────────────────┤
                                v                       v
                            ┌────────┐              ┌────────┐
                            │Phase 10│              │Phase 14│   P4
                            └───┬────┘              └───┬────┘
                                └───────┬───────────────┘
                                        v
                                    ┌────────┐    ┌────────┐
                                    │Phase 15│    │Phase 12│   P5
                                    └───┬────┘    └───┬────┘
                                        └───┬─────────┘
                                            v          v
                                        ┌────────┐  ┌────────┐
                                        │Phase 11│  │Phase 13│   P6
                                        └────────┘  └────────┘
```

### Per-phase definition-of-done summary

| Tier | Phase | Gate predecessor | Key files to create | Key files to modify | DoD test that must pass |
|---|---|---|---|---|---|
| P0 | 0 | — | `mortfm/canonical_trainer.py`; `training/canonical_mortfm_losses.py`; `tests/mortfm/test_canonical_trainer_smoke.py`; `tests/mortfm/test_canonical_trainer.py`; `tests/mortfm/test_canonical_loss_router.py` | `mortfm/trainer.py` (rename to `LegacyMORTFMTrainer`); `training/loss_router.py` (+ `validator_fn`); `mortfm/__init__.py`; `tests/mortfm/test_mortfm_trainer_smoke.py` → `test_legacy_trainer_smoke.py` | `test_canonical_no_legacy_call` (spy on `forward_legacy`, assert 0 calls) — phase 0 §5.1 |
| P1 | 1 | 0 | `tests/mortfm/test_canonical_trajectory_training.py` | `mortfm/schemas.py` (+ `future_snapshot_batch`, `delta_t`, `has_future_snapshot`, `treatment_window`); `data/mortfm_dataset.py` (replace lines 235-246); `data/trajectory_pair_builder.py` (+ `TemporalPairProvenance`); `mortfm/trainer.py` (delete truncation block 287-296) | `grep -R "target\[..., : pred_T" ResistanceMap/` returns empty — phase 1 §1.9 |
| P1 | 7 | 0 | `mortfm/trajectory/grid.py`; `tests/mortfm/test_grid_alignment.py`; `tests/mortfm/test_survival_hitting_consistency.py`; `tests/mortfm/test_hitting_time_nll.py`; `evaluation/survival_metrics.py`; `baselines/survival.py` | `mortfm/trajectory/graph_energy_sde.py` (+ `t_grid` kwarg); `mortfm/survival/competing_risk_head.py` (+ `t_grid`); `mortfm/model.py` (build canonical grid; pass to SDE + head); `training/canonical_mortfm_losses.py` (+ `hitting_time_nll`, `survival_hitting_consistency_loss`) | `torch.equal(model.lens_sde.t_grid, model.lens_competing_risk.t_grid)` — phase 7 §7.8 |
| P1 | 2 | 0 | `data/feature_registry.py`; `configs/mortfm_minimal.yaml`; `configs/mortfm_full_multiomic.yaml`; `configs/mortfm_patient_longitudinal.yaml`; `tests/mortfm/test_feature_registry.py` | `data/scatac_loader.py`, `data/methylation_loader.py`, `data/histone_ptm_loader.py`, `data/phosphoproteomics_loader.py` (all rewritten to emit `ModalityTensor`); `data/mortfm_dataset.py` (wire 4 loaders behind config flags); `tests/mortfm/test_schemas.py` | `test_missing_modality_masked_not_zero_filled` — phase 2 §2.8 |
| P1 | 3 | 0 | `data/drug_ontology.py` (MM_DRUG_MANIFEST × 19 ChEMBL-verified entries); `mortfm/schemas.py::DrugBatch`; `models/encoders/drug_encoder_v2.py`; tests | `data/mortfm_dataset.py` (`_stack_drug` rewrite); `mortfm/trajectory/graph_energy_sde.py` (`drug` arg = `DrugBatch` not 3-scalar); `data/drug_target_loader.py` | `test_drug_identity_reaches_sde` — phase 3 §X DoD |
| P2 | 9 | 1, 2 | `mortfm/foundation/__init__.py`; `mortfm/foundation/decoders.py` (6 decoder classes); `mortfm/foundation/masked_modality.py`; `tests/mortfm/test_foundation_pretraining.py`; `tests/mortfm/test_missing_modality.py` | `training/canonical_mortfm_losses.py` (+ `reconstruction_loss_dispatch`); `mortfm/canonical_trainer.py` (Stage A/B blocks) | `test_masked_modality_task_does_not_use_held_out_input` — phase 9 §9.8 |
| P2 | 6 | 1, 7 | `data/basin_label_builder.py` (IMWG-anchored basin labels); `interpretability/landscape_viz.py`; `evaluation/landscape_metrics.py`; `tests/mortfm/test_basin_label_builder.py`; `tests/mortfm/test_landscape_metrics.py` | `mortfm/schemas.py` (+ `basin_label`); `data/mortfm_dataset.py` (stack `basin_label`); `training/canonical_mortfm_losses.py` (+ `basin_label_loss`, `barrier_regularization_loss`, `attractor_stability_loss`, `monotonicity_loss`); `landscape/landscape_regularizers.py` (+ `lens_*` adapters) | `landscape_viz.plot_potential_2d` raises if `z_samples.shape[0] < 20` — phase 6 §6.9 |
| P2 | 4 | 3 | `data/graph_builder.py`; `data/string_debiasing.py`; `data/reactome_loader.py`; `data/uniprot_loader.py`; `models/encoders/graph_encoder.py` (HGT); `tests/mortfm/test_graph_encoder.py`; `tests/mortfm/test_drug_target_graph.py` | `mortfm/model.py` (replace `LatentToGraphProjector`); `training/canonical_mortfm_losses.py` (+ 3 losses); `mortfm/trajectory/graph_projector.py` deprecated | `test_canonical_targets_seed_matches_chembl` (Venetoclax CHEMBL3137343 → BCL2) — phase 4 §4.9 |
| P3 | 5 | 4 | `mortfm/heads/pathway_route_head.py` (V2); `mortfm/heads/drug_risk_head.py` (V2); `mortfm/heads/uncertainty_head.py` (V2); `mortfm/causal/counterfactual_simulator.py`; `tests/mortfm/test_pathway_route_head.py`; `tests/mortfm/test_counterfactual_simulator.py` | `mortfm/model.py` (head instantiation + 19-key return dict); `training/canonical_mortfm_losses.py` (+ `pathway_bce_loss`, `drug_risk_ranking_loss`, `perturbation_consistency_loss`); `tests/mortfm/test_canonical_forward.py` | `test_phase5_keys_present` (assert 19 keys) — phase 5 §5.4 |
| P3 | 8 | 5 | `mortfm/causal/evidence.py` (ChannelEvidence, ClaimEvidence); `data/perturbation_loader.py` upgrade (4 sources unified); `mortfm/causal/claim_matrix.py`; `scripts/mortfm_causal_evidence_report_v2.py` (the script the existing shim expects) | `mortfm/causal/intervention_graph.py` (remove `hash()` basis); `mortfm/causal/edge_evidence_report.py` (K-of-N); deprecate `interpretability/counterfactual_engine.py` to `legacy.*` | `test_evidence_k_of_n_passes_with_3_of_5` — phase 8 §6 |
| P4 | 10 | 8 | `baselines/registry.py`; `baselines/static_ml.py`; `baselines/deep_static.py`; `baselines/sequence.py`; `baselines/survival.py`; `baselines/graph.py`; `baselines/mortfm/__init__.py`; `baselines/mortfm/ablations.py`; `evaluation/model_comparison.py`; `scripts/mortfm/10_run_baselines.py`; `tests/mortfm/test_baseline_registry.py` | `baselines/__init__.py` (re-export registry) | `test_no_synthetic_fallback_in_results` — phase 10 §10 |
| P4 | 14 | 8 | `governance/claim_matrix.py` (CLAIM_GATE_REGISTRY × 15 rows); `governance/imwg_validator.py`; tests | `mortfm/acceptance_gate.py` (consume `ClaimGate`); `mortfm/registries/claim_gate_registry.py` (enforce, not document) | `test_resistance_emergence_blocks_os_endpoint` — phase 14 §3 |
| P5 | 15 | 14 | `reproducibility/schemas.py` (RunManifest); `reproducibility/run_manifest.py`; `reproducibility/bundle.py`; `tests/reproducibility/*.py` | (none — pure-new module) | `verify_bundle` re-hashes data and reports mismatches — phase 15 §15.4 |
| P5 | 12 | 10, 14, 15 | `agents/mortfm/base.py`; `agents/mortfm/scientific_dag.py`; 17 agent classes under `agents/mortfm/` | `agents/__init__.py` | `ClaimCriticAgent` refuses to let `FigureGenerationAgent` advance past L9 when any claim refuted — phase 12 §12.4 |
| P6 | 13 | 12 | `infrastructure/task_backend.py`; `infrastructure/gpu_scheduler.py`; `infrastructure/hydra_runner.py`; `infrastructure/optuna_runner.py` | `agents/mortfm/ablation.py`, `agents/mortfm/foundation.py` (use TaskBackend) | `LocalAsyncBackend` smoke test runs 4 parallel ablations — phase 13 §13.4 |
| P6 | 11 | 15 | `visualization/__init__.py` (MissingArtifactError, ProvenanceMismatchError, `load_artifact`, `stamp_figure`); `visualization/latent.py`, `trajectory.py`, `waddington.py`, `landscape_metrics.py`, `attribution.py`, `survival.py`, `baselines.py`, `figure1_architecture.py`; `scripts/mortfm/11_generate_figures.py` | `paper/generate_paper_figures.py` deprecated (replaced) | Plot called on missing artifact saves as `*_UNVERIFIED.png` and raises — phase 11 §0 + §1 |

---

## 4. Unified file-change list (table)

Convention: priority is the tier label (P0–P6) from §3; LOC estimate is a
conservative bound based on the Wave-1 doc's documented signatures and test
plans. "Depends on" lists *other rows in this table* whose presence is a
hard prerequisite.

### 4a. New files (created)

| Path | Phase | Priority | LOC est | Depends on |
|---|---|---|---|---|
| `resistancemap/mortfm/canonical_trainer.py` | 0 | P0 | 600 | — |
| `resistancemap/training/canonical_mortfm_losses.py` | 0/1/6/7/9 | P0 (created in 0) | 900 (accreted across phases) | — |
| `tests/mortfm/test_canonical_trainer_smoke.py` | 0 | P0 | 220 | canonical_trainer.py |
| `tests/mortfm/test_canonical_trainer.py` | 0 | P0 | 180 | canonical_trainer.py |
| `tests/mortfm/test_canonical_loss_router.py` | 0 | P0 | 160 | canonical_trainer.py, loss_router.py |
| `tests/mortfm/test_canonical_trajectory_training.py` | 1 | P1 | 220 | canonical_trainer.py, canonical_mortfm_losses.py |
| `resistancemap/mortfm/trajectory/grid.py` | 7 | P1 | 25 | — |
| `tests/mortfm/test_grid_alignment.py` | 7 | P1 | 60 | grid.py, model.py |
| `tests/mortfm/test_survival_hitting_consistency.py` | 7 | P1 | 120 | canonical_mortfm_losses.py |
| `tests/mortfm/test_hitting_time_nll.py` | 7 | P1 | 80 | canonical_mortfm_losses.py |
| `resistancemap/evaluation/survival_metrics.py` | 7 | P1 | 200 | — |
| `resistancemap/baselines/survival.py` | 7/10 | P1 | 350 | — |
| `resistancemap/data/feature_registry.py` | 2 | P1 | 300 | — |
| `configs/mortfm_minimal.yaml` | 2 | P1 | 30 | — |
| `configs/mortfm_full_multiomic.yaml` | 2 | P1 | 50 | feature_registry.py |
| `configs/mortfm_patient_longitudinal.yaml` | 2 | P1 | 45 | mortfm_full_multiomic.yaml |
| `tests/mortfm/test_feature_registry.py` | 2 | P1 | 180 | feature_registry.py |
| `resistancemap/data/drug_ontology.py` | 3 | P1 | 700 (19-drug manifest + aliases + ChEMBL refresh) | — |
| `resistancemap/models/encoders/drug_encoder_v2.py` | 3 | P1 | 220 | drug_ontology.py |
| `tests/mortfm/test_drug_ontology.py` | 3 | P1 | 200 | drug_ontology.py |
| `resistancemap/mortfm/foundation/__init__.py` | 9 | P2 | 10 | — |
| `resistancemap/mortfm/foundation/decoders.py` | 9 | P2 | 350 (6 decoder classes) | — |
| `resistancemap/mortfm/foundation/masked_modality.py` | 9 | P2 | 220 | decoders.py |
| `tests/mortfm/test_foundation_pretraining.py` | 9 | P2 | 240 | decoders.py, masked_modality.py |
| `tests/mortfm/test_missing_modality.py` | 9 | P2 | 220 | masked_modality.py |
| `resistancemap/data/basin_label_builder.py` | 6 | P2 | 200 | — |
| `resistancemap/interpretability/landscape_viz.py` | 6 | P2 | 280 | — |
| `resistancemap/evaluation/landscape_metrics.py` | 6 | P2 | 220 | — |
| `tests/mortfm/test_basin_label_builder.py` | 6 | P2 | 180 | basin_label_builder.py |
| `tests/mortfm/test_landscape_metrics.py` | 6 | P2 | 140 | landscape_metrics.py |
| `resistancemap/data/graph_builder.py` | 4 | P2 | 450 | drug_ontology.py |
| `resistancemap/data/string_debiasing.py` | 4 | P2 | 200 | — |
| `resistancemap/data/reactome_loader.py` | 4 | P2 | 180 | — |
| `resistancemap/data/uniprot_loader.py` | 4 | P2 | 220 | — |
| `resistancemap/models/encoders/graph_encoder.py` | 4 | P2 | 380 (HGT) | graph_builder.py |
| `tests/mortfm/test_graph_encoder.py` | 4 | P2 | 220 | graph_encoder.py |
| `tests/mortfm/test_drug_target_graph.py` | 4 | P2 | 180 | graph_builder.py, graph_encoder.py |
| `resistancemap/mortfm/heads/pathway_route_head.py` (V2) | 5 | P3 | 180 | graph_encoder.py |
| `resistancemap/mortfm/heads/drug_risk_head.py` (V2) | 5 | P3 | 220 | graph_encoder.py |
| `resistancemap/mortfm/heads/uncertainty_head.py` (V2) | 5 | P3 | 200 | model.py |
| `resistancemap/mortfm/causal/counterfactual_simulator.py` | 5 | P3 | 350 | graph_encoder.py |
| `tests/mortfm/test_pathway_route_head.py` | 5 | P3 | 160 | pathway_route_head.py |
| `tests/mortfm/test_counterfactual_simulator.py` | 5 | P3 | 200 | counterfactual_simulator.py |
| `resistancemap/mortfm/causal/evidence.py` | 8 | P3 | 380 (ChannelEvidence + 5 factory fns) | — |
| `resistancemap/mortfm/causal/claim_matrix.py` | 8 | P3 | 300 | evidence.py |
| `scripts/mortfm_causal_evidence_report_v2.py` | 8 | P3 | 250 | claim_matrix.py |
| `tests/mortfm/test_evidence.py` | 8 | P3 | 220 | evidence.py |
| `resistancemap/baselines/registry.py` | 10 | P4 | 120 | — |
| `resistancemap/baselines/static_ml.py` | 10 | P4 | 300 | registry.py |
| `resistancemap/baselines/deep_static.py` | 10 | P4 | 380 (MLP+CNN1D+FT-Transformer) | registry.py |
| `resistancemap/baselines/sequence.py` | 10 | P4 | 280 (LSTM+GRU) | registry.py |
| `resistancemap/baselines/graph.py` | 10 | P4 | 240 (GraphSAGE + no-graph MLP) | registry.py |
| `resistancemap/baselines/mortfm/__init__.py` | 10 | P4 | 30 | registry.py |
| `resistancemap/baselines/mortfm/ablations.py` | 10 | P4 | 380 | registry.py, model.py |
| `resistancemap/evaluation/model_comparison.py` | 10 | P4 | 220 | — |
| `scripts/mortfm/10_run_baselines.py` | 10 | P4 | 280 | registry.py, all baseline modules |
| `tests/mortfm/test_baseline_registry.py` | 10 | P4 | 220 | registry.py |
| `resistancemap/governance/claim_matrix.py` | 14 | P4 | 800 (15-row CLAIM_GATE_REGISTRY) | evidence.py |
| `resistancemap/governance/imwg_validator.py` | 14 | P4 | 320 | claim_matrix.py |
| `tests/governance/test_claim_matrix.py` | 14 | P4 | 380 | claim_matrix.py |
| `resistancemap/reproducibility/schemas.py` | 15 | P5 | 280 (RunManifest, ClaimGate, MetricEntry, DataProvenance) | claim_matrix.py |
| `resistancemap/reproducibility/run_manifest.py` | 15 | P5 | 300 | schemas.py |
| `resistancemap/reproducibility/bundle.py` | 15 | P5 | 240 | schemas.py |
| `tests/reproducibility/test_manifest.py` | 15 | P5 | 240 | run_manifest.py |
| `tests/reproducibility/test_bundle.py` | 15 | P5 | 180 | bundle.py |
| `resistancemap/agents/mortfm/base.py` | 12 | P5 | 220 | — |
| `resistancemap/agents/mortfm/scientific_dag.py` | 12 | P5 | 280 | base.py |
| `resistancemap/agents/mortfm/{inventory,harmonization,cohort,leakage,preprocess,foundation,graph_pretrain,baselines,trajectory,survival,pathway,counterfactual,ablation,external,calibration,claim_critic,figures,tables}.py` | 12 | P5 | 17 × ~250 = 4250 | base.py, scientific_dag.py |
| `tests/agents/mortfm/test_scientific_dag.py` | 12 | P5 | 300 | scientific_dag.py |
| `tests/agents/mortfm/test_claim_critic_agent.py` | 12 | P5 | 240 | claim_matrix.py, scientific_dag.py |
| `resistancemap/infrastructure/task_backend.py` | 13 | P6 | 380 | — |
| `resistancemap/infrastructure/gpu_scheduler.py` | 13 | P6 | 140 | — |
| `resistancemap/infrastructure/hydra_runner.py` | 13 | P6 | 180 | task_backend.py |
| `resistancemap/infrastructure/optuna_runner.py` | 13 | P6 | 200 | task_backend.py |
| `tests/infrastructure/test_task_backend.py` | 13 | P6 | 260 | task_backend.py |
| `resistancemap/visualization/__init__.py` | 11 | P6 | 200 (errors + load_artifact + stamp_figure) | reproducibility/schemas.py |
| `resistancemap/visualization/latent.py` | 11 | P6 | 280 | __init__.py |
| `resistancemap/visualization/trajectory.py` | 11 | P6 | 320 | __init__.py |
| `resistancemap/visualization/waddington.py` | 11 | P6 | 380 | __init__.py |
| `resistancemap/visualization/{landscape_metrics,attribution,survival,baselines,figure1_architecture}.py` | 11 | P6 | 5 × ~250 = 1250 | __init__.py |
| `scripts/mortfm/11_generate_figures.py` | 11 | P6 | 220 | visualization/*.py |

### 4b. Modified files

| Path | Phase | Priority | Change scope | Depends on |
|---|---|---|---|---|
| `resistancemap/mortfm/trainer.py` | 0 | P0 | Rename class → `LegacyMORTFMTrainer`; add `MORTFMTrainer = LegacyMORTFMTrainer` alias + DeprecationWarning; delete trajectory truncation block (lines 287-296) in P1 | — |
| `resistancemap/mortfm/__init__.py` | 0 | P0 | Export `LegacyMORTFMTrainer`, `CanonicalMORTFMTrainer` | canonical_trainer.py |
| `resistancemap/training/loss_router.py` | 0 | P0 | + `validator_fn` kwarg; + `_mortbatch_validator` | — |
| `tests/mortfm/test_mortfm_trainer_smoke.py` → `test_legacy_trainer_smoke.py` | 0 | P0 | Rename verbatim; docstring update only | — |
| `resistancemap/mortfm/schemas.py` | 1/6/3 | P1 | + `future_snapshot_batch`, `delta_t`, `has_future_snapshot`, `treatment_window`, `basin_label`, `DrugBatch` (P3) | — |
| `resistancemap/data/mortfm_dataset.py` | 1/2/3 | P1 | Rewrite `mort_collate` lines 235-246 (future_snapshot_batch); wire 4 new loaders; `_stack_drug` → DrugBatch | feature_registry.py, drug_ontology.py |
| `resistancemap/data/trajectory_pair_builder.py` | 1 | P1 | + `TemporalPairProvenance` populated for every pair | schemas.py |
| `resistancemap/mortfm/trajectory/graph_energy_sde.py` | 7 | P1 | Accept external `t_grid` kwarg; expose `.t_grid`; `integration_time` defaults to 12.0 (months) | grid.py |
| `resistancemap/mortfm/survival/competing_risk_head.py` | 7 | P1 | Accept `t_grid` kwarg; store as buffer | grid.py |
| `resistancemap/mortfm/model.py` | 7/4/5 | P1 | Build canonical grid; replace `LatentToGraphProjector` with `PatientConditionedGraphEncoder`; instantiate V2 heads; extend forward return dict to 19 keys | grid.py, graph_encoder.py, all V2 heads |
| `resistancemap/data/scatac_loader.py` | 2 | P1 | Rewrite to emit `ModalityTensor` | feature_registry.py |
| `resistancemap/data/methylation_loader.py` | 2 | P1 | Rewrite to emit `ModalityTensor` | feature_registry.py |
| `resistancemap/data/histone_ptm_loader.py` | 2 | P1 | Rewrite to emit `ModalityTensor` | feature_registry.py |
| `resistancemap/data/phosphoproteomics_loader.py` | 2 | P1 | Rewrite to emit `ModalityTensor` | feature_registry.py |
| `resistancemap/data/drug_target_loader.py` | 3 | P1 | Replace ad-hoc loader with `drug_ontology.MMDrugOntology` lookups | drug_ontology.py |
| `tests/mortfm/test_schemas.py` | 2/1 | P1 | + `test_mortbatch_composes_all_eight_modalities`, `test_missing_modality_masked_not_zero_filled` | schemas.py |
| `resistancemap/landscape/landscape_regularizers.py` | 6 | P2 | + `lens_smoothness_penalty`, `lens_basin_separation_penalty` adapters | — |
| `resistancemap/mortfm/trajectory/graph_projector.py` | 4 | P2 | Deprecate (keep for checkpoint compat; emit DeprecationWarning) | graph_encoder.py |
| `resistancemap/mortfm/causal/intervention_graph.py` | 8 | P3 | Remove `abs(hash(eid))` basis (B4); use deterministic per-edge tensor seeded from `graph_builder` | graph_builder.py |
| `resistancemap/mortfm/causal/edge_evidence_report.py` | 8 | P3 | Replace hard 3-AND with K-of-N from ClaimMatrix | claim_matrix.py |
| `resistancemap/data/perturbation_loader.py` | 8 | P3 | + `PerturbationRecord`, `load_l1000_signatures`, `load_prism_viability`, MM cell-line filtering | — |
| `resistancemap/interpretability/counterfactual_engine.py` | 8 | P3 | Move to `resistancemap.legacy.interpretability_v6.counterfactual_engine` + DeprecationWarning shim at old path | — |
| `scripts/mortfm/08_eval_causal_evidence_v2.py` | 8 | P3 | Point shim to the new `scripts/mortfm_causal_evidence_report_v2.py` | — |
| `resistancemap/baselines/__init__.py` | 10 | P4 | Re-export `BaselineRegistry` + new baseline modules | registry.py |
| `resistancemap/mortfm/acceptance_gate.py` | 14 | P4 | Consume `ClaimGate` from `governance/claim_matrix.py`; remove implicit research/strong thresholds | claim_matrix.py |
| `resistancemap/mortfm/registries/claim_gate_registry.py` | 14 | P4 | Enforce `forbidden_endpoint_types` at runtime (not documentation) | claim_matrix.py |
| `resistancemap/agents/__init__.py` | 12 | P5 | Export `mortfm` sub-package | agents/mortfm/scientific_dag.py |
| `paper/generate_paper_figures.py` | 11 | P6 | Deprecate; replace with `scripts/mortfm/11_generate_figures.py` calling `visualization/*` | visualization/*.py |
| `scripts/mortfm/05_train_stage_E.py` and 06/07 | 0 | P0 | Swap `MORTFMTrainer(...)` → `CanonicalMORTFMTrainer(...)` | canonical_trainer.py |

**Total new LOC estimate:** ~18,500 across ~80 new files (the largest single
chunk is Phase 12 — 17 agents × ~250 LOC = 4250 LOC — and the next is
Phase 14 claim matrix at 800 LOC).
**Total modified LOC estimate:** ~3,000 across ~25 existing files.

---

## 5. Falsification framework — uniqueness brief

This is the one-page argument we will defend to reviewers. Each comparator
below is named in either a Wave-1 doc, in the user-provided context, or in
project memory; no SOTA is invented here.

### What ResistanceMap genuinely owns (and SOTA does not):

The differentiator is **not** a metric beat. Memory makes this explicit:
v11.5 ties mmSYGNAL on PFS C-index, v12 fell below v11.5, trivial
per-drug-mean wins drug-MSE (`project_resistancemap_floor_evidence.md`).
The differentiator is the **closed-loop falsification framework**, which
is the composition of four artefacts that no comparator system co-locates:

1. **F1–F10 strict gates** (`project_v10_*_outcome.md` sprint records) —
   pre-registered binary gates with structural-fail reporting (Sprint 7:
   F8 STRICT FAIL 0/16 cells, *expected* at N_paired = 29 per spec §1
   identifiability limit). Comparators report metrics; we report
   gates + identifiability ceilings.
2. **15-row claim-gate matrix** anchored to IMWG 2016 (PMID 27210871),
   MAIA (31141632), CASSIOPEIA (31171419) (Phase 14 §2). Every claim has
   a `blocked_endpoints` list enforced at runtime — e.g., GDSC IC50 is
   explicitly blocked for `resistance_emergence` because no published
   GDSC → IMWG mapping exists (Phase 14 §3, hard constraint 3).
3. **ClaimCriticAgent at DAG layer L9** (Phase 12 §12.4) that refuses to
   let `FigureGenerationAgent` advance if any upstream claim is refuted
   *or unsupported*. The v12-audit finding "v12 doesn't beat v11.5"
   (`project_v12_refactor_audit_outcome.md`) would have been auto-caught.
4. **RunManifest + reproducibility bundle** (Phase 15 §15.5) that makes
   `run_manifest.json` the *only* allowed metric source for figures — closes
   the `np.random` fabrication risk
   (`project_figure_fabrication_risk.md`, B11 above).

### Side-by-side vs. named comparators

| System | What it does | What it does **not** do that we do |
|---|---|---|
| **MOFA+** (`integration_baselines.py:178`) | Linear factor model across modalities | No survival head; no SDE on latent; no claim-gate matrix; no IMWG enforcement |
| **TrajectoryNet** (PMID 34337419, phase 10 §7) | Dynamic-OT continuous normalising flow on single-cell trajectories | Single-modality; no clinical-cohort endpoint adjudication; no per-claim evidence-channel gating |
| **mmSYGNAL** (web tool; **no canonical PMID** retrievable per phase 10 §7) | TF/miRNA systems-genetics MM tool | No counterfactual simulator; no K-of-N evidence aggregation; no run-manifest provenance |
| **TRACERx Lung ctDNA** (PMID 37055640, phase 10 §7) | Phylogenetic relapse prediction from longitudinal liquid biopsy in NSCLC | Different cancer; no SDE; no claim-gate matrix; no MM-specific basin labelling |
| **iMLGAM (2025)** | **0 PubMed hits as of 2026-05-20** (phase 10 §7) — unverified citation; not a defensible comparator until disambiguated | — |
| **PERCEPTION** (Nat Cancer 2024, `perception_baseline.py:208`) | Cell-line-to-patient transfer for drug response | No SDE; no Waddington landscape; no causal evidence module; no claim-gate matrix |
| **CPA / chemCPA** (`cpa_baseline.py:184`) | Compositional perturbation autoencoder | Perturbation generative; no survival; no longitudinal patient supervision |
| **scVI** (`integration_baselines.py:484`) | Single-cell latent embedding | No graph; no survival; no clinical-endpoint adjudication |
| **CellRank 2** (`integration_baselines.py:327`) | Fate mapping on single-cell trajectories | Single-cell only; no clinical cohort gating |
| **PathCNN** (PMID 34252964, phase 10 §7) | Pathway-CNN multi-omics survival | No SDE/landscape; no counterfactual; no IMWG-anchored gate matrix |

**Defensible "uniqueness" statement (the one to put in the abstract):**
ResistanceMap/MORT-FM is the first MM-specific multi-omics foundation
stack to combine a Waddington-regularised graph-conditioned SDE with a
run-time-enforced 15-row IMWG claim-gate matrix and a ClaimCriticAgent
that gates figure generation on K-of-N causal-evidence aggregation. We
make no metric-beat claim against MOFA+, mmSYGNAL, or PERCEPTION; we
claim a strictly stronger *governance* surface. Reviewers who accept
this framing will accept the contribution; reviewers who demand a
headline metric beat will reject it. The project memory is explicit that
the metric-beat framing is unwinnable, and pretending otherwise is the
fastest way to a desk-reject.

---

## 6. Evidence-gating cross-product

Each row in the 15-row Phase 14 claim matrix needs one or more evidence
channels. The cross-product below shows which Wave-1 phase delivers which
channel, and therefore which gates are achievable as soon as which tier
lands. A blank cell means the phase does not contribute to that gate.

| Claim domain (Phase 14 §3) | Min Patients | Achievable after | Evidence channels delivered by phases |
|---|---|---|---|
| technical_pipeline (R1) | 10 cell lines | **P0** | Phase 0 (canonical_trainer exit code 0); Phase 10 (pipeline_artifact_present); Phase 15 (run_manifest documents train/val split) |
| static_drug_response (R2) | 100 cell lines | P0 + P4 (baselines) | Phase 10 (spearman, baseline comparison); Phase 5 (DrugRiskHeadV2 spearman) |
| hematologic_specimen_drug_response (R3) | 100 specimens | P0 + P4 | Phase 2 (BeatAML loader); Phase 10 (transfer gain CI); Phase 14 (validator enforces label) |
| single_cell_state (R4) | 50 donors | P2 (Phase 9) | Phase 9 (modality ablation); Phase 6 (pseudotime marker correlation via basin_label); Phase 10 (macro_f1 baseline) |
| multiomic_foundation (R5) | 200 patients | **P2** | Phase 9 (held-out recon, cross-modal imputation, modality_dropout); Phase 2 (3 modalities live); Phase 6 (latent geometry separability via basin labels) |
| epigenetic_plasticity (R6) | 150 paired RNA + ATAC/meth | P2 + P3 | Phase 2 (ATAC/meth loaders); Phase 1 (trajectory shift under treatment via latent_future_state_loss); Phase 8 (perturbation_consistency_check via perturbation_loader upgrade) |
| sequence_aware (R7) | 50 sequences | P2 (Phase 3) | Phase 3 (ChEMBL coverage); Phase 4 (uniprot_loader); Phase 5 (DrugRiskHeadV2 embedding ablation) |
| pathway_context (R8) | 200 patients/cell lines | P2 (Phase 4) | Phase 4 (graph_smoothness, pathway_sparsity, drug_target_consistency losses); Phase 5 (PathwayRouteHeadV2); Phase 8 (top-k edge stability) |
| drug_target_mechanism (R9) | 200 patients/cell lines | P3 (Phase 8) | Phase 3 (ChEMBL coverage); Phase 8 (CRISPR essential overlap via perturbation_loader); Phase 5 (drug-target attribution direction via DrugRiskHeadV2) |
| **survival_prediction (R10)** | 200 patients, 50 events, **6mo follow-up, IMWG-consistent labels** | **P4 (Phase 14)** | Phase 7 (C-index lower CI via hitting_time_nll + lens_survival_loss); Phase 7 (Brier via integrated_brier_score, ECE via calibration_slope_intercept); Phase 14 (IMWG validator) |
| longitudinal_trajectory (R11) | 100 paired, ≥60d real calendar time | **P1 (Phase 1)** | Phase 1 (latent_future_state_loss; pseudotime substitution forbidden via has_future_snapshot semantics); Phase 7 (calendar-time mapped to ODE grid) |
| **resistance_emergence (R12)** | 200 patients, 80 resistance events, **12mo follow-up, IMWG PD** | **P4** | Phase 7 (time-to-resistance AUC); Phase 14 (IMWG PD adjudication; blocks OS; blocks GDSC IC50); Phase 6 (basin transition under treatment) |
| **causal_mechanism (R13)** | 200 patients + N≥20 perturbation, 50 events, 12mo follow-up | **P3 (Phase 8) + P4** | Phase 8 (5-of-5 channel evidence); Phase 5 (CounterfactualSimulator delta_basin_probs); Phase 4 (no-graph ablation via Phase 10 graph baselines); Phase 14 (negative control pathway) |
| **patient_level_clinical_prediction (R14)** | **300 patients + ext ≥100, 80 events, 18mo external follow-up** | **P5 (Phase 15) + P4 + P3** | Phase 14 (5-of-5); Phase 12 (claim-critic report); Phase 15 (external cohort provenance); Phase 7 (ECE ≤ 0.10) |
| external_validation (R15) | 100 ext, 30 events, 6mo holdout | **P5** | Phase 15 (bundle provenance); Phase 14 (validator: same weights, no re-training) |

**Reading the table:** with P0 + P1 + P2 landed, rows 1–6 are achievable.
With P3 + P4 landed, rows 7–10 + 12–13 are achievable. **Rows 11 and 14
are the bottleneck**: R11 (longitudinal_trajectory) needs N_paired ≥ 100
and *real* calendar time, which contradicts the memory finding that
N_paired = 29 today (`project_v10_sprint7_outcome.md`); R14 needs N ≥ 300
+ external ≥ 100 + 18-month follow-up, which is a data-acquisition
problem, not a code problem. Both should be flagged in the limitations
section of the paper, not promised as a v19 deliverable.

---

## 7. Risk register (honest)

| # | Risk | Likelihood | Severity | Owner phase | Mitigation |
|---|---|---|---|---|---|
| R1 | **N_paired = 29 identifiability ceiling** (memory: `project_v10_sprint7_outcome.md`) prevents Phase 1 trajectory loss from passing Sprint-7-style strict gates | High | High | 1, 6 | Report as identifiability-limit STRICT FAIL (consistent with spec §1); flag in paper limitations; do NOT promise R11 (longitudinal_trajectory) in v19 |
| R2 | **Encoder dim mismatch when activating ATAC/methylation** — registry feature universes (~50k peaks, ~450k CpGs) blow up the input projection layer; encoder LayerNorm chosen over BatchNorm but recon-loss normaliser per-modality is missing | High | Medium | 2, 9 | Phase 2 §2.8: encoder `d_token` is already uniform; Phase 9 §6 explicitly adds per-modality first-epoch median normaliser; add regression test in Phase 9 that absent modality has zero grad to its encoder |
| R3 | **ChEMBL API caching latency** — Phase 3 §2.1 says 19 drugs are static-baked, but Phase 4 graph builder calls `get_mechanism` per drug on each build → first build of full STRING + Reactome graph could take >30 min and is rate-limited | Medium | Medium | 3, 4 | Cache all ChEMBL responses to a content-addressed Parquet under `data/cache/chembl_v34/`; `graph_builder` reads from cache, refreshes only on explicit `--refresh-chembl` |
| R4 | **HGT memory blow-up on full STRING + Reactome graph** — STRING v12 human has ~6M edges at threshold 700; Reactome adds ~50k pathway memberships; with `d_model=64` and per-head attention this can OOM on a single A100 | High | High | 4 | Phase 4 §4.3 already specifies `max_degree_percentile=0.99` hub removal; add neighbour sampling (`NeighborLoader` from PyG) for training; cap n_pathways at 500 in `mortfm_full_multiomic.yaml` |
| R5 | **IMWG label adjudication labor cost** — Phase 14 §3 requires every PFS/OS/TT2L event in rows 10–15 to be IMWG-2016-consistent; MMRF CoMMpass IA22 may have labels under multiple criteria versions | High | High | 14 | Phase 14 §3 hard constraint 1: labels under EBMT 1998 are *not accepted*; build a per-patient adjudication metadata column; document any cohort that fails this as "excluded" with reason, not silently coerced |
| R6 | **sklearn-vs-pytorch baseline determinism** — Phase 10 §8 `test_seed_determinism` requires re-running a baseline with same seed reproduces metric within 1e-9; sklearn ensembles are deterministic, but PyTorch RNG state (CUDA dropout) is not by default | Medium | Medium | 10 | Use `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`; for non-deterministic CUDA ops (e.g., scatter-add) fall back to CPU baseline evaluation for the deterministic-seed test only |
| R7 | **Fabrication-firewall false positives on legitimate runs** — Phase 11 §0 raises `MissingArtifactError` when artifact's `metadata.run_id` doesn't match; if a user re-saves a checkpoint without re-stamping metadata, every figure regression-tests as `*_UNVERIFIED.png` | Medium | Low | 11, 15 | Phase 15 `run_manifest.write` is the canonical stamper; add a `scripts/mortfm/util_restamp_checkpoint.py` for legitimate metadata refresh that is *not* implicit in any other tool |
| R8 | **Phase 4 → Phase 5 head-dim coupling** — V2 heads consume `graph_emb` of dim `d_model`; if Phase 4 lands with `d_model=64` and an existing checkpoint has `d_model=16` (old `LatentToGraphProjector` output), load_state_dict fails silently on the new head | Medium | High | 4, 5 | `mortfm/model.py` `__init__` already gates instantiation on `config.d_graph`; require a one-time checkpoint migration script `scripts/mortfm/util_migrate_checkpoint_v18_to_v19.py` that re-initialises the graph branch with `strict=False`; emit warning |
| R9 | **K-of-N evidence gate too permissive** — Phase 8 reformulates `causal_mechanism_gate` from hard AND-3 to K-of-N (B13 fix). If K is mis-set the gate becomes weaker than v18 | Medium | Medium | 8 | Phase 14 §3 row 13 (causal_mechanism) explicitly requires **5-of-5** (no K-of-N), preserving v18 strictness; K-of-N is only for rows with `is_required=False` channels |
| R10 | **TaskBackend RayBackend instability** — Phase 13 §13.2 imports ray lazily, but the GPU scheduler context-manager (`acquire(n=1)`) holds the lock across `await` boundaries; can deadlock if a task crashes inside the `with` block | Low | High | 13 | Phase 13 §13.3 GPUScheduler uses `Condition`, but `acquire` should be wrapped in `try/finally` with explicit `notify_all()` (already in spec); add chaos test that kills a task mid-acquire |
| R11 | **Phase 12 agent DAG hash-chain break under partial re-run** — if a user re-runs only SurvivalAgent, downstream verification hashes from CalibrationAgent/AblationAgent become stale; the orchestrator does not currently detect this | Medium | Medium | 12 | `Orchestrator.run` already chains hashes; Phase 15 adds `agent_hashes` to RunManifest; add explicit "stale hash" detection that recomputes downstream agents automatically or refuses to advance L9 |
| R12 | **Phase 11 PHATE/UMAP backend missing** — Phase 11 §2.2 prefers PHATE if installed, falls back to UMAP then PCA; if neither is installed, figures fall back to PCA silently and the title attribution may be wrong | Low | Low | 11 | Phase 11 §2.2 explicitly says "axis labels reflect whichever reducer was actually used"; add a runtime guard that raises if `reducer="phate"` is requested and `phate` package is absent (no silent fallback when user explicitly asked) |

---

## 8. Wave-2 implementation status

**Verified at write-time** by `ls` against the working tree:

| Wave-2 artefact | Path | Status (2026-05-20) |
|---|---|---|
| `canonical_trainer.py` | `ResistanceMap/resistancemap/mortfm/canonical_trainer.py` | **does not exist** — in flight (per task brief, parallel agent) |
| `baselines/registry.py` | `ResistanceMap/resistancemap/baselines/registry.py` | **does not exist** — in flight |
| Existing baselines directory | `ResistanceMap/resistancemap/baselines/` | confirmed: `classical_baselines.py`, `cpa_baseline.py`, `integration_baselines.py`, `perception_baseline.py`, `mortfm/` (empty), `__init__.py` |
| Existing mortfm trainer (legacy) | `ResistanceMap/resistancemap/mortfm/trainer.py` | confirmed present; bugs B1, B2, B14, B15 verified at lines 227, 289, schemas.py:464-496 + loss_router.py exist but uncalled, 299-312 |
| Existing intervention_graph hash bug | `ResistanceMap/resistancemap/mortfm/causal/intervention_graph.py:105` | confirmed: `abs(hash(eid))` at line 105, `cos(... * 0.01)` at line 107 |
| Existing `_stack_drug` 3-scalar packing | `ResistanceMap/resistancemap/data/mortfm_dataset.py:113-200` and `mort_collate` future_state fallback at 235-246 | confirmed both |

**Implication:** the roadmap above must treat Phases 0 and 10 as *unstarted*
when scheduling resource allocation. As soon as the parallel agent's
`canonical_trainer.py` lands, this section should be re-verified and the
Phase-0 row in §3 marked as in-progress. The Phase 10 registry is a
prerequisite for Phase 14 enforcement (R10 survival_prediction validator
needs `BaselineSuiteAgent` for the "MORT-FM beats clinical-only Cox"
evidence channel).

---

## 9. Cut-list — what to NOT do for v19

Cutting these defers labor without giving up any v19 claim. Each cut is
justified against the Phase 14 claim matrix: if a feature is not on the
critical path for *any* claim row we are willing to assert, it can be
deferred. All cuts are recoverable in v20+.

| Cut | Rationale | Re-evaluate when |
|---|---|---|
| **Phase 13 SLURM backend** | LocalAsync + Ray cover laptop and single-node multi-GPU; SLURM is needed only for cluster fan-out of Phase 12 AblationAgent at full scale. ICML / Cell Systems submission can run on a single node with Ray. | When external cohort N ≥ 300 (R14) becomes the bottleneck |
| **Phase 11 Figure 1 (BioRender / Figma architecture diagram)** | Phase 11 §spec lists 9 figures; Figure 1 is the conceptual architecture and can be a placeholder hand-drawn diagram for v19. Code-generated figures (Figures 2–9) are the load-bearing fabrication-firewall test. | Camera-ready / post-acceptance |
| **Phase 2 histone-PTM modality activation** | CPTAC histone-PTM 6-mark panel is the lowest-N modality in the multi-omics stack; if patient overlap with MMRF + paired follow-up is < 30 patients, the modality contributes < 5% to Phase 9 cross-modal recon; deferring leaves us with RNA + proteomics + ATAC + methylation + phosphoproteomics, which is still 5 modalities and satisfies `multiomic_foundation` (R5) "3 modalities" threshold by a wide margin | When a real MM histone-PTM cohort of N ≥ 100 lands |
| **Phase 8 L1000 Connectivity Map loader** | Phase 8 §2 lists L1000 alongside DepMap CRISPR / Perturb-seq / PRISM; the K-of-N gate for R13 (causal_mechanism) requires 5-of-5 channels but the 5 channels are (CRISPR, drug-target, Reactome, external cohort, perturb-seq), not L1000. L1000 is *aspirational* not required. | When Phase 8 ClaimMatrix is extended to 6-channel for v20 |
| **Phase 12 PaperTableAgent** | The 17-agent DAG includes both `FigureGenerationAgent` and `PaperTableAgent` at L10; tables can be produced by a simple Pandas dump from RunManifest until the agent infra is mature. Avoid building two L10 agents simultaneously. | When figures are stable and we have time for table polish |
| **Phase 6 3D landscape plot (`plot_potential_3d`)** | Phase 6 §6.6 specifies both `plot_potential_2d` and `_3d`; the 2D contour plot already conveys basin topology and barrier height. 3D adds aesthetic load with no new claim. | Camera-ready / supplementary |
| **Phase 3 BCMA bispecifics + CAR-T entries** | Phase 3 §2.2 includes teclistamab, elranatamab, ide-cel, cilta-cel (4 of 19); MMRF CoMMpass IA22 enrollment predates most of these (FDA approvals 2022). They have zero patients in the cohort. Keeping them in the manifest is fine (data is correct); building DrugBatch tests against them is wasted labor for v19. | When a post-2022 enrollment cohort lands (e.g., MMRF CoMMpass IA23+) |
| **Phase 4 `perturbed_by` (DepMap CRISPR co-essentiality) edge type** | Phase 4 §4.2 includes 5 edge types; the 5th (`perturbed_by`) is optional. PPI + regulates + targets + member_of are sufficient for the HGT to outperform `mlp_no_graph` ablation, which is what the Phase 14 R8 (pathway_context) gate requires | When CRISPR co-essentiality becomes a separate evidence channel in R13 K-of-N |

**Cuts that are NOT acceptable** (preserve in v19):

- Phase 0 (P0 blocker for everything).
- Phase 1 + Phase 7 (without them, every survival and trajectory claim
  is unevaluable, see B2/B3/B7).
- Phase 14 IMWG validator (without it, R10 / R12 / R14 cannot pass
  honest review even if metrics are good).
- Phase 11 fabrication firewall on Figures 2–9 (closes B11 — the
  highest-reputation-risk bug in the codebase).
- Phase 15 RunManifest (sole legitimate metric source per Phase 11
  contract).

---

## 10. Three falsification criteria that flip this roadmap's verdict

A roadmap is itself a claim. These three observations would invalidate
the priority ordering above:

1. **If Phase 0 lands and `test_canonical_no_legacy_call` (Phase 0 §5.1)
   passes but stage-F C-index *drops* below the v18 baseline** — then
   the canonical `CompetingRiskHead` is computing a different loss
   surface than the legacy `TimeToResistanceHead`, and Phase 7 grid
   alignment needs to land in the *same* sprint as Phase 0, not in
   tier P1. Roadmap revision: merge Phase 0 + Phase 7 into P0.
2. **If Phase 2 lands with all 4 stub loaders rewritten and Stage A
   per-modality recon loss *fails to decrease* over the first epoch on
   real data** (Phase 9 §9.8 DoD criterion 2) — then the multi-omic
   foundation premise (R5) is structurally broken: the decoders we are
   adding don't fit the data. Roadmap revision: pause Phase 9, run a
   dedicated single-modality decoder ablation per modality before
   continuing.
3. **If Phase 14 IMWG validator rejects > 50% of MMRF CoMMpass IA22
   PFS events** (because labels were adjudicated under non-IMWG
   criteria) — then the patient_level_clinical_prediction (R14) and
   resistance_emergence (R12) gates are *unachievable* on the cohort
   we have, regardless of code quality. Roadmap revision: re-scope the
   paper to a R5 + R8 + R11 contribution (foundation +
   pathway-context + trajectory) and explicitly drop the survival
   chapter, OR acquire a second cohort with documented IMWG
   adjudication before paper submission.

---

## 11. Open questions for the user (decisions needed before P0 starts)

These cannot be inferred from the Wave-1 docs. The user must answer them
in writing in the next sprint planning doc, or the roadmap stalls.

1. **Target venue.** ICML / Cell Systems / Nat Methods have different
   bars for the falsification framing in §5. Naming the venue lets us
   tune the abstract and which claim rows we promise.
2. **Cohort decision for R14.** MMRF CoMMpass IA22 is the obvious training
   cohort; for the external N ≥ 100 we need either (a) an IA23+ holdout, or
   (b) Beat AML 1.0 cross-disease (which `project_v10_sprint6_outcome.md`
   already shows fails F3 — that's a feature, not a bug, for an honest
   falsification paper, but we need to commit before paper-writing).
3. **GPU budget.** Phase 4 HGT on full STRING+Reactome graph is
   memory-bound; budget determines whether neighbour-sampling
   (R4 mitigation) is required or whether we can train full-graph.
4. **Strict-mode FeatureRegistry default.** Phase 2 §2.3 declares the
   registry refuses to silently coerce unknown IDs; default `strict=False`
   today. For paper-grade runs we should set `strict=True` — confirm.
