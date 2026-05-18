# MORT-FM (v15) — Architecture

> **What is this document?** The single source of truth for the v15
> Multi-Omic Resistance Trajectory Foundation Model added in the
> `v15-mort-fm` branch. For the v14 ResistanceMap static cell-line pipeline,
> see `ARCHITECTURE.md`. The two pipelines share a package root but **do not
> share trained checkpoints, configs, or claims**.

## 0. Scope and non-claims

MORT-FM is intended to learn the mapping:

```
patient/cell molecular snapshot at t_0
    + drug exposure
    + biological graph (PPI / GRN / drug-target)
        ⟶ future resistance state (which attractor basin)
           time-to-resistance (survival)
           trajectory distribution (Wasserstein-matched future state)
           pathway route (which proteins/edges mediate the transition)
           drug-specific risk
           uncertainty (posterior over paths)
           counterfactual intervention ranking
```

**As of this commit, MORT-FM is a code release only.** No claims about
patient-level time-to-resistance prediction are permitted until the model has
been trained against a real :class:`ResistanceOutcome` cohort with
`N_patients ≥ min_patient_n_for_survival_claim` (200 by default) and
externally validated. See `docs/MORTFM_LIMITATIONS.md`.

## 1. Module map

```
resistancemap/
├── mortfm/                              # v15 namespace
│   ├── schemas.py                       # ModalityTensor, PatientCellSnapshot,
│   │                                    # ResistanceOutcome, TemporalTrainingPair,
│   │                                    # MORTBatch, DrugContext, FoundationState,
│   │                                    # TrajectoryPrediction, MORTFMConfig
│   ├── model.py                         # End-to-end MORTFM (composition of below)
│   ├── data_module.py                   # patient-disjoint split + DataLoader factory
│   └── trainer.py                       # staged trainer (A..H)
│
├── data/
│   ├── trajectory_pair_builder.py       # snapshots + outcomes -> TemporalTrainingPair
│   ├── mortfm_dataset.py                # MORTFMDataset + mort_collate
│   ├── clinical_outcome_loader.py       # MMRF -> ResistanceOutcome adapter
│   ├── scrna_loader.py                  # .h5ad -> ModalityTensor (RNA)
│   ├── scatac_loader.py                 # .h5ad -> ModalityTensor (ATAC)
│   ├── methylation_loader.py            # CpG matrices
│   ├── histone_ptm_loader.py            # HDAC-mark matrices
│   ├── proteomics_loader.py             # Bulk / RPPA / DIA-MS
│   ├── phosphoproteomics_loader.py      # Signaling activity
│   ├── cite_seq_loader.py               # Paired RNA + ADT
│   ├── drug_ontology.py                 # MM-relevant drug classes + targets
│   ├── treatment_exposure.py            # MMRF treatment regimens -> DrugContext
│   ├── drug_target_loader.py            # DrugBank / DGIdb edges
│   ├── perturbation_loader.py           # DepMap CRISPR + Perturb-seq
│   └── mmrf_loader.py                   # (v14, unchanged) MMRF parser
│
├── models/
│   ├── encoders/
│   │   ├── rna_encoder.py               # NB likelihood, library-size norm
│   │   ├── atac_encoder.py              # Bernoulli on binarised peaks
│   │   ├── methylation_encoder.py       # Beta / M-value
│   │   ├── histone_ptm_encoder.py       # HDAC-mark grouped features
│   │   ├── proteomic_encoder.py         # Gaussian + optional protein attention
│   │   ├── phosphoproteomic_encoder.py  # Sign-preserving Gaussian
│   │   ├── clinical_encoder.py          # Mixed + missing-mask
│   │   ├── drug_encoder.py              # identity + class + targets + dose
│   │   └── modality_tokenizer.py        # routes MORTBatch -> token tensor
│   │
│   ├── foundation_fusion.py             # tokenised cross-modal transformer (CLS)
│   │
│   ├── dynamics/
│   │   ├── graph_conditioned_drift.py   # f_theta(z, t, drug, graph)
│   │   ├── treatment_conditioned_dynamics.py
│   │   ├── neural_ode.py                # torchdiffeq + RK4 fallback
│   │   ├── neural_sde.py                # -grad U + g, with diffusion
│   │   ├── jump_sde.py                  # + compound Poisson jumps
│   │   ├── trajectory_sampler.py        # unified ODE/SDE/Jump-SDE API
│   │   └── temporal_decoder.py          # latent(t) -> per-modality token
│   │
│   └── heads/
│       ├── resistance_state_head.py     # CE over {sensitive, persister, MRD, resistant}
│       ├── time_to_resistance_head.py   # Cox or discrete-time survival
│       ├── trajectory_head.py           # Future-state distribution (mean, log_var)
│       ├── pathway_route_head.py        # protein + edge + pathway attribution
│       ├── drug_risk_head.py            # per-drug risk under each candidate
│       ├── uncertainty_head.py          # Evidential (Sensoy / Amini)
│       └── counterfactual_head.py       # rank node/drug interventions
│
├── landscape/
│   ├── potential.py                     # WaddingtonPotential (mixture + MLP residual)
│   ├── attractor_basin.py               # nearest / gradient-descent assignment
│   ├── barrier_estimation.py            # linear-path barrier heights
│   ├── landscape_regularizers.py        # smoothness, basin sep, drug barrier
│   └── waddington_visualizer.py         # 2-D contour plot
│
└── training/
    └── mortfm_losses.py                 # composable losses + assemble_total_loss
```

## 2. Forward pass

```
MORTBatch
  └── ModalityTokenizer (per-modality encoders)
        └── tokens (N, K, d_token), presence_mask (N, K)
              └── MultiOmicFoundationFusion (cross-modal Transformer + CLS)
                    └── FoundationState.z0 (N, d_latent), modality_gates (N, K)
                          └── TrajectorySampler (Neural ODE / SDE / Jump-SDE)
                                └── z_path (T, N, d_latent), z_samples (S, T, N, d)
                                      └── Heads:
                                          - ResistanceStateHead → state_logits
                                          - TimeToResistanceHead → hazard / survival_curve
                                          - TrajectoryDistributionHead → (mean, log_var)
                                          - PathwayRouteHead → protein/edge/pathway scores
                                          - DrugSpecificTrajectoryRiskHead → per-drug risk
                                          - EvidentialUncertaintyHead → Dirichlet alpha
                                          - CounterfactualInterventionHead → ranked dict[]
```

## 3. Training stages

| Stage | What it trains                                   | Active losses                              |
|------:|---------------------------------------------------|-----------------------------------------------|
| A     | Per-modality encoders                             | recon                                         |
| B     | Cross-modal foundation fusion                     | recon + masked + contrastive                  |
| C     | Static drug-response (CCLE/GDSC/PRISM)            | drug                                          |
| D     | Patient-domain adaptation                         | contrastive (cross-batch MMD planned)         |
| E     | Longitudinal trajectory training                  | trajectory + landscape                        |
| F     | Survival / time-to-resistance                     | survival + state                              |
| G     | Pathway + counterfactual validation               | pathway                                       |
| H     | Calibration (temperature scaling)                 | survival + calibration                        |

Each stage is opt-in via `MORTFMTrainer.fit_stage("X", n_epochs=N)`. The
canonical curriculum is `("A", "B", "C", "E", "F")`.

## 4. Honest-failure design

* **Patient-disjoint splits**: `data_module.split_patients` raises if any
  patient ID appears in more than one split.
* **No fabricated labels**: every loss validates its required label tensors
  upfront (`MORTBatch.validate_for_loss`). Missing labels raise rather than
  silently zero-fill.
* **No fabricated data**: every loader raises `FileNotFoundError` when the
  underlying file is absent. Subsampling is deterministic (head-of-list), never
  random — both for reproducibility and to satisfy the fabrication-sentinel
  hook.
* **Pre-training claim guard**: `MORTFMConfig.min_patient_n_for_survival_claim`
  is read by the Claim Critic Agent before any survival-curve figure can be
  exported.

## 5. Differences from v14 (ResistanceMap)

| Aspect                          | v14 ResistanceMap                            | v15 MORT-FM                                  |
|---------------------------------|----------------------------------------------|----------------------------------------------|
| Latent state                    | Proteome→Epigenome VAE                       | Multi-modal foundation fusion (CLS token)    |
| Trajectory                      | ODE with nominal 3/6/12-mo horizons          | True longitudinal (X_t, X_{t+Δ}) pairs       |
| Time supervision                | Pseudotime + score range [0, 1]              | Real calendar time + survival labels         |
| Drug representation             | One-hot over 11 GDSC drugs                   | Identity + class + targets + dose + combo    |
| Patient identity                | Implicit (cell line)                         | First-class (`patient_id` everywhere)        |
| Waddington landscape            | UMAP layout + drug-resistance head           | Learned potential U_θ(z, drug) + grad field  |
| Outputs                         | IC50 + resistance score                      | 7 heads: state, time, traj, path, risk, U, CF |
| Stochasticity                   | Deterministic ODE                            | Neural SDE (+ optional jump-SDE)             |

## 6. Smoke test

```
python scripts/mortfm_smoke_test.py --config configs/mortfm_debug.yaml
```

Runs 4 stages on a 10-patient structural cohort in ~3 seconds on CPU.
Confirms all wiring; produces no claimable metrics.

## 7. Test suite

```
python -m pytest tests/mortfm/ -q --override-ini="addopts="
```

Currently 38 tests covering schemas, temporal pairing, patient-disjoint
splits, every loss, the trainer (one epoch per representative stage), and
checkpoint round-trip.
