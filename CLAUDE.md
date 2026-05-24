# ResistanceMap / MORT-FM Pipeline

Pharmacogenomic ML pipeline for drug resistance prediction in multiple myeloma (MM). 461 Python files, ~127K lines. Two co-existing architectures: legacy v6 pipeline and MORT-FM (v15+) foundation model. Current branch: `v19`.

## Quick Commands

```bash
# Run full pipeline (agentic DAG mode — default)
python ResistanceMap/main.py --config configs/h100.yaml

# Sequential mode (legacy)
python ResistanceMap/main.py --sequential

# Single stage
python ResistanceMap/main.py --stage vae_pretrain

# Evaluation governance (Tier A/B/C/D)
python ResistanceMap/main.py --evaluate

# v10 falsification sprints
python ResistanceMap/main.py --v10-sprint all

# Run tests
cd ResistanceMap && python -m pytest tests/ -v --cov=resistancemap

# Canonical MORT-FM workflow (numbered steps)
bash scripts/mortfm/00_download.sh   # through 09_validate.sh
```

## Project Layout

```
ResistanceMap/
├── main.py                          # Thin wrapper -> resistancemap.main:main()
├── resistancemap/
│   ├── main.py                      # Primary entry: CLI, 11 sequential stages, agentic DAG, eval governance
│   ├── config.py                    # 9 dataclasses (DataConfig, VAEConfig, TrajectoryConfig, etc.)
│   ├── _facts.py                    # Single source of truth for pipeline constants (CI-enforced)
│   ├── v10_runner.py                # Subprocess orchestrator for v10 sprint scripts
│   ├── agents/                      # DAG-based agent orchestration (BaseAgent, Orchestrator, 10 specialized)
│   ├── agentops/                    # 3-layer observability (Tracer, Evaluator, Optimizer, Dashboard)
│   ├── models/                      # Neural architectures (28 files, ~14K lines)
│   │   ├── vae.py                   # ProteomeToEpigenomeVAE (cyclical KL, FiLM, domain-adversarial)
│   │   ├── trajectory.py            # ChromatinODE, MemoryStabilityScorer, TrajectoryForecaster, NeuralJumpSDE
│   │   ├── protein_network.py       # ESM2Embedder + PPIGraphNetwork (GAT) + PerDrugHead
│   │   ├── fusion.py                # CrossModalFusionNet (cross-attention/tensor/concat)
│   │   ├── foundation_fusion.py     # MultiOmicFoundationFusion (MORT-FM tokenized transformer)
│   │   ├── per_drug_head.py         # Per-drug regression with inverse-variance weighting
│   │   ├── identifiable_ode.py      # StructuredBiologicalODE (8 programs, sparse A, Lyapunov)
│   │   ├── causal_fusion.py         # CausalFusionNetwork (SCM layers, do-calculus)
│   │   ├── disentangled_vae.py      # Beta-TCVAE with anchored biological dimensions
│   │   ├── cell_state_plasticity.py # Multi-scale: SDE + double-well + mutation ratchet
│   │   ├── clinical_anchored_ode.py # FiLM-conditioned Neural ODE with treatment sequences
│   │   ├── bistability.py           # Jacobian eigenvalue analysis + DualHeadGNN
│   │   ├── cell_communication.py    # Heterogeneous GNN (ligand-receptor)
│   │   ├── clonal_heterogeneity.py  # Replicator dynamics ODE + adaptive therapy
│   │   ├── domain_adaptation.py     # DANN + Wasserstein-GP cell-line-to-patient transfer
│   │   ├── rl_treatment_optimization.py  # IQL/CQL offline RL (21 treatments, 8-component reward)
│   │   ├── encoders/                # 8 modality-specific encoders (RNA/NB, ATAC/Bernoulli, etc.)
│   │   └── rl/ope.py               # Off-policy evaluation safety gate (WIS, doubly-robust)
│   ├── mortfm/                      # MORT-FM foundation model namespace (59 files)
│   │   ├── schemas.py               # Core types: MORTBatch, MORTFMConfig, ModalityTensor, etc.
│   │   ├── model.py                 # MORTFM nn.Module (dual path: forward_legacy vs forward canonical)
│   │   ├── trainer.py               # MORTFMTrainer (8-stage curriculum A-H)
│   │   ├── canonical_trainer.py     # CanonicalMORTFMTrainer (LENS stages E/F/G/H only)
│   │   ├── data_module.py           # Patient-disjoint splits + DataLoader factory
│   │   ├── acceptance_gate.py       # 4-tier gate (debug/technical/research/strong)
│   │   ├── blocks/                  # Block A (DepMap), B (BeatAML), C (scRNA), D (ESM-2+graph), E (integrated)
│   │   ├── causal/                  # InterventionGraph (SHA-256 deterministic), CounterfactualRunner, evidence joiners
│   │   ├── drug_response/           # DrugConditionedSpecimenResponseHead, PerDrugPairSampler
│   │   ├── graph/                   # IDHarmonizer (STRING ENSP <-> UniProt <-> HGNC bridge)
│   │   ├── longitudinal/            # PatientTimeline, LongitudinalPair, cohort registry (7 cohorts), temporal splitter
│   │   ├── registries/              # ArtifactRegistry, 12 claim levels, 9 endpoint types, 8 feature spaces
│   │   ├── single_cell/             # SupCon loss, pseudotime ordinal loss, stratified sampler
│   │   ├── survival/                # CompetingRiskHead, ClinicalOnlyCox, RegularizedLENS, calibration, metrics
│   │   └── trajectory/              # GraphEnergyResistanceSDE, WaddingtonPotential, ResistanceBasin, HittingTime, canonical grid
│   ├── evaluation/                  # Multi-agent tiered evaluation (47 files, ~15K lines)
│   │   ├── base.py                  # EvalAgent, EvalFinding, EvalVerdict
│   │   ├── orchestrator.py          # EvalOrchestrator (tier-ordered DAG, hard-stop on Tier A FAIL)
│   │   ├── charter.py               # 3 testable claims: state, when, pathway
│   │   ├── rubric.py                # Weighted scoring with YAML I/O
│   │   ├── metrics/                 # calibration, survival, forecasting, pathway_stability, per_drug, shift
│   │   ├── tier_a/                  # DataAdequacy, MeasurementIntegration, BiasFairness (hard-stop gates)
│   │   ├── tier_b/                  # ArchitectureAuditor, CellStateTrajectory, PathwayPPI
│   │   ├── tier_c/                  # ForecastingUncertainty, BaselineAdversary, ClinicalSafety
│   │   ├── tier_d/                  # PrincipalIntegrator (TRL 1-9 grading)
│   │   ├── baselines/               # 4 honest baselines (clinical-only, transcriptome-only, LOCF, simple-survival)
│   │   ├── leakage.py               # Patient-split + temporal leakage detection
│   │   └── benchmarking.py          # 11-method SOTA comparison (DrugCell, DeepCDR, etc.)
│   ├── clinical/                    # MM-specific clinical modules (10 files, ~6.3K lines)
│   │   ├── treatment_conditioned.py # DragonNet + FiLM ODE + IPTW + Actor-Critic
│   │   ├── dynamic_treatment_regimes.py  # Q-learning + CRN + AIPW + feasibility mask
│   │   ├── relapse_prediction.py    # Neural ODE competing risks (5 causes)
│   │   ├── early_relapse_detection.py  # M-protein kinetics + CUSUM + EWMA + cytokines
│   │   ├── mrd_prediction.py        # VAE + cross-attention + trajectory transformer
│   │   ├── emd_prediction.py        # 5-layer EMD risk + Weibull + CNS-MM flagging
│   │   ├── immunotherapy_response.py  # MAML few-shot + antigen loss LSTM + sequential planning
│   │   └── smm_progression.py       # PANGEA + clonal expansion + Weibull cure model
│   ├── interpretability/            # Explanations (6 files, ~5.4K lines)
│   │   ├── pathway_attribution.py   # Temporal Integrated Gradients + BH-FDR
│   │   ├── counterfactual_engine.py # Counterfactual trajectories (DEPRECATED latent-clamping)
│   │   ├── temporal_explanations.py # Critical windows, trajectory decomposition, phase transitions
│   │   ├── uncertainty_explanations.py  # 4-source UQ decomposition + conformal calibration
│   │   └── visualization.py         # Waddington 3D, Circos, attribution heatmaps (Nature style)
│   ├── theory/                      # Mathematical foundations (3 files, ~2K lines)
│   │   ├── identifiability_proof.py # FIM, Lie derivatives, empirical convergence
│   │   └── stability_analysis.py    # Fixed points, bifurcations, Lyapunov exponents, basins
│   ├── data/                        # Data loading and preprocessing (33 files, ~9.4K lines)
│   │   ├── loaders.py               # MultiOmicsDataset, CCLE/STRING/GDSC/scRNA/MMRF/CRISPR loaders
│   │   ├── preprocessors.py         # harmonize_omics(), build_train_val_test_splits()
│   │   ├── mmrf_loader.py           # Full MMRF CoMMpass dataset (GDC + MMRF-RG sources)
│   │   ├── mortfm_dataset.py        # MORTFMDataset + mort_collate (TemporalTrainingPair -> MORTBatch)
│   │   ├── drug_ontology.py         # 21 MM drugs with targets, classes, synonyms
│   │   ├── identifier_mapping.py    # HGNC <-> Ensembl <-> UniProt <-> STRING <-> Reactome <-> ChEMBL
│   │   ├── protein_sequence_cache.py  # UniProt FASTA + ESM-2 embedding cache
│   │   └── [modality loaders]       # scrna, scatac, cite_seq, proteomics, phospho, methylation, histone_ptm
│   ├── training/                    # Loss functions and trainer (6 files, ~3.1K lines)
│   │   ├── losses.py                # v6 composite: 8 task losses + Kendall auto-weighting
│   │   ├── mortfm_losses.py         # MORT-FM composable losses (recon, contrastive, drug, trajectory, survival)
│   │   ├── canonical_mortfm_losses.py  # LENS canonical losses (survival NLL+Brier, hitting NLL, basin CE, SDE path)
│   │   ├── loss_router.py           # Disciplined composition with per-step audit trail
│   │   └── trainer.py               # AdamW + cosine annealing + EMA + mixed precision
│   ├── infrastructure/              # Deployment infrastructure (8 files, ~4.5K lines)
│   │   ├── missing_modality.py      # PoE-VAE, Flex-MoE, training-time masking, selective prediction
│   │   ├── uncertainty_quantification.py  # 6-layer UQ (ensemble, evidential, Bayesian ODE, conformal, selective, clinical)
│   │   ├── continual_learning.py    # Joint longitudinal-survival + EWC + Neural ODE Kalman + streaming Cox
│   │   ├── scrna_integration.py     # Foundation model encoder + scArches + Harmony + gated MIL + MILO
│   │   ├── imaging_radiomics.py     # Tile encoder + TransMIL + 4-layer hierarchical cross-attention
│   │   └── multi_site_harmonization.py  # ComBat-seq + Harmony + FederatedBatchNorm + LoRA
│   ├── baselines/                   # Baseline models (9 files, ~3.6K lines)
│   │   ├── registry.py              # BaselineModel protocol + factory registry (10 models registered)
│   │   ├── classical_baselines.py   # Ridge, RF, ElasticNet, XGBoost, SVM, KNN, trivial mean
│   │   ├── static_ml.py             # Phase 10 registry: elasticnet, ridge, rf, xgboost, lightgbm
│   │   ├── deep_static.py           # MLP, CNN-feature-order, FT-Transformer-lite
│   │   ├── sequence.py              # LSTM/GRU visit-level baselines
│   │   ├── cpa_baseline.py          # CPA/chemCPA perturbation autoencoder
│   │   ├── perception_baseline.py   # PERCEPTION (Sinha et al. 2024) transfer baseline
│   │   └── integration_baselines.py # MOFA+, CellRank, scVI
│   ├── landscape/                   # Resistance landscape (12 files, ~1.9K lines)
│   │   ├── potential.py             # WaddingtonPotential (Gaussian mixture + MLP residual)
│   │   ├── scalar_potential.py      # DSM + PPI Tikhonov training
│   │   ├── neural_ode_flow.py       # Adaptive Dopri5 flow over learned potential
│   │   ├── rwr_propagation.py       # RWR + NPI z-score on STRING
│   │   ├── gat_propagation.py       # Multi-alpha GAT ensemble
│   │   ├── hbayes_outer.py          # numpyro hierarchical Bayes over cytogenetic strata
│   │   └── predictor.py             # L5 ResistanceLandscape + TargetRanker + EvidentialHead
│   ├── verification/                # Zero-trust + guardrails (4 files, ~1.6K lines)
│   ├── observability/               # JSONL recorder + v11 stage decorator (3 files)
│   ├── data_quality/                # DataProfiler with SOTA benchmarks (2 files)
│   ├── mcp_integration/             # v7-disarmed stubs (use evidence-curator subagent instead)
│   ├── legacy/                      # Archived v15 heads + dynamics (19 files, forward_legacy only)
│   ├── experiments/                 # Ablation harness + statistical testing (3 files)
│   ├── inference/                   # FastAPI server + L0-L5 pipeline (3 files)
│   ├── governance/                  # Claim gates + endpoint validator (3 files)
│   ├── latent_compute/              # Async GPU job scheduler (2 files)
│   └── utils/                       # Checkpoint, logging, metrics, fairness (4 files)
├── data/                            # Top-level data helpers
│   ├── clinical_labels.py           # 4 resistance states (Sensitive/PrimaryRefractory/Acquired/MDR)
│   ├── graph_preprocessing.py       # STRINGGraphBuilder + edge type categorization
│   └── velocity_integration.py      # RNA velocity + CytoTRACE + discrete resistance states
├── configs/                         # YAML configs + ablation generator
│   ├── h100.yaml, default.yaml, full.yaml
│   ├── mortfm.yaml, mortfm_gates.yaml
│   └── ablation_configs.py          # 12 ablation conditions (component/modality/scale)
├── scripts/                         # Pipeline scripts (~138 files)
│   ├── mortfm/                      # Canonical workflow steps 00-09
│   ├── v10/                         # Falsification sprints 1-7 (38 scripts)
│   ├── v11/                         # Geneformer upgrade + F1-F8 replication (11 scripts)
│   ├── v12/                         # Head-to-head comparisons (4 scripts)
│   ├── mortfm_*.py                  # Data ingestion, training, evaluation scripts (~50 files)
│   ├── train.py                     # L0-L5 pipeline orchestrator
│   └── run_baselines_real.py        # Real-data baseline runner
├── tests/                           # 35 test files, ~200 test functions
│   ├── test_agents.py               # DAG, parallelism, verification chain
│   ├── test_ml_integration.py       # VAE, GRL, OT, fusion, ODE, evidential, pipeline
│   └── mortfm/                      # 23 MORT-FM tests (canonical forward, losses, supervision, schemas)
└── paper/v8_artifacts/              # Paper figure/table generation
```

## Architecture: Two Co-existing Pipelines

### Legacy v6 Pipeline (sequential stages)
```
L0_DataValidation -> L1_DataPrep -> L2_VAEPretrain || L2_ESM2Embed
-> L3_VAEFinetune -> L4_Trajectory -> L5_ProteinNet -> L6_Fusion
-> L7_Landscape -> L8_Validation
```
- 10 agents in a DAG with `asyncio.gather` parallelism per layer
- SHA-256 zero-trust verification chain between agents
- Checkpoint chain: data_ready -> vae_pretrained -> ... -> pipeline_validated

### MORT-FM Foundation Model (v15+, canonical v18)
```
8 modality encoders -> ModalityTokenizer -> MultiOmicFoundationFusion (Transformer)
-> GraphEnergyResistanceSDE (Euler-Maruyama on Waddington potential)
-> CompetingRiskHead + ResistanceBasin + HittingTime
```
- Stages A-D: pretraining (recon, contrastive, drug response)
- Stages E-H: LENS canonical surface (trajectory, survival, basin, calibration)
- 4-tier acceptance gate (debug/technical/research/strong)
- Strict supervision policy: E>=80%, F>=90%, G>=70% coverage required

### Data Sources
- CCLE proteomics/epigenomics, STRING v12 PPI (7,853 MM-subnet nodes)
- GDSC + CTRPv2 + PRISM drug sensitivity (11 target drugs)
- MMRF CoMMpass (N=787 baseline, 29 paired patients)
- scRNA-seq: GSE124310 (26K cells), GSE271107 (126K cells)
- DepMap CRISPR gene effect, BeatAML 1.0
- UniProt/ESM-2 protein sequences, Reactome, ChEMBL

## Key Invariants (MUST preserve)

1. **No synthetic/fabricated data**: Loaders raise FileNotFoundError; MORTBatch.validate_for_loss raises ValueError on missing supervision; np.random NEVER used for metrics
2. **Patient-disjoint splits**: All split functions enforce no patient overlap with runtime assertions
3. **OS is NOT a resistance endpoint**: Endpoint registry explicitly forbids OS for resistance_emergence claims
4. **Pseudotime is NOT calendar time**: disease_stage_pseudotime forbidden for trajectory/resistance/survival claims
5. **Per-drug metrics required alongside pooled**: No pooled AUROC without per-drug table
6. **Canonical time grid (v19)**: Single CanonicalTimeGridConfig shared by SDE and CompetingRiskHead
7. **k-of-N causal gate**: 2-of-3 evidence channels (CRISPR, drug-target, Reactome) for causal_mechanism claim
8. **Inverse-variance loss weighting**: Computed from train split only; prevents HDAC gradient dominance
9. **Tier A FAIL hard-stops everything**: Evaluation governance caps TRL at 2 if Tier A fails
10. **MCP connectors are v7-disarmed stubs**: Use the evidence-curator Claude subagent for live queries

## Configuration

Primary config: `configs/h100.yaml` loaded into `ResistanceMapConfig` (9 nested dataclasses).

Key parameters:
- `data.target_drugs`: 11 MM drugs (Bortezomib, Lenalidomide, Panobinostat, ...)
- `vae.latent_dim`: 64 (resistance state embedding)
- `trajectory.ode_solver`: "euler" (dopri5 available but can underflow dt during training)
- `protein_net.esm2_model`: "facebook/esm2_t33_650M_UR50D" (1280-dim output)
- `fusion.fusion_type`: "cross_attention" (also: tensor, concat)

MORT-FM config: `configs/mortfm.yaml` -> `MORTFMConfig` dataclass in `mortfm/schemas.py`.

## Testing

```bash
cd ResistanceMap
python -m pytest tests/ -v --cov=resistancemap
python -m pytest tests/mortfm/ -v              # MORT-FM specific
python -m pytest tests/ -m "not slow"           # skip GPU/slow tests
```

Markers: `unit`, `integration`, `slow`, `gpu`. Coverage configured in `pyproject.toml`.

Key test files:
- `tests/mortfm/test_canonical_forward.py` — v18 LENS dict vs legacy vs foundation
- `tests/mortfm/test_canonical_loss_router.py` — per-stage dispatch + n_supervised contracts
- `tests/mortfm/test_phase1_trajectory_target.py` — collate + latent loss (Bug B1 regression)
- `tests/mortfm/test_phase7_canonical_grid.py` — time grid + hitting NLL (Bug B19 regression)
- `tests/mortfm/test_stage_supervision_strict.py` — MissingSupervisionError enforcement
- `tests/mortfm/test_intervention_graph_determinism.py` — cross-process SHA-256 determinism

## v10 Falsification Framework (F1-F10)

The v10 sprint pipeline runs via `--v10-sprint all` and tests 10 falsification hypotheses:

| Sprint | Tests | Key Result |
|--------|-------|------------|
| S1 | F4 (DSM+PPI), F5 (scalar potential) | F4 PASS (0.0146), F5 substitute fails |
| S2 | F1 (HBayes), F2 (strata), F5-paired | F1+F2 PASS, F5-paired refuted (expected) |
| S3 | F3 (RWR drug), F6 (propagation), F7 (PRISM) | F3 6/10 STRICT, F6 30σ PASS, F7 2/3 (Vorinostat structural fail) |
| S4 | F8 (mediation NIE), F9, F10 | F8+F9 PASS, F10 structural fail (tipping-point confirmed) |
| S5 | Mondrian JK+ conformal | 5/5 strata within ±3% — STRICT PASS |
| S6 | Cross-disease BeatAML | F3 REFUTED (MM proteasome is MM-specific, expected) |
| S7 | Diffusion posterior F8 | F8 STRICT FAIL 0/16 (identifiability-limit at N_paired=29) |

## Known Issues and Bugs

Named regression tests exist for bugs B1, B3, B4, B5, B6, B13, B14, B19.

- **B1**: TrajectoryAgent collate fails on fresh-run (future_snapshot_batch handling)
- **B19**: d_graph mismatch between SDE and projector at construction time
- **generate_paper_figures.py**: Most panels use `np.random` when data is empty — NEVER call `generate_all()` for real figures; use `scripts/run_real_figures.py` instead

## Important Conventions

- Drug sensitivity is z-score normalized on train split only (`_zscore_normalize_on_train`)
- Per-drug heads use independent `Linear(hidden, 1)` per drug (not shared output)
- Trajectory forecaster horizon labels (3/6/12 months) are **nominal indices, not calendar-calibrated**
- ESM-2 embeddings are cached via SHA-256 hash of protein names
- `_facts.py` is the single source of truth; `scripts/check_docs_consistent.py` enforces consistency
- All timestamp fields in longitudinal pairs must be real calendar days, never pseudotime
- `forward()` (canonical) returns LENS dict; `forward_legacy()` returns TrajectoryPrediction dataclass
- Claim levels are ordered: technical < static_drug_response < ... < causal_mechanism (12 levels)

## Evaluation Governance

Orthogonal to training. 10 agents across 4 tiers:

- **Tier A** (hard-stop gates): DataAdequacy, MeasurementIntegration, BiasFairness
- **Tier B** (scientific coherence): ArchitectureAuditor, CellStateTrajectory, PathwayPPI
- **Tier C** (prediction quality): ForecastingUncertainty, BaselineAdversary, ClinicalSafety
- **Tier D** (chair): PrincipalIntegrator — TRL 1-9 grading (max TRL 4 from eval alone)

TRL grading: Tier A FAIL -> TRL <= 2; Tier B FAIL -> TRL <= 3; no clinical utility -> TRL <= 6.

## Latest Run: r-2026-05-23-v19-fullrun

Full sequential pipeline on 1x H100 NVL 96GB, 31.8 min total.

### Stage Timings

| Stage | Time | Key Metric |
|-------|------|------------|
| data_validate | 0.0 min | All files found |
| data_prep | 1.5 min | 886 samples, 19177 proteins, 473860 PPI edges, 11 drugs |
| vae_pretrain | 1.4 min | best_val_loss=0.998, early-stop epoch 70/200 |
| vae_finetune | 0.4 min | best_val_loss=1.152, early-stop epoch 21/100 |
| trajectory_calibrate | 4.4 min | best_loss=0.0001, 692 valid samples |
| trajectory_forecast | 1.0 min | val_loss=0.0078, early-stop epoch 21/200 |
| protein_net_train | 15.2 min | val_loss=1.781 (19177 nodes, 929520 edges) |
| fusion_train | 7.6 min | val_loss=1.276, early-stop epoch 16/150 |
| landscape_train | 0.1 min | val_loss=1.345, early-stop epoch 31/100 |
| validate | 0.1 min | **test_mse=2.845**, 132 test samples |

### SOTA Benchmark (6 reimplemented models, same train/test splits)

| Rank | Model | Test MSE | Mean Spearman | Architecture |
|------|-------|----------|---------------|--------------|
| 1 | TCRP | 2.779 | 0.289 | Transfer learning + per-drug heads |
| 2 | DeepCDR | 2.796 | 0.300 | Multi-omics MLP fusion |
| 3 | DrugCell | 2.843 | 0.236 | Pathway-guided visible NN |
| 4 | **ResistanceMap** | **2.845** | **0.113** | **VAE+ODE+GAT+CrossAttn fusion** |
| 5 | PRECISE | 2.847 | 0.282 | Domain-adaptation encoder |
| 6 | PaccMann | 2.852 | 0.202 | Transformer self-attention |
| 7 | GraphDRP | 2.855 | 0.223 | Bilinear cell-drug fusion |

All models within MSE 2.78-2.86 (predict-mean floor at 2.81). SOTA models achieve higher per-drug Spearman. Script: `scripts/sota_benchmark_and_visualizations.py`.

### Per-Drug Results (ResistanceMap)

| Drug | N | MSE | Spearman |
|------|---|-----|----------|
| Dinaciclib | 100 | 0.129 | 0.411 |
| Cyclophosphamide | 76 | 0.325 | 0.227 |
| Bortezomib | 102 | 0.887 | 0.206 |
| Venetoclax | 74 | 0.026 | 0.180 |
| Palbociclib | 81 | 0.201 | 0.163 |
| Panobinostat | 101 | 22.689 | -0.101 |

### Generated Figures

SOTA benchmark (`paper/v8_artifacts/sota_benchmark/`):
- `fig_a` — SOTA comparison bar chart with bootstrap CIs
- `fig_b` — Per-drug MSE heatmap (models x drugs)
- `fig_c` — Radar chart (1/MSE, Spearman, drug coverage)
- `fig_d` — PHATE embedding of 886 cell lines (sensitivity, stability, lineage)
- `fig_e` — PPI network subgraphs (top-20 attributed proteins, 4 drugs)
- `fig_f` — Cell-line x drug residual heatmap
- `fig_g` — Pred-vs-truth scatter (top 3 drugs by Spearman)
- `fig_h` — Pipeline architecture diagram
- `fig_i` — Training curves for all 6 SOTA models
- `fig_j` — Per-drug Spearman grouped bars (ResistanceMap vs SOTA)

Krishnaswamy visualizations (`paper/v8_artifacts/visualizations/`):
- `fig_latent_manifold` — PHATE/UMAP of VAE latents
- `fig_pseudotime_per_drug` — Per-drug pred-vs-truth colored by stability
- `fig_meld_density` — Drug-class density on manifold
- `fig_residuals_heatmap` — Per cell-line per drug residuals
- `fig_resistance_landscape` — Kernel-smoothed resistance on PHATE

### Integration Audit

| Rank | Method | Test MSE |
|------|--------|---------|
| 1 | Late fusion (avg) | 2.365 |
| 2 | ResistanceMap CrossModalFusionNet | 2.373 |
| 3 | MOFA+-like factors -> Ridge | 2.444 |
| 4 | Naive concat -> Ridge | 2.458 |
| 5 | PCA per-modality -> Ridge | 3.737 |

## Run Ledger

Every numeric claim in README/ARCHITECTURE must cite a row in `RUNS.md`. If a row has no W&B link or persistent log, its numbers cannot appear in docs. Use `release-bouncer` subagent before tagging.

## Dependencies

Python >= 3.11. Core: torch, numpy, pandas, scipy, PyYAML.
Optional: torch_geometric (GNN), torchdiffeq (ODE), transformers (ESM-2), numpyro+jax (HBayes), scanpy (scRNA), fastapi+uvicorn (serving), wandb, sklearn, xgboost, lightgbm.

All external dependencies have graceful fallbacks (ImportError guards).

---

# DETAILED PER-FILE REFERENCE (Appendices)

The sections below document every Python file in the repository line-by-line: every class, function, dataclass, constant, and data flow. Generated by 10 parallel analysis agents reading all 461 files (~127K lines).

---

## Appendix A: Core Entry Points, Config, Agents, AgentOps

# ResistanceMap Codebase Analysis — Part 1: Core Pipeline, Agents, AgentOps

BASE = `ResistanceMap/`

---

## 1. `main.py` (top-level entry)

**Full path:** `main.py`
**Lines:** 11
**Purpose:** Thin wrapper that delegates to `resistancemap.main:main()`.

### Imports
- `resistancemap.main.main`

### Logic
- If run as `__main__`, calls `main()`.

### Data flow
- In: CLI invocation
- Out: delegates entirely to `resistancemap/main.py`

---

## 2. `resistancemap/__init__.py`

**Full path:** `resistancemap/__init__.py`
**Lines:** 22
**Purpose:** Package init; declares version metadata and imports Phase 3 subpackages.

### Constants
- `__version__` = `"0.1.0"`
- `__author__` = `"Anthropic"`
- `__license__` = `"MIT"`

### Imports (subpackages)
- `resistancemap.interpretability`
- `resistancemap.training`
- `resistancemap.theory`
- `resistancemap.baselines` (aliased as `rm_baselines`)
- `resistancemap.experiments`
- `resistancemap.data` (aliased as `rm_data`)

---

## 3. `resistancemap/_facts.py`

**Full path:** `resistancemap/_facts.py`
**Lines:** 99
**Purpose:** Single source of truth for pipeline numbers, shapes, and topology. All documentation and assertions must reference this module. A CI check (`scripts/check_docs_consistent.py`) fails the build if docs drift from these values.

### Key constant: `FACTS` (dict)

#### `FACTS["ppi"]`
- `source`: "STRING v12"
- `raw_nodes`: 19,566
- `raw_edges`: 11,938,498
- `mm_subnet_nodes`: 7,853
- `mm_subnet_edges`: 460,212
- `edge_direction`: "undirected"

#### `FACTS["vae"]`
- `latent_dim`: 64
- `pretrain_epochs`: 200
- `finetune_epochs`: 100
- `pretrain_cohort`: "CCLE_pan_cancer"
- `finetune_cohort`: "CCLE_hematologic"

#### `FACTS["trajectory"]`
- `ode_solver.name`: "dopri5"
- `ode_solver.family`: "Dormand-Prince Runge-Kutta (4,5)" (NOT Adams-Bashforth)
- `ode_solver.source`: "torchdiffeq"
- `horizons_months`: [3, 6, 12]

#### `FACTS["protein_net"]`
- `conv`: "sage" (GraphSAGE with neighbour sampling)
- `n_layers`: 4
- `num_neighbors`: [25, 10, 10, 5]
- `in_dim`: 1280 (ESM-2 t33 output)
- `hidden_dim`: 256

#### `FACTS["fusion"]`
- `kind`: "cross_attention"
- `hidden_dim`: 256 (reconciled from 128)
- `n_heads`: 4

#### `FACTS["layers"]` (9 pipeline stages)
```
L0_DataValidation, L1_DataPrep, L2_VAEPretrain, L2_ESM2Embed,
L3_VAEFinetune, L4_Trajectory, L5_ProteinNet, L6_Fusion,
L7_Landscape, L8_Validation
```
Note: L2 has two parallel agents (VAEPretrain || ESM2Embed).

#### `FACTS["data_sources"]`
8 data sources with size_gb, public flag, and access requirements. Notable: MMRF_CoMMpass is 50 GB, requires dbGaP DAR.

#### `FACTS["verification"]`
- `label`: "Per-stage SHA256 content-hash chain for reproducibility"
- `nist_zta_compliant`: False (explicitly NOT claimed)

### Functions

#### `total_raw_data_gb() -> float`
Sums `size_gb` across all entries in `FACTS["data_sources"]`.

#### `public_only_data_gb() -> float`
Sums `size_gb` for entries where `public` is True.

---

## 4. `resistancemap/config.py`

**Full path:** `resistancemap/config.py`
**Lines:** 445
**Purpose:** Configuration dataclasses for every pipeline component. Loaded from YAML via `load_config()`.

### Imports
- `dataclasses` (dataclass, field)
- `pathlib.Path`
- `typing.Optional`
- `torch`
- `yaml`

### Dataclasses

#### `DataConfig`
Paths and parameters for data loading/preprocessing.

| Field | Type | Default | Notes |
|---|---|---|---|
| `ccle_proteomics_path` | Path | `data/raw/ccle_proteomics.csv` | |
| `ccle_epigenomics_dir` | Path | `data/raw/ccle_epigenomics/` | |
| `string_ppi_path` | Path | `data/raw/string_ppi.txt` | |
| `gdsc_path` | Path | `data/raw/gdsc_drug_sensitivity.csv` | |
| `ctrpv2_path` | Path | `data/raw/ctrpv2_drug_sensitivity.csv` | |
| `scrna_gse124310_path` | Path | `data/raw/gse124310.h5ad` | MM patient samples |
| `scrna_gse271107_path` | Path | `data/raw/gse271107.h5ad` | Lenalidomide response |
| `mmrf_commpass_dir` | Path | `data/raw/mmrf_commpass/` | Clinical + genomic |
| `scrna_summary_path` | Path | `checkpoints/scrna_summary.pt` | Pre-aggregated pseudobulk |
| `depmap_crispr_path` | Path | `data/raw/depmap/CRISPRGeneEffect.csv` | |
| `hmcl_keats_dir` | Path | `data/raw/hmcl_keats/` | |
| `prism_path` | Path | `data/raw/prism/secondary-screen-...csv` | |
| `use_esm2_sequences` | bool | False | Real ESM-2 forward pass toggle |
| `uniprot_cache_dir` | Path | `data/external/uniprot` | |
| `uniprot_bulk_fasta` | Optional[Path] | None | |
| `uniprot_allow_network` | bool | False | REST fallback off by default |
| `ccle_24q4_rna_path` | Optional[Path] | None | CCLE 24Q4 RNA expansion |
| `ccle_24q4_proteomics_path` | Optional[Path] | None | |
| `depmap_model_path` | Optional[Path] | None | 24Q4 Model.csv |
| `prism_24q2_path` | Optional[Path] | None | PRISM 24Q2 |
| `min_coverage` | float | 0.7 | Drop proteins missing >30% |
| `imputation` | str | "knn" | knn / median / zero |
| `normalization` | str | "quantile" | quantile / zscore / log2 |
| `ppi_confidence` | float | 0.7 | STRING combined score cutoff |
| `scrna_min_genes` | int | 200 | |
| `scrna_max_genes` | int | 8000 | |
| `scrna_min_counts` | int | 1000 | |
| `scrna_max_mt` | float | 0.2 | Max mitochondrial % |
| `test_fraction` | float | 0.15 | |
| `val_fraction` | float | 0.15 | |
| `random_seed` | int | 42 | |
| `target_drugs` | list[str] | 11 MM-relevant drugs | Bortezomib, Lenalidomide, Panobinostat, Vorinostat, Romidepsin, Venetoclax, Dinaciclib, Palbociclib, Doxorubicin, Etoposide, Cyclophosphamide |

#### `VAEConfig`
Hyperparameters for the Proteome-to-Epigenome conditional VAE.

| Field | Type | Default | Notes |
|---|---|---|---|
| `input_dim` | int | 8000 | # proteins |
| `epigenome_dim` | int | 50000 | # ATAC-seq peaks |
| `latent_dim` | int | 64 | Resistance state embedding |
| `encoder_hidden_dims` | list[int] | [2048, 1024, 512] | |
| `decoder_hidden_dims` | list[int] | [512, 1024, 2048] | |
| `dropout` | float | 0.1 | |
| `use_batch_norm` | bool | True | |
| `activation` | str | "gelu" | |
| `pretrain_epochs` | int | 200 | |
| `finetune_epochs` | int | 100 | |
| `pretrain_lr` | float | 1e-3 | |
| `finetune_lr` | float | 1e-4 | |
| `weight_decay` | float | 1e-5 | |
| `batch_size` | int | 512 | |
| `gradient_clip_norm` | float | 1.0 | |
| `kl_anneal_cycles` | int | 4 | Cyclical KL annealing |
| `kl_anneal_ratio` | float | 0.5 | |
| `kl_weight_max` | float | 1.0 | |
| `atac_weight` | float | 1.0 | Recon loss weight |
| `h3k4me3_weight` | float | 0.5 | |
| `h3k27me3_weight` | float | 0.5 | |
| `gradient_checkpointing` | bool | True | |
| `patience` | int | 20 | Early stopping |
| `min_delta` | float | 1e-4 | |
| `conditioning_dim` | int | 0 | FiLM conditioning (0=disabled) |
| `domain_adversarial` | bool | False | |
| `adversarial_weight` | float | 0.1 | |
| `adversarial_warmup_epochs` | int | 10 | |
| `use_stochastic_decoder` | bool | False | |
| `expanded_latent_dim` | Optional[int] | None | |

#### `TrajectoryConfig`
ODE-based resistance trajectory modeling hyperparameters.

| Field | Type | Default | Notes |
|---|---|---|---|
| `ode_solver` | str | "euler" | euler / dopri5 / rk4 |
| `ode_rtol` | float | 1e-5 | |
| `ode_atol` | float | 1e-7 | |
| `integration_time` | float | 100.0 | Arbitrary time units |
| `forecast_horizons` | list[int] | [3, 6, 12] | Months (nominal) |
| `n_perturbation` | int | 50 | MC samples |
| `calibration_lr` | float | 1e-3 | |
| `calibration_epochs` | int | 500 | |
| `calibration_batch_size` | int | 64 | |
| `score_range` | tuple | (0.0, 1.0) | |
| `reader_writer_proteins` | list[str] | 20 chromatin proteins | EZH2, KDM6A, KDM6B, KMT2A, KMT2D, DNMT1, DNMT3A, DNMT3B, TET1, TET2, HDAC1, HDAC2, KAT2A, KAT2B, EP300, BRD4, SMARCA4, ARID1A, SUZ12, EED |
| `use_sde` | bool | False | Neural Jump-SDE |
| `sde_drift_hidden` | int | 128 | |
| `sde_diffusion_hidden` | int | 64 | |
| `sde_jump_hidden` | int | 64 | |
| `sde_dt` | float | 0.01 | |
| `sde_n_mc_samples` | int | 50 | |
| `survival_calibrate` | bool | False | |

**Alias:** `StabilityConfig = TrajectoryConfig` (backward compatibility)

#### `ProteinNetConfig`
Protein network with ESM-2 embeddings.

| Field | Type | Default | Notes |
|---|---|---|---|
| `esm2_model` | str | "facebook/esm2_t33_650M_UR50D" | |
| `esm2_dim` | int | 1280 | ESM2-T33 output |
| `gnn_hidden` | int | 256 | |
| `gnn_layers` | int | 4 | |
| `gnn_heads` | int | 8 | |
| `gnn_dropout` | float | 0.2 | |
| `gnn_conv_type` | str | "gat" | gat / gcn / graphsage |
| `ppi_proteins` | int | 7853 | |
| `ppi_edges` | int | 460000 | |
| `dropedge_rate` | float | 0.1 | Anti-over-smoothing |
| `use_pairnorm` | bool | True | |
| `use_jumping_knowledge` | bool | True | |
| `jk_mode` | str | "cat" | cat / max / lstm |
| `use_graphmask` | bool | False | |
| `graphmask_sparsity_weight` | float | 0.01 | |
| `esm2_bottleneck_dim` | int | 256 | |
| `drug_embedding_dim` | int | 0 | 0 = disabled |
| `n_drug_types` | int | 11 | |
| `use_evidential_head` | bool | False | |
| `evidential_n_classes` | int | 3 | |
| `phospho_dim` | int | 0 | 0 = disabled |
| `string_debias_textmining` | bool | True | |
| `string_debias_threshold` | float | 0.7 | |

#### `FusionConfig`
Multi-modal fusion hyperparameters.

| Field | Type | Default |
|---|---|---|
| `hidden_dim` | int | 128 |
| `n_heads` | int | 4 |
| `dropout` | float | 0.2 |
| `fusion_type` | str | "cross_attention" |
| `fusion_lr` | float | 5e-4 |
| `fusion_epochs` | int | 150 |
| `weight_decay` | float | 1e-4 |
| `batch_correction` | bool | False |
| `n_batches` | int | 10 |
| `batch_embed_dim` | int | 16 |
| `handle_missing_modalities` | bool | True |

#### `LandscapeConfig`
Resistance landscape visualization.

| Field | Type | Default |
|---|---|---|
| `n_top_targets` | int | 20 |
| `confidence_threshold` | float | 0.8 |
| `visualization` | bool | True |
| `umap_n_neighbors` | int | 15 |
| `umap_min_dist` | float | 0.1 |
| `use_evidential` | bool | False |

#### `APIConfig`
| Field | Type | Default |
|---|---|---|
| `port` | int | 8000 |
| `workers` | int | 4 |
| `host` | str | "0.0.0.0" |

#### `HardwareConfig`
| Field | Type | Default |
|---|---|---|
| `device` | str | "cuda" |
| `dtype` | str | "bfloat16" |
| `compile` | bool | True |
| `compile_mode` | str | "reduce-overhead" |
| `pin_memory` | bool | True |
| `num_workers` | int | 8 |
| `prefetch_factor` | int | 4 |
| `distributed` | bool | False |
| `local_rank` | int | 0 |
| `world_size` | int | 1 |
| `deterministic` | bool | False |
| `seed` | int | 42 |

#### `EvaluationConfig`
Evaluation governance layer settings (orthogonal to training DAG).

| Field | Type | Default |
|---|---|---|
| `enabled` | bool | True |
| `log_root` | Path | "logs/evaluation" |
| `run_id` | Optional[str] | None |
| `rubric_path` | Optional[Path] | None |
| `tier_a_hard_stop` | bool | True |
| `skip_tier_b` | bool | False |
| `skip_tier_c` | bool | False |
| `skip_tier_d` | bool | False |

#### `ResistanceMapConfig`
Top-level container aggregating all sub-configs.

**Fields:** `data`, `vae`, `trajectory`, `protein_net`, `fusion`, `landscape`, `api`, `hardware`, `evaluation`, plus `checkpoint_dir` (Path), `log_dir` (Path), `wandb_project` (str), `resume_checkpoint` (Optional[Path]).

**Properties:**
- `device -> torch.device`: Returns CUDA device with local_rank or CPU.
- `amp_dtype -> torch.dtype`: Maps hardware.dtype string to torch dtype.

### Function: `load_config(path: Path) -> ResistanceMapConfig`
- If path doesn't exist, returns defaults.
- Reads YAML via `yaml.safe_load`.
- Maps nested YAML keys to dataclass fields by section name.
- Handles Path conversion, type coercion for float/int from strings.
- Returns fully populated `ResistanceMapConfig`.

---

## 5. `resistancemap/main.py`

**Full path:** `resistancemap/main.py`
**Lines:** 2171
**Purpose:** Primary entry point with CLI arg parsing, pipeline routing between agentic (DAG) and sequential modes, plus all sequential stage implementations.

### Imports
- Standard: `argparse`, `asyncio`, `logging`, `sys`, `time`, `pathlib.Path`
- External: `numpy`, `torch`, `torch.distributed`
- Internal: `resistancemap.config.{load_config, ResistanceMapConfig}`, `resistancemap.utils.checkpoint.CheckpointManager`

### Top-level functions

#### `build_agent_dag(tracer, evaluator, optimizer, guardrails) -> Orchestrator`
Constructs the 10-agent DAG with dependency edges.

**DAG topology:**
```
Layer 0: [DataValidation]
Layer 1: [DataPrep]
Layer 2: [VAEPretrain, ESM2Embed]   <- parallel
Layer 3: [VAEFinetune]
Layer 4: [Trajectory]
Layer 5: [ProteinNet]
Layer 6: [Fusion]
Layer 7: [Landscape]
Layer 8: [Validation]
```

Imports from `resistancemap.agents.orchestrator` and `resistancemap.agents.specialized`. Creates an `Orchestrator` with optional tracer/evaluator/optimizer/guardrails. Registers all 10 specialized agents.

---

#### `async run_agentic_pipeline(config: ResistanceMapConfig) -> dict`
Full DAG-based pipeline execution.

**Steps:**
1. Set up logging, CheckpointManager, deterministic mode (seeds, cudnn).
2. Initialize AgentOps (Tracer singleton, Evaluator, Optimizer, Dashboard).
3. Initialize ZeroTrustVerifier and GuardrailEngine.
4. Build agent DAG via `build_agent_dag()`, validate topology via `prepare_execution()`.
5. Log the execution plan (layers and agent names).
6. Initialize DataProfiler for data quality pre-check.
7. Initialize LatentComputeScheduler with GPU budget.
8. Execute DAG via `orchestrator.run(config)`.
9. Collect timing report, verification chain, results summary.
10. Export AgentOps dashboard as JSON.
11. Surface guardrail violations from the run.
12. Optionally run evaluation governance (Tier A/B/C/D) post-training.
13. Return dict with results, timing, verification chain, guardrail violations, governance report.

---

#### `_maybe_compile(model, config) -> Module`
Applies `torch.compile` if `config.hardware.compile` is True. Sets `_dynamo.config.suppress_errors = True` so Inductor failures fall back to eager mode rather than crashing.

#### `_maybe_distribute(model, config) -> Module`
Wraps model in `DistributedDataParallel` if `config.hardware.distributed` and `dist.is_initialized()`.

---

### Sequential Stage Functions

#### `validate_data_files(config, ckpt_mgr) -> Path`
Checks existence of CCLE proteomics, STRING PPI, GDSC or CTRPv2 (at least one), and scRNA-seq GSE124310. Raises `FileNotFoundError` with specific missing paths.

#### `prepare_data(config, ckpt_mgr) -> Path`
Loads all multi-omics datasets (proteomics, epigenomics, PPI, scRNA, MMRF, CRISPR), harmonizes via `harmonize_omics()`, builds train/val/test splits. Saves checkpoint "data_ready". Skips if checkpoint exists.

#### `pretrain_vae(config, ckpt_mgr) -> Path`
Pretrains `ProteomeToEpigenomeVAE` on pan-cancer CCLE data. Deep-copies config to set input_dim/epigenome_dim from actual data shape. Applies `_maybe_compile` and `_maybe_distribute`. Saves checkpoint "vae_pretrained".

#### `finetune_vae(config, ckpt_mgr) -> Path`
Fine-tunes VAE on hematological cell lines. Loads pretrained weights (stripping `_orig_mod.` prefix from compiled models). Saves checkpoint "vae_finetuned".

#### `calibrate_trajectory(config, ckpt_mgr) -> Path`
Calibrates `MemoryStabilityScorer` (ODE-based resistance trajectory). Requires data_ready and vae_finetuned checkpoints. Saves "stability_calibrated".

#### `train_trajectory_forecaster(config, ckpt_mgr) -> Path`
Trains `TrajectoryForecaster` on top of calibrated ODE.

**Key details:**
- Loads frozen VAE for latent extraction and frozen MemoryStabilityScorer.
- Transfers calibrated ODE weights from scorer to forecaster.
- Pre-computes VAE latents (N, 64) and reader/writer levels (N, 20) for all samples.
- Builds supervision targets from drug sensitivity z-scores: per-sample mean IC50 normalized to [0,1]. High resistance (1.0) = expect low stability; sensitive (0.0) = high stability.
- **Training:** Only trains `latent_to_params` (Linear(64->8)) and `horizon_scale` (scalar). ODE core stays frozen.
- Horizon weights: {3: 0.2, 6: 0.3, 12: 0.5}.
- Uses Euler solver during training (dopri5 can underflow dt).
- Early stopping with patience=20, max 200 epochs, batch_size=32.
- **LIMITATION (documented):** Supervision is contemporaneous snapshot, NOT longitudinal. "3/6/12-month" labels are nominal indices, not calendar months.
- Saves "trajectory_forecaster_trained".

#### `train_protein_network(config, ckpt_mgr) -> Path`
Trains protein interaction GNN on STRING PPI graph.

**Node features per protein:**
- With ESM-2: bottleneck(256) + VAE latent(64) + stability(1) = 321-dim
- Without ESM-2: abundance(1) + VAE latent(64) + stability(1) = 66-dim

**Key details:**
- Loads frozen VAE and stability scorer.
- Pre-computes all latents and stability scores.
- Optionally computes ESM-2 embeddings: checks for protein sequences, requires >=50% coverage, uses ESM2Embedder with batch_size=8 for memory safety, caches raw embeddings via SHA256 hash key. Applies ESM2Bottleneck (1280->256).
- Builds PPI edge index from STRING: maps protein pairs to indices, creates undirected edges. Falls back to self-loop k-NN graph if no protein-name overlap.
- Constructs `PPIGraphNetwork` GNN and `PerDrugHead` (per-drug prediction with inverse-variance weighting).
- Inner batch size = 4 (each sample materializes full PPI graph forward).
- Early stopping with patience=15, max 100 epochs.
- **P2.2 Attribution:** On best validation, computes per-(protein, drug) input-gradient L2-norm saliency across up to 64 validation samples. Top-K (K=20) ranked per drug. Persisted in checkpoint.
- Saves "protein_net_trained" with GNN state, pred head, attribution schema.

#### `train_fusion(config, ckpt_mgr) -> Path`
Trains multi-modal fusion combining four modalities.

**Key details:**
- Loads all frozen upstream models: VAE, MemoryStabilityScorer, TrajectoryForecaster, PPIGraphNetwork.
- Pre-computes all modality embeddings:
  - `epi_states` (N, 64): VAE latent mu
  - `traj_states` (N, 10): Concatenation of [initial_stab, stab@3m/6m/12m, trans@3m/6m/12m, a@3m, a@12m, r@12m]
  - `pnet_outputs` (N, gnn_hidden): GNN mean-pooled node embeddings
  - `stab_scores` (N, 1): Stability scores
- `traj_projector`: Linear(10->64) + GELU to project raw trajectory features.
- Modality dims: epigenetic=64, trajectory=64 (projected), protein_network=gnn_hidden, stability=1.
- `CrossModalFusionNet` with cross-attention.
- `PerDrugHead` with inverse-variance loss weighting (HDAC gradient poisoning mitigation).
- Batch size 64, early stopping patience=15, max `fusion_epochs` (150).
- Saves "fusion_trained" with fusion state dict, drug head, traj projector, pre-computed embeddings.

#### `train_landscape(config, ckpt_mgr) -> Path`
Builds resistance landscape predictor.

**Key details:**
- Loads pre-computed fused representations from fusion checkpoint.
- Rebuilds traj_projector (10->64) and CrossModalFusionNet (or legacy concat fallback).
- Pre-computes fused representations for all samples.
- `ResistanceLandscape` model: takes fusion_dim input, outputs drug resistance, state, and target predictions.
- Batch size 64, patience=15, max 100 epochs. Uses MSE loss on first-timepoint predictions vs drug sensitivity.
- Saves "landscape_trained".

#### `validate_pipeline(config, ckpt_mgr) -> Path`
End-to-end validation on held-out test data.

**Key details:**
- Rebuilds fusion + landscape models from checkpoints.
- Computes test fused representations.
- Evaluates landscape predictions: test_mse, per-drug breakdown (n_obs, MSE, MAE, Spearman).
- Writes `per_drug_metrics.csv` to log_dir.
- Runs baseline comparison via subprocess (`scripts/run_baselines_real.py`), reads `paper/tables/baseline_comparison.json`.
- **Actionability scoring (v8):** Per-drug classification:
  - "Actionable for screening": n_obs >= 30, Spearman >= 0.25 (better than random).
  - "Well-calibrated": also MSE < 1.0.
  - Drug class aggregation (Proteasome, IMiD, HDAC, BCL2, CDK, DNA-damage).
- Saves "pipeline_validated" with metrics + baseline + actionability summaries.

#### `serve_api(config, ckpt_mgr) -> None`
Launches FastAPI inference server via uvicorn. Creates app from `resistancemap.inference.api.create_app`.

---

### `STAGES` (Sequential stage registry)
Ordered list of (name, function) tuples:
```
data_validate, data_prep, vae_pretrain, vae_finetune,
trajectory_calibrate, trajectory_forecast, protein_net_train,
fusion_train, landscape_train, validate, serve
```

#### `run_sequential_pipeline(config, stage=None) -> None`
Executes pipeline sequentially. If `stage` is provided, runs only that stage. Otherwise runs all stages in order. Initializes distributed if needed, enables TF32.

---

### Evaluation Governance

#### `async run_evaluation_governance(config) -> None`
Orthogonal to training DAG. Registers 10 evaluation agents across 4 tiers:
- **Tier A (gates):** DataAdequacyAgent, MeasurementIntegrationAgent, BiasFairnessShiftAgent
- **Tier B (architecture):** ArchitectureAuditorAgent, CellStateTrajectoryAgent, PathwayPPIReasoningAgent
- **Tier C (prediction quality):** ForecastingUncertaintyAgent, BaselineAdversaryAgent, ClinicalTranslationSafetyAgent
- **Tier D (chair):** PrincipalIntegratorAgent

Tier A FAIL hard-stops downstream. Prints final report with TRL grade, per-tier verdicts, required changes.

---

### CLI

#### `parse_args() -> Namespace`
Arguments:
- `--config` (Path, default `configs/h100.yaml`)
- `--sequential` (flag)
- `--stage` (choices from STAGES)
- `--resume-from-latest` (flag)
- `--resume` (Path)
- `--port` (int)
- `--evaluate` (flag for governance mode)
- `--v10-sprint` (choices from v10_runner.VALID_STAGE_NAMES)

#### `main() -> None`
Entry point routing:
1. Parse args, load config.
2. Override port/resume if provided.
3. Initialize distributed, enable TF32.
4. If `--evaluate`: run evaluation governance, return.
5. If `--v10-sprint`: run v10 sprint orchestrator via `run_v10_e2e()`, exit with return code.
6. If `--stage`: run single stage (sequential).
7. If `--sequential`: run sequential pipeline.
8. Else (default): run agentic DAG pipeline, print summary.

---

### Helper Functions

#### `_init_distributed(config) -> None`
Initializes `torch.distributed` process group if `WORLD_SIZE > 1`. Sets local_rank, world_size, CUDA device.

---

## 6. `resistancemap/v10_runner.py`

**Full path:** `resistancemap/v10_runner.py`
**Lines:** 202
**Purpose:** V10 sprint orchestrator that runs the v10 paper-spec pipeline end-to-end via subprocess calls to `scripts/v10/s*.py` scripts.

### Imports
- `logging`, `shlex`, `subprocess`, `sys`, `time`
- `pathlib.Path`

### Constants

#### `ROOT`
`Path(__file__).resolve().parents[1]` -- the ResistanceMap directory root.

#### `V10_STAGES: list[tuple[str, list[list[str]]]]`
8 stage definitions, each mapping a semantic stage name to a list of script invocations:

1. **`landscape_dsm`** (Sprint 1): 5 scripts -- builds MMRF baseline matrix, encodes z64, builds PPI Laplacian, trains scalar potential (DSM + PPI Tikhonov), runs F4/F5 falsification gates.
2. **`hbayes_strata`** (Sprint 2): 8 scripts -- paired patients, encode paired, PPI Laplacian (v10s2 tag), scalar potential, falsification gates, F5 paired, HBayes inference, F1/F2 gates.
3. **`propagation_rwr`** (Sprint 3): 10 scripts -- PPI adjacency, DepMap haem subset, CRISPR oracle, drug seed propagation, F3/F6/F7 gates, v2/v3 strict hardening, v3 F7.
4. **`mediation_nie`** (Sprint 4): 7 scripts -- build outcome/treatment, build mediator, merge analysis table, NIE estimation, falsification neg control, sensitivity alt mediators, v2 focused neg control.
5. **`conformal_mondrian`** (Sprint 5): 1 script -- Mondrian jackknife+.
6. **`beataml_xdisease`** (Sprint 6): 4 scripts -- extract Beat AML, cross-disease F3, AML native, conformal Beat AML.
7. **`dps_singlesnapshot`** (Sprint 7): 1 script -- diffusion posterior.
8. **`fixes`**: 3 scripts -- F7 PRISM oracle, F10 richer mediator, F10 tipping point.

#### `VALID_STAGE_NAMES`
`["all", "landscape_dsm", "hbayes_strata", "propagation_rwr", "mediation_nie", "conformal_mondrian", "beataml_xdisease", "dps_singlesnapshot", "fixes"]`

### Functions

#### `_run_one_script(args: Sequence[str], log_path: Path) -> int`
Runs a single Python script via `subprocess.run` with unbuffered Python (`-u`), appends stdout+stderr to log_path. Returns exit code.

#### `run_v10_e2e(stage_name: str, log_dir: Path | None) -> int`
Main orchestrator. If `stage_name == "all"`, runs all stages. Otherwise runs the named stage only.

**Algorithm:**
- For each stage, truncates the log file, runs each script in order.
- If any script returns non-zero, aborts the stage and breaks.
- Prints a summary table: STAGE, EXIT, WALL, SCRIPTS_OK.
- Returns 0 on success, 1 if any stage failed.

#### `__main__` block
Argparse with `stage` (choices from VALID_STAGE_NAMES) and `--log-dir`.

---

## 7. `resistancemap/agents/__init__.py`

**Full path:** `resistancemap/agents/__init__.py`
**Lines:** 59
**Purpose:** Package init exporting all agent classes and orchestrator components.

### Exports (`__all__`)
- `AgentState`, `AgentResult`, `BaseAgent` (from `.base`)
- `AgentDAG`, `Orchestrator` (from `.orchestrator`)
- 10 specialized agents (from `.specialized`): DataValidationAgent, DataPrepAgent, VAEPretrainAgent, VAEFinetuneAgent, ESM2EmbedAgent, TrajectoryAgent, ProteinNetAgent, FusionAgent, LandscapeAgent, ValidationAgent

---

## 8. `resistancemap/agents/base.py`

**Full path:** `resistancemap/agents/base.py`
**Lines:** 262
**Purpose:** Abstract base class and shared abstractions for all agents.

### Imports
- `hashlib`, `json`, `logging`, `abc.ABC`, `abc.abstractmethod`
- `dataclasses.{dataclass, field}`, `enum.Enum`
- `typing.{Any, Optional}`, `pathlib.Path`
- `torch`

### Enum: `AgentState`
Values: `IDLE`, `RUNNING`, `COMPLETED`, `FAILED`, `WAITING_VERIFICATION`

### Dataclass: `AgentResult`
| Field | Type | Notes |
|---|---|---|
| `agent_name` | str | |
| `status` | AgentState | Auto-coerced from string in `__post_init__` |
| `output` | Any | The actual result |
| `verification_hash` | str | SHA256 of output |
| `metadata` | dict[str, Any] | Default empty |
| `error` | Optional[str] | Only if FAILED |

**Methods:**
- `to_dict() -> dict`: Serializes (excludes `output` to avoid large payloads).

### Class: `BaseAgent(ABC)`
Abstract base class for all 10 specialized agents.

**Attributes:** `name` (str), `dependencies` (list[str])

**Constructor:** `__init__(name, dependencies=None)`

**Abstract method:**
- `async execute(inputs: dict, config: Any) -> AgentResult`: Must validate inputs, do work, compute hash, return result.

**Concrete methods:**

- `verify_inputs(inputs: dict) -> tuple[bool, str]`: Default checks inputs is a dict. Subclasses override.

- `verify_output(result: AgentResult, expected_hash=None) -> tuple[bool, str]`: Recomputes hash of `result.output` and compares to `result.verification_hash`. Optionally compares to `expected_hash`.

- `compute_hash(data: Any) -> str` (static): SHA256 hash. Handles torch.Tensor (shape+dtype+bytes), dict/list (JSON sorted), str, int/float/bool, None, and fallback via `str()`.

- `_make_result(status, output=None, error=None, metadata=None) -> AgentResult`: Helper that auto-computes verification_hash.

- `async _safe_execute(inputs, config) -> AgentResult`: Wraps `execute()` with input verification, execution, output verification, and exception handling. Not meant to be overridden.

---

## 9. `resistancemap/agents/orchestrator.py`

**Full path:** `resistancemap/agents/orchestrator.py`
**Lines:** 511
**Purpose:** DAG-based orchestrator for parallel agent execution with zero-trust verification.

### Imports
- `asyncio`, `logging`, `time`, `collections.{defaultdict, deque}`, `dataclasses`
- `resistancemap.agents.base.{BaseAgent, AgentState, AgentResult}`
- `resistancemap.config.ResistanceMapConfig`

### Dataclass: `AgentDAG`
**Attributes:**
- `agents: dict[str, BaseAgent]` -- agent name to instance
- `edges: dict[str, list[str]]` -- agent name to dependency names

**Methods:**

- `add_agent(agent: BaseAgent)`: Registers agent and its dependencies. Raises `ValueError` if name exists.

- `topological_sort() -> list[list[str]]`: Kahn's algorithm with layer grouping for parallel execution. First validates all dependencies exist. Returns layers (each layer = agents that can run in parallel). Raises `ValueError` on cycles or missing dependencies.

- `get_ready_agents(completed: set[str]) -> list[str]`: Returns agents whose dependencies are all in `completed`.

- `validate() -> tuple[bool, str]`: Calls `topological_sort()` in try/except.

### Dataclass: `ExecutionPlan`
- `layers: list[list[str]]`
- `agent_order: list[str]` -- flattened in `__post_init__`

### Class: `Orchestrator`
**Constructor kwargs (all optional, v7):**
- `tracer`: AgentOps Tracer for span emission
- `evaluator`: AgentOps Evaluator for task result recording
- `optimizer`: AgentOps Optimizer (handoff timing)
- `guardrails`: GuardrailEngine for output checking

**Attributes:**
- `dag: AgentDAG`
- `results: dict[str, AgentResult]`
- `execution_plan: ExecutionPlan | None`
- `_verification_chain: list[str]`
- `_timing_log: dict[str, tuple[float, float]]`
- `_trace_id: str | None`
- `_guardrail_violations: list[dict]`

**Methods:**

- `add_agent(agent)`: Delegates to `self.dag.add_agent(agent)`.

- `prepare_execution() -> tuple[bool, str]`: Validates DAG and creates ExecutionPlan from topological sort.

- `async run(config) -> dict[str, AgentResult]`: Main execution loop.
  1. Starts a trace via `_tracer.start_trace()` if tracer available.
  2. Iterates through execution plan layers.
  3. For each layer, runs all agents concurrently via `asyncio.gather()` calling `_run_agent_isolated()`.
  4. Stores results, appends verification hashes.
  5. Logs COMPLETED/FAILED for each agent.
  6. Ends the trace.
  7. Returns all results.

- `async _run_agent_isolated(agent, agent_name, config) -> AgentResult`:
  1. Starts an AgentOps span if tracer available.
  2. Gathers inputs from completed dependencies.
  3. Zero-trust verifies each dependency's output hash.
  4. Calls `agent._safe_execute(inputs, config)`.
  5. Records timing.
  6. Runs GuardrailEngine.check_all on output (non-blocking; violations recorded but don't fail agent).
  7. Records result in Evaluator.
  8. Ends span in finally block.

- `get_execution_plan() -> list[list[str]]`: Returns layers from plan.
- `get_verification_chain() -> list[str]`: Copy of hash chain.
- `get_timing_report() -> dict`: Per-agent start/end/elapsed.
- `get_results_summary() -> dict`: Counts by status, failed agent list.

---

## 10. `resistancemap/agents/specialized.py`

**Full path:** `resistancemap/agents/specialized.py`
**Lines:** 476
**Purpose:** 10 domain-specific agents wrapping the sequential stage functions from `resistancemap.main`.

### Helper Function
#### `_get_ckpt_mgr(config) -> CheckpointManager`
Gets the shared CheckpointManager from `config._ckpt_mgr` (attached by `run_agentic_pipeline`), or creates new one from `config.checkpoint_dir`.

### Agent Classes

All agents follow the same pattern:
1. Constructor sets `name` and `dependencies`.
2. `verify_inputs()` checks required dependency outputs exist.
3. `execute()` calls the corresponding `resistancemap.main` function via `asyncio.to_thread()`, then loads the checkpoint to read back metrics.

#### Layer 0: `DataValidationAgent`
- Name: `"data_validation"`, Dependencies: `[]`
- Calls `validate_data_files(config, ckpt_mgr)`.
- Output: `{valid, errors, paths_checked}`.

#### Layer 1: `DataPrepAgent`
- Name: `"data_prep"`, Dependencies: `["data_validation"]`
- Verifies data_validation passed (valid=True).
- Calls `prepare_data(config, ckpt_mgr)`.
- Output: `{checkpoint_path, proteomics_shape, epigenomics_shape, n_samples, n_proteins}`.

#### Layer 2: `VAEPretrainAgent`
- Name: `"vae_pretrain"`, Dependencies: `["data_prep"]`
- Calls `pretrain_vae(config, ckpt_mgr)`.
- Output: `{checkpoint_path, model_type, metrics}`.

#### Layer 2: `ESM2EmbedAgent`
- Name: `"esm2_embed"`, Dependencies: `["data_prep"]`
- Does NOT run the ESM-2 forward pass itself. Pre-validates protein metadata.
- Checks `dataset.protein_sequences` availability.
- Output: `{n_proteins, esm2_model, embedding_dim, has_sequences, embeddings_produced_here=False}`.
- Logs whether real ESM-2 embeddings will be available for ProteinNetAgent.

#### Layer 3: `VAEFinetuneAgent`
- Name: `"vae_finetune"`, Dependencies: `["vae_pretrain"]`
- Calls `finetune_vae(config, ckpt_mgr)`.

#### Layer 4: `TrajectoryAgent`
- Name: `"trajectory"`, Dependencies: `["vae_finetune"]`
- Calls BOTH `calibrate_trajectory()` AND `train_trajectory_forecaster()` sequentially.
- Output includes both checkpoint paths and merged metrics.

#### Layer 5: `ProteinNetAgent`
- Name: `"protein_net"`, Dependencies: `["esm2_embed", "vae_finetune", "trajectory"]`
- Calls `train_protein_network(config, ckpt_mgr)`.

#### Layer 6: `FusionAgent`
- Name: `"fusion"`, Dependencies: `["trajectory", "protein_net"]`
- Calls `train_fusion(config, ckpt_mgr)`.

#### Layer 7: `LandscapeAgent`
- Name: `"landscape"`, Dependencies: `["fusion"]`
- Calls `train_landscape(config, ckpt_mgr)`.

#### Layer 8: `ValidationAgent`
- Name: `"validation"`, Dependencies: `["landscape"]`
- Calls `validate_pipeline(config, ckpt_mgr)`.

---

## 11. `resistancemap/agents/example_usage.py`

**Full path:** `resistancemap/agents/example_usage.py`
**Lines:** 197
**Purpose:** Standalone demo script showing how to use the orchestrator.

### Function: `async main() -> bool`
1. Sets up logging.
2. Loads config from `config.yaml`.
3. Creates Orchestrator, registers all 10 agents.
4. Prepares execution plan, displays layers.
5. Runs orchestrator.
6. Collects and prints: results summary, verification chain, timing report, detailed per-agent results.
7. Saves to `checkpoints/`: `verification_chain.json`, `execution_plan.json`, `results_summary.json`.
8. Returns True on success.

### `__main__` block
Calls `asyncio.run(main())`, exits with 0/1.

---

## 12. `resistancemap/agentops/__init__.py`

**Full path:** `resistancemap/agentops/__init__.py`
**Lines:** 79
**Purpose:** Package init for the AgentOps framework. Three layers: Tracer (observability), Evaluator (evaluation), Optimizer (optimization).

### Exports (`__all__`)
- Tracer: `Tracer`, `Trace`, `Span`
- Evaluator: `Evaluator`, `EvaluationMetrics`, `GuardrailViolation`, `TaskResult`, `AccuracyCheck`, `ClinicalCheck`, `FirstPassCheck`
- Optimizer: `Optimizer`, `OptimizationMetrics`, `TokenUsageRecord`, `RetrievalRecord`, `HandoffRecord`, `FlowStepRecord`
- Dashboard: `AgentOpsDashboard`, `AgentPerformance`

---

## 13. `resistancemap/agentops/tracer.py`

**Full path:** `resistancemap/agentops/tracer.py`
**Lines:** 466
**Purpose:** End-to-end distributed tracing inspired by OpenTelemetry.

### Dataclass: `Span`
A single unit of work.

| Field | Type | Default | Notes |
|---|---|---|---|
| `span_id` | str | uuid4() | |
| `trace_id` | str | "" | |
| `parent_id` | Optional[str] | None | |
| `agent_name` | str | "" | |
| `operation` | str | "" | |
| `start_time` | float | time.time() | |
| `end_time` | Optional[float] | None | |
| `status` | str | "running" | running/completed/failed |
| `metadata` | Dict | {} | |
| `children` | List[Span] | [] | |

**Properties:**
- `duration_ms`: Elapsed time in ms (live if end_time is None).
- `cost_usd`: From `metadata["cost_usd"]`.
- `token_count`: `metadata["input_tokens"] + metadata["output_tokens"]`.

**Method:** `to_dict()` -- serializes all fields including computed properties.

### Dataclass: `Trace`
End-to-end trace of a pipeline execution.

| Field | Type | Default |
|---|---|---|
| `trace_id` | str | uuid4() |
| `spans` | Dict[str, Span] | {} |
| `start_time` | float | time.time() |
| `end_time` | Optional[float] | None |

**Properties:**
- `total_duration_ms`, `total_cost`, `total_tokens`

**Methods:**
- `get_agent_handoff_latencies() -> Dict[str, float]`: Computes ms between consecutive spans' end->start for agent-to-agent transitions.
- `get_tool_execution_latencies() -> Dict[str, float]`: Average duration_ms per operation name.
- `get_cost_per_request() -> float`: Same as total_cost.
- `get_critical_path() -> List[Span]`: DFS longest path from root spans through parent->child relationships.
- `to_dict()`: Serializes trace and all spans.

### Class: `Tracer` (Singleton)
Thread-safe singleton for distributed tracing.

**Class variables:** `_instance`, `_lock` (threading.Lock)

**Instance attributes:**
- `active_traces: Dict[str, Trace]`
- `completed_traces: List[Trace]`
- `_span_stack: Dict[int, List[str]]` -- per-thread span ID stack
- `_thread_lock: threading.Lock`

**Methods:**
- `get_instance() -> Tracer`: Classic double-checked locking singleton.
- `start_trace(trace_id=None) -> Trace`: Creates and registers a new active trace.
- `start_span(trace_id, agent_name, operation, parent_span_id=None) -> Span`: Creates span within a trace. Pushes to per-thread span stack.
- `end_span(span_id, status="completed", metadata=None)`: Sets end_time, status, updates metadata. Pops from span stack.
- `end_trace(trace_id)`: Moves trace from active to completed.
- `export_metrics() -> Dict`: Aggregates all completed traces: total traces/spans/cost/tokens/duration, average handoff latencies, average tool latencies.
- `get_active_trace_ids()`, `get_completed_trace_ids()`, `get_trace(trace_id)`: Lookup methods.
- `trace_agent(agent_name, operation, trace_id=None)`: Context manager that auto-starts/ends a span. Finds active trace if trace_id not given. Uses current thread's span stack for parent_id.
- `clear()`: Clears all state.

---

## 14. `resistancemap/agentops/evaluator.py`

**Full path:** `resistancemap/agentops/evaluator.py`
**Lines:** 450
**Purpose:** Task completion tracking, guardrail violations, accuracy checks, clinical reviews.

### Dataclasses

#### `GuardrailViolation`
Fields: `timestamp`, `agent_name`, `violation_type`, `severity`, `description`, `input_hash`. Has `to_dict()`.

#### `TaskResult`
Fields: `task_id`, `status`, `verified`, `metadata`, `timestamp`. Has `to_dict()`.

#### `AccuracyCheck`
Fields: `claim`, `source`, `verified`, `confidence`, `timestamp`. Has `to_dict()`.

#### `ClinicalCheck`
Fields: `recommendation`, `appropriate`, `reviewer`, `notes`, `timestamp`. Has `to_dict()`.

#### `FirstPassCheck`
Fields: `output_id`, `approved`, `timestamp`. Has `to_dict()`.

#### `EvaluationMetrics`
Aggregate metrics: `task_completion_rate`, `guardrail_violation_rate`, `factual_accuracy_rate`, `clinical_appropriateness`, `first_pass_approval_rate`, `violations_by_type`, `violations_by_severity`, `computed_at`.

### Class: `Evaluator`
Thread-safe evaluation tracker.

**Internal lists (protected by `_lock`):**
- `_violations`, `_task_results`, `_accuracy_checks`, `_clinical_checks`, `_first_pass_checks`

**Record methods:**
- `record_task_result(task_id, status, verified=False, metadata=None)`: Appends TaskResult.
- `record_guardrail_violation(agent_name, violation_type, severity, description, input_hash=None)`: Appends GuardrailViolation.
- `record_accuracy_check(claim, source, verified, confidence=1.0)`: Appends AccuracyCheck.
- `record_clinical_check(recommendation, appropriate, reviewer="", notes="")`: Appends ClinicalCheck.
- `record_first_pass(output_id, approved)`: Appends FirstPassCheck.

**Computed metrics:**
- `get_metrics() -> EvaluationMetrics`: Computes all rates from internal lists:
  - task_completion_rate = completed_or_partial / total_tasks
  - guardrail_violation_rate = violations / (tasks + accuracy_checks + clinical_checks)
  - factual_accuracy_rate = verified / total_accuracy_checks
  - clinical_appropriateness = appropriate / total_clinical
  - first_pass_approval_rate = approved / total_first_pass
  - violations_by_type and violations_by_severity dicts

**Query methods:**
- `get_violations_by_agent()`, `get_violations_by_severity()`: Group violations.
- `get_all_violations()`, `get_all_task_results()`, etc.: Return copies.
- `export_violations_json() -> List[Dict]`
- `clear()`: Resets all lists.

---

## 15. `resistancemap/agentops/optimizer.py`

**Full path:** `resistancemap/agentops/optimizer.py`
**Lines:** 509
**Purpose:** Efficiency metrics and improvement tracking: token efficiency, retrieval precision, handoff success, flow efficiency.

### Dataclasses

#### `TokenUsageRecord`
Fields: `input_tokens`, `output_tokens`, `useful_tokens`, `timestamp`. Has `to_dict()`.

#### `RetrievalRecord`
Fields: `query`, `results_count`, `relevant_results_count`, `k`, `timestamp`.
Property: `precision_at_k = relevant / k`.

#### `HandoffRecord`
Fields: `from_agent`, `to_agent`, `success`, `latency_ms`, `timestamp`.

#### `FlowStepRecord`
Fields: `step_id`, `necessary`, `duration_ms`, `timestamp`.

#### `MetricSnapshot`
Fields: `timestamp`, `metrics` (dict). Point-in-time snapshot for velocity computation.

#### `OptimizationMetrics`
Aggregate: `prompt_token_efficiency`, `retrieval_precision_at_k` (dict[int, float] for P@1/5/10), `handoff_success_rate`, `flow_step_efficiency`, `improvement_velocity`, `avg_handoff_latency_ms`, `total_tokens_saved`, `total_steps_eliminated`.

### Class: `Optimizer`
Thread-safe optimization tracker.

**Internal lists (protected by `_lock`):**
- `_token_usage`, `_retrieval_results`, `_handoffs`, `_flow_steps`, `_metric_history`

**Record methods:**
- `record_token_usage(input_tokens, output_tokens, useful_tokens=0)`: If useful_tokens=0, defaults to output_tokens.
- `record_retrieval(query, results, relevant_results, k=10)`
- `record_handoff(from_agent, to_agent, success, latency_ms)`
- `record_flow_step(step_id, necessary, duration_ms)`
- `record_metric_snapshot(metrics: Dict)`

**Computed metrics:**
- `get_metrics() -> OptimizationMetrics`:
  - token_efficiency = total_useful / total_input
  - retrieval_precision at k=[1,5,10]
  - handoff_success_rate = successful / total
  - flow_step_efficiency = necessary / total
  - improvement_velocity via `_compute_improvement_velocity(window_days=7)`

- `get_improvement_velocity(metric_name, window_days=7) -> float`: Rate of change per day for a specific metric over a 7-day window. Requires >=2 snapshots.

- `suggest_optimizations() -> List[str]`: Generates actionable suggestions based on thresholds:
  - Token efficiency < 70%
  - P@1 < 50%
  - Handoff success < 90%
  - Flow efficiency < 70%
  - Negative or positive velocity

**Query methods:**
- `get_all_token_usage()`, `get_all_retrieval_records()`, `get_all_handoff_records()`, `get_all_flow_steps()`
- `clear()`

---

## 16. `resistancemap/agentops/dashboard.py`

**Full path:** `resistancemap/agentops/dashboard.py`
**Lines:** 346
**Purpose:** Unified dashboard combining Tracer, Evaluator, and Optimizer for comprehensive reporting.

### Dataclass: `AgentPerformance`
Per-agent performance metrics: `agent_name`, `num_operations`, `avg_operation_duration_ms`, `total_cost_usd`, `success_rate`, `avg_accuracy`, `violations_count`, `score` (composite 0-100).

### Class: `AgentOpsDashboard`

**Constructor:** Takes optional `tracer`, `evaluator`, `optimizer`. Defaults to singleton/new instances.

**Methods:**

- `get_full_report() -> Dict`: Combines all three layers:
  - `observability`: traces, spans, duration, cost, tokens, handoff latencies, tool latencies
  - `evaluation`: task completion, guardrail violations, accuracy, clinical, first-pass rates
  - `optimization`: token efficiency, retrieval precision, handoff success, flow efficiency, velocity
  - `agent_performance`: leaderboard
  - `optimization_suggestions`: from optimizer

- `export_json(path: str)`: Writes full report to JSON file.

- `print_summary()`: Human-readable terminal output with sections for observability, evaluation, optimization, agent leaderboard (top 10 by composite score), and suggestions.

- `_compute_agent_leaderboard() -> List[AgentPerformance]`:
  - Builds per-agent data from completed traces (durations, costs, tokens, success count).
  - Gets violations from evaluator.
  - Composite score = success_rate * 40 + accuracy * 30 + (1 - min(violations/10, 1)) * 20 + efficiency_score * 10.
  - efficiency_score = max(0, 1 - avg_duration_ms/5000).

- `get_agent_leaderboard()`, `get_observability_metrics()`, `get_evaluation_metrics()`, `get_optimization_metrics()`: Convenience accessors.

- `clear_all()`: Clears all three layers.

---

## Cross-Module Data Flow Summary

```
CLI (main.py)
  |
  v
resistancemap/main.py::main()
  |
  +-- --sequential --> run_sequential_pipeline()
  |                       |
  |                       +-- STAGES[0..10] called in order
  |
  +-- --evaluate --> run_evaluation_governance()
  |                   |
  |                   +-- EvalOrchestrator with 10 tier agents
  |
  +-- --v10-sprint --> v10_runner.run_v10_e2e()
  |                     |
  |                     +-- subprocess calls to scripts/v10/s*.py
  |
  +-- (default) --> run_agentic_pipeline()
                      |
                      +-- AgentOps init (Tracer, Evaluator, Optimizer, Dashboard)
                      +-- ZeroTrust + Guardrails init
                      +-- build_agent_dag() --> Orchestrator with 10 agents
                      +-- orchestrator.run(config)
                            |
                            +-- Topological sort into 9 layers
                            +-- For each layer: asyncio.gather()
                                  |
                                  +-- _run_agent_isolated() per agent
                                        |
                                        +-- Verify dep outputs (SHA256)
                                        +-- agent._safe_execute()
                                              |
                                              +-- verify_inputs()
                                              +-- execute() [specialized]
                                              |     |
                                              |     +-- asyncio.to_thread(main.stage_fn)
                                              +-- verify_output()
                                        +-- GuardrailEngine.check_all()
                                        +-- Evaluator.record_task_result()
```

### Checkpoint Chain
```
data_ready --> vae_pretrained --> vae_finetuned --> stability_calibrated
     \                                                    |
      \                                                   v
       +-------> ESM2 metadata -------> trajectory_forecaster_trained
                                              |
                                              v
                            protein_net_trained
                                    |
                                    v
                              fusion_trained
                                    |
                                    v
                            landscape_trained
                                    |
                                    v
                          pipeline_validated
```

### Key Design Decisions

1. **Dual execution modes**: Agentic (DAG parallel) is default; sequential is legacy fallback.
2. **Zero-trust verification**: Every agent output is SHA256 hashed; downstream agents verify hashes before consuming.
3. **AgentOps observability**: Three-layer system (Tracer/Evaluator/Optimizer) with unified dashboard.
4. **Per-drug heads**: Separate Linear(fusion_dim, 1) per drug to prevent HDAC-class gradient poisoning via Adam normalization.
5. **Inverse-variance loss weighting**: Computed from training split only to avoid leakage.
6. **ESM-2 integration**: Optional; falls back to abundance-only features. Cached embeddings via SHA256 hash of protein names.
7. **Trajectory forecaster limitation**: Contemporaneous supervision only; horizon labels are nominal, not calendar-time calibrated.
8. **v10 sprint orchestrator**: Completely orthogonal to the legacy STAGES and evaluation governance; runs scripts via subprocess.
9. **Guardrail violations are non-blocking**: Recorded for dashboard but don't fail agents (user can promote to hard-stop later).
10. **Evaluation governance tiers**: Tier A gates block downstream; Tier D is integrative chair.

---

## Appendix B: Neural Network Architectures (models/)

## `resistancemap/models/` -- Model Architectures (28 files, ~13,800 lines)

### `__init__.py`
Re-exports from three modules: `identifiable_ode` (BiologicalProgram, IdentifiableODEConfig, SparseInteractionMatrix, TreatmentHyperNetwork, DrugInterventionModule, LyapunovNet, LyapunovStabilityCertifier, StructuredBiologicalODE, IdentifiableODELoss), `causal_fusion` (Modality, CausalFusionConfig, CausalModalityGraph, SCMLayer, CausalAttentionBlock, ConfounderEncoder, CausalFusionNetwork), and `disentangled_vae` (DisentangledVAEConfig, AnchorSpec, Encoder, Decoder, DisentangledVAE, DCIMetrics, TraversalGenerator).

---

### `vae.py` -- Proteome-to-Epigenome Conditional VAE (Module 1)

**Purpose**: Maps proteomic profiles to latent epigenetic memory states and reconstructs expected epigenomic profiles (ATAC-seq peaks, histone modifications). The latent space (64-dim default, scalable to 256) represents the inferred chromatin landscape.

#### Classes

**`_GradientReversalFunction(torch.autograd.Function)`**
- Custom autograd function. Forward: identity. Backward: negate and scale gradients by `lambda_`.

**`GradientReversalLayer(nn.Module)`**
- `__init__(self, lambda_: float = 1.0)`
- `forward(self, x: Tensor) -> Tensor`: wraps `_GradientReversalFunction.apply`.

**`FiLMLayer(nn.Module)`**
- Feature-wise Linear Modulation: `output = gamma(conditioning) * input + beta(conditioning)`.
- `__init__(self, conditioning_dim: int, feature_dim: int)`: learns `fc_gamma` and `fc_beta` Linear layers.
- `forward(self, x, conditioning) -> Tensor`.

**`ConditionalDomainDiscriminator(nn.Module)`**
- Domain classifier with GRL for adversarial training.
- `__init__(self, latent_dim: int, n_domains: int = 2)`: GRL + MLP (latent_dim -> 128 -> LeakyReLU -> Dropout(0.3) -> 64 -> LeakyReLU -> Dropout(0.3) -> n_domains).
- `forward(self, z) -> Tensor`: (B, n_domains) logits.

**`_EncoderBlock(nn.Module)`** / **`_DecoderBlock(nn.Module)`**
- Single layer: Linear -> BatchNorm (optional) -> GELU -> Dropout.

**`StochasticDecoder(nn.Module)`**
- Outputs distribution parameters (mean, log_var) over epigenomic space instead of deterministic reconstruction.
- `__init__(self, latent_dim, epigenome_dim, decoder_hidden_dims, dropout, use_batch_norm)`: backbone of `_DecoderBlock` layers + `mean_head` and `logvar_head` Linear layers.
- `forward(self, z) -> (mean, log_var)`.

**`ProteomeToEpigenomeVAE(nn.Module)`**
- `__init__(self, config: VAEConfig)`:
  - `latent_dim`: uses `config.expanded_latent_dim` if set, else `config.latent_dim`.
  - Encoder: sequence of `_EncoderBlock` layers with optional `FiLMLayer` interleaved if `conditioning_dim > 0`.
  - Latent projections: `fc_mu` and `fc_log_var` (Linear).
  - Decoder: either `StochasticDecoder` or standard `_DecoderBlock` sequence + `output_head` Linear.
  - Optional `ConditionalDomainDiscriminator` if `domain_adversarial=True`.
  - Xavier uniform init, skipping StochasticDecoder children.
- `reparameterize(self, mu, log_var) -> z`: standard reparam trick; deterministic at inference.
- `encode(self, x, conditioning=None) -> (mu, log_var)`: supports gradient checkpointing.
- `decode(self, z)`: returns Tensor (deterministic) or (mean, log_var) (stochastic).
- `forward(self, x, conditioning=None)`: returns (recon, mu, log_var) or (recon_mean, recon_logvar, mu, log_var).
- `get_memory_state(self, x, conditioning=None) -> Tensor`: deterministic mu, the primary output for downstream modules.
- `get_epigenome_reconstruction_with_uncertainty(self, x, conditioning=None) -> (mean, log_var)`: stochastic decoder only.
- `domain_classify(self, z) -> Tensor`: requires `domain_adversarial=True`.

#### Standalone Functions

- `_kl_divergence(mu, log_var) -> Tensor`: KL(N(mu,var) || N(0,I)).
- `_stochastic_reconstruction_loss(target, mean, log_var) -> Tensor`: Gaussian NLL for stochastic decoder.
- `_cyclical_kl_weight(step, total_steps, n_cycles, ratio, max_weight) -> float`: cyclical beta-annealing schedule (Fu et al. 2019).
- `extract_latent_trajectory(model, proteomics_timeseries) -> Tensor`: encodes time series (T, P) or (B, T, P) into latent trajectories.
- `train_vae(model, dataset, splits, config, subset, ckpt_mgr, stage_name) -> dict`: full training loop with cyclical KL annealing, mixed precision (bf16), gradient checkpointing, early stopping, optional domain-adversarial loss.

**Key hyperparameters**: `kl_anneal_cycles`, `kl_anneal_ratio`, `kl_weight_max`, `gradient_clip_norm`, `patience`, `min_delta`.

---

### `trajectory.py` -- Memory Stability Scorer and Trajectory Forecaster (Module 2)

**Purpose**: ODE-based Sneppen-Ringrose chromatin bistability model. Computes basin-of-attraction depth as stability score (0 = transient, 1 = locked-in). NOT a longitudinal forecaster -- trained on contemporaneous (X, IC50) snapshots.

#### Classes

**`SinkhornOT(nn.Module)`**
- `__init__(self, epsilon: float = 0.1, n_iterations: int = 100)`.
- `_sinkhorn_iterations(self, cost_matrix, n_iters, tolerance) -> (transport_plan, dual_vars)`: log-domain Sinkhorn with early stopping.
- `forward(self, cloud1, cloud2) -> (transport_plan, interpolation)`: McCann displacement interpolation at t=0.5.

**`DriftNetwork(nn.Module)`**
- `__init__(self, latent_dim=64, hidden_dim=32)`: MLP (latent_dim+1 -> hidden -> hidden -> latent_dim) with GELU.
- `forward(self, z, t) -> drift_vector`.

**`DiffusionNetwork(nn.Module)`**
- Same architecture as DriftNetwork but with Softplus output for positive-definiteness.
- `forward(self, z, t) -> diagonal_diffusion_coefficients`.

**`JumpProcess(nn.Module)`**
- Models therapy-induced discontinuities via Poisson jump process.
- `__init__(self, latent_dim=64, hidden_dim=32)`: rate_net (-> 1, Softplus) and size_net (-> latent_dim).
- `jump_rate(self, z, t) -> (B, 1)`.
- `jump_size(self, z) -> (B, D)`.
- `forward(self, z, t, dt, rng) -> (z_after_jump, n_jumps)`: Poisson sampling.

**`SurvivalTimeCalibrator(nn.Module)`**
- Maps latent pseudotime to calendar time via Weibull hazard model.
- `__init__(self, latent_dim=64)`: learnable `shape` and `scale` parameters.
- `forward(self, pseudotime) -> calendar_time`: Weibull quantile function `lambda * (-log(1-u))^(1/k)`.
- `survival_function(self, t) -> S(t)`: `exp(-(t/lambda)^k)`.

**`NeuralJumpSDE(nn.Module)`**
- SDE with drift + diffusion + jumps wrapping `ChromatinODE`.
- `__init__(self, ode: ChromatinODE, latent_dim=64, hidden_dim=32, include_jumps=True)`.
- `forward(self, z0, t_span, n_steps=100, stochastic=True, n_samples=1) -> trajectory`: Euler-Maruyama integration.

**`ChromatinODE(nn.Module)`**
- Two-state ODE: active mark (a) vs repressive mark (r).
- `__init__(self, config: StabilityConfig)`:
  - `protein_to_params`: MLP (n_proteins -> 64 -> 32 -> 8, Softplus). Outputs [w_a, w_r, e_a, e_r, d_a, d_r, basal_a, basal_r].
  - Learnable Hill function params: `hill_n` (cooperativity), `hill_k` (half-max).
- `forward(self, t, state) -> derivatives`: state is (B, 2+8) where [:, 0:2] = (a, r) and [:, 2:] = ODE params (zero derivative).
- Dynamics: `da/dt = w_a * hill(a) - e_r * hill(r) * a - d_a * a + basal_a` (symmetric for r).

**`MemoryStabilityScorer(nn.Module)`**
- `__init__(self, config: StabilityConfig)`: wraps ChromatinODE + learnable `basin_center` and `basin_scale` for sigmoid normalization.
- `extract_reader_writer_levels(self, proteomics, all_protein_names) -> Tensor`: extracts specific proteins by name matching.
- `_find_steady_state(self, ode_params) -> (a_steady, r_steady)`: integrates ODE from balanced (0.5, 0.5) state using Euler with step_size=0.1.
- `_estimate_basin_depth(self, ode_params, a_steady, r_steady) -> Tensor`: computes 2x2 Jacobian via finite differences, eigenvalues via quadratic formula, basin depth = -lambda_max.
- `forward(self, proteomics, all_protein_names) -> Tensor`: full pipeline, output in [0, 1] via `sigmoid(basin_scale * (depth - basin_center))`.

**`TrajectoryForecaster(nn.Module)`**
- `__init__(self, config, protein_names, use_sde=False)`:
  - `horizon_times`: {3: 30.0, 6: 60.0, 12: 120.0} (nominal, NOT calendar-calibrated).
  - `latent_to_params`: Linear(64, 8) (reserved for future use).
  - `basin_transition_net`: MLP (66 -> 32 -> 16 -> 2) for basin assignment logits.
  - Optional SDE components: `NeuralJumpSDE`, `SurvivalTimeCalibrator`, `SinkhornOT`.
- `forecast(self, initial_state, protein_abundances, horizons=[3,6,12], n_samples=1) -> dict`: rolls out ODE from (0.5, 0.5) initial state at each horizon, returns states, stability_scores, transition_probs, temporal_disclaimers.
- `forecast_with_uncertainty(...)`: Monte Carlo SDE rollouts with confidence intervals.

**`ForecastReliability` (dataclass)** / **`ForecastReliabilityScorer`**
- Grade system: A (0-3m), B (3-6m), C (6-9m), F (9-12m+).
- Exponential confidence decay: `base_decay_rate^months`.
- `score()`, `validate_forecast_request()`, `apply_confidence_decay()`.

#### Standalone Function

- `calibrate_scorer(scorer, dataset, vae_checkpoint, config, ckpt_mgr, ...) -> dict`: calibrates stability scorer using drug sensitivity variance as proxy (high variance -> low stability target).

**Key constants**: `TEMPORAL_DISCLAIMERS` dict with horizon-specific epistemic warnings.

---

### `protein_network.py` -- L3 Protein Network Propagator

**Purpose**: Combines ESM-2 protein embeddings with GNN-based PPI propagation and pathway-aware encoding for per-protein resistance scoring.

#### Classes

**`DropEdge(nn.Module)`**: randomly drops edges during training (regularization). `__init__(drop_rate=0.2)`.

**`PairNorm(nn.Module)`**: PairNorm normalization (Zhao & Akoglu 2020). Centers features and scales to unit variance.

**`ESM2Bottleneck(nn.Module)`**: Linear(1280, 256) + LayerNorm + GELU + Dropout(0.1). Reduces ESM-2 embeddings.

**`DrugConditionedAttention(nn.Module)`**: Modulates GAT attention scores based on drug identity via a gate network (drug_embedding_dim -> 64 -> 1, Sigmoid).

**`JumpingKnowledge(nn.Module)`**: Aggregates node representations across GNN layers. Modes: "cat" (concat + project), "max" (element-wise max), "lstm".

**`GraphMASKLayer(nn.Module)`**: Learns differentiable binary mask over edges for causal attribution. Gate: Linear(edge_dim, 64) -> GELU -> Linear(64, 1) -> Sigmoid.

**`EvidentialClassificationHead(nn.Module)`**: Dirichlet-based uncertainty via Softplus(logits) + 1. `uncertainty(alpha) = K / sum(alpha)`. Default 3 classes (sensitive/intermediate/resistant).

**`ESM2Embedder(nn.Module)`**: Wraps `facebook/esm2_t33_650M_UR50D` from HuggingFace. LRU cache (10000 sequences). Embedding dim = 1280. Mean-pools over sequence length.

**`PPIGraphNetwork(nn.Module)`**
- `__init__(self, in_dim=321, hidden_dim=256, n_layers=4, n_heads=8, dropout=0.2, edge_dim=1, dropedge_rate=0.0, use_pairnorm=False, use_jumping_knowledge=False, use_graphmask=False)`.
- Architecture: Input projection (Linear) -> N GATConv layers (hidden//heads per head) with residual connections + LayerNorm + ReLU + Dropout. Optional DropEdge, PairNorm, JumpingKnowledge, GraphMASK.
- `forward(self, data: PyGData) -> Tensor`: (num_nodes, hidden_dim) node embeddings.

**`PathwayAwareProteinEncoder(nn.Module)`**
- Cross-attention between protein features and learnable pathway embeddings (n_pathways=50).
- `forward(self, x, mask=None) -> Tensor`: (num_nodes, output_dim).

**`ResistancePropagator(nn.Module)`**
- GlobalAttention pooling (gate: Linear -> GELU -> Linear(1)) + resistance scoring head.
- `forward(self, node_emb, batch_vec) -> (global_resistance, per_protein_resistance)`.

**`ProteinNetworkPropagator(nn.Module)`**
- Top-level propagator combining ESM2Embedder + ESM2Bottleneck + PPIGraphNetwork + PathwayAwareProteinEncoder + ResistancePropagator + optional EvidentialClassificationHead.
- `__init__(self, n_proteins=10000, hidden_dim=256, n_pathway_embeddings=50, ...)`.
- Node feature dim: `esm2_bottleneck_dim(256) + latent_dim(64) + stability_dim(1) + phospho_dim`.
- `forward(self, sequences, latent_states, stability_scores, edge_index, ...) -> dict` with keys: node_embeddings, pathway_aware, global_resistance, per_protein_resistance, optional evidential_alpha/uncertainty.

---

### `fusion.py` -- L4 Multi-Modal Fusion

**Purpose**: Combines four modalities (epigenetic 64-d, trajectory 64-d, protein network 256-d, stability 1-d) into unified representation for landscape prediction.

#### Classes

**`ResidualBatchCorrector(nn.Module)`**
- `__init__(self, feature_dim, n_batches, embedding_dim=None)`: batch embeddings + Linear projection + learnable `residual_scale` (init 0.1).
- `forward(self, features, batch_ids) -> corrected_features`.

**`CrossModalAttentionBlock(nn.Module)`**
- MultiheadAttention(dim, n_heads) + LayerNorm + FFN (dim -> 2*dim -> dim) + LayerNorm. All with residual connections.
- `forward(self, query, key_value) -> (attended, attn_weights)`.

**`CrossModalFusionNet(nn.Module)`**
- `__init__(self, modality_dims: dict, hidden_dim=128, n_heads=4, dropout=0.2, output_dim=None)`.
- Per-modality Linear projection + default embeddings for missing modalities.
- CrossModalAttentionBlock per modality (each attends to concatenation of others).
- Gated fusion: Linear(hidden*n_mod -> hidden -> n_mod, Softmax) with missing-modality mask + renormalization.
- Output projection: Linear(hidden, output_dim).
- Static method `compute_mmd_loss()` for domain adaptation (RBF or linear kernel).

**`TensorFusion(nn.Module)`**
- For 2 modalities: outer product `einsum("bi,bj->bij")` -> flatten -> MLP.
- For >2: concatenation -> MLP.

**`ResistanceMapFusion(nn.Module)`**
- `__init__(self, hidden_dim=256, output_dim=128, fusion_type="cross_attention"|"tensor"|"concat", ...)`.
- Default modality dims: epigenetic=64, trajectory=64, protein_network=256, stability=1.
- Optional `ResidualBatchCorrector` per modality.
- `forward(self, epigenetic_state, trajectory_state, protein_network_output, stability_score, batch_ids=None) -> dict`.

**`MultiModalFusionPipeline(nn.Module)`**: wraps `ResistanceMapFusion`, returns just `fused_representation`.

---

### `foundation_fusion.py` -- MORT-FM Tokenised Cross-Modal Transformer Fusion

**Purpose**: Foundation-model fusion for MORT-FM. Accepts variable modality tokens, handles missing modalities via key-padding mask, uses [CLS] token for foundation latent z0.

**`MultiOmicFoundationFusion(nn.Module)`**
- `__init__(self, config: MORTFMConfig, *, return_attention=False)`:
  - `tokenizer`: `ModalityTokenizer(config)`.
  - `cls_token`: learnable (1, 1, d_token), trunc_normal init.
  - `time_proj`: Linear(1, d_token) for time auxiliary token.
  - `patient_token`: learnable (1, 1, d_token).
  - `transformer`: nn.TransformerEncoder with `config.fusion_layers` layers, d_model=d_token, nhead=config.fusion_heads, dim_feedforward=4*d_token, GELU activation, norm_first=True, batch_first=True.
  - `layer_norm`: LayerNorm(d_token).
  - `latent_proj`: Linear(d_token, d_latent).
- `_build_token_sequence(modality_tokens, presence, time)`: prepends [CLS] + patient tokens, optionally time token, appends modality tokens. Returns (seq, mask).
- `forward(self, batch: MORTBatch) -> FoundationState`:
  - Calls tokenizer -> builds token sequence -> applies transformer with key_padding_mask (~presence) -> extracts CLS output -> projects to z0.
  - Computes modality gates via cosine similarity between CLS and modality tokens, softmax-normalised across present modalities.
  - Returns `FoundationState(z0, modality_embeddings, attention_maps, modality_gates, uncertainty=None)`.

**Imports**: `MORTFMConfig`, `FoundationState`, `MORTBatch` from `resistancemap.mortfm.schemas`; `ModalityTokenizer` from encoders.

---

### `encoders/` -- MORT-FM Modality-Specific Encoders (10 files)

#### `encoders/__init__.py`
Exports: RNAEncoder, ATACEncoder, MethylationEncoder, HistonePTMEncoder, ProteomicEncoder, PhosphoproteomicEncoder, ClinicalEncoder, DrugEncoder, ModalityTokenizer.

**Design rules**: Every encoder accepts a presence mask and zeros output for absent rows. Every encoder exposes `reconstruct(z)` for modality reconstruction losses.

#### `encoders/rna_encoder.py` -- RNAEncoder
- Negative-binomial likelihood + library-size normalisation (scVI-style).
- `__init__(input_dim, d_token=64, hidden_dims=(512,256), dropout=0.1, n_batches=0, use_layernorm=True)`.
  - Optional batch embedding (nn.Embedding(n_batches, 16)).
  - Encoder: MLP (input_dim[+16] -> hidden_dims -> d_token) with LayerNorm + GELU + Dropout.
  - Decoder: `decoder_mu` Linear(d_token, input_dim), `log_theta` Parameter(input_dim), `decoder_pi` Linear(d_token, input_dim).
- `encode(x, batch_id, presence_mask) -> z`: (N, d_token).
- `decode(z, library_size) -> (rate, theta, pi_logits)`: for NB-ZI reconstruction.
- `forward(x, ...) -> z`: delegates to encode.
- Standalone: `nb_negative_log_likelihood(counts, mu, theta, ...)`: NB log-likelihood.

#### `encoders/atac_encoder.py` -- ATACEncoder
- Bernoulli reconstruction for binarised scATAC peaks.
- `__init__(input_dim, d_token=64, hidden_dims=(1024,256), dropout=0.1)`.
- MLP encoder + Linear decoder (logits).
- `reconstruct(z) -> logits`.
- Standalone: `atac_bernoulli_nll(peak_targets, peak_logits, ...)`.

#### `encoders/methylation_encoder.py` -- MethylationEncoder
- Beta-distributed reconstruction (for beta values in [0,1]) or Gaussian (for M-values).
- `__init__(input_dim, d_token=64, hidden_dims=(512,256), value_kind="beta")`.
- Decoder: `decoder_param_a`, `decoder_param_b` Linear layers.
- `reconstruct(z)`: returns `(alpha, beta)` (Softplus+1e-3) or `(mean, log_var)`.
- Standalone: `methylation_nll(target, params, value_kind, ...)`.

#### `encoders/histone_ptm_encoder.py` -- HistonePTMEncoder
- Gaussian reconstruction. Targets HDAC inhibitor failure case by giving MORT-FM direct histone-mark view.
- `__init__(input_dim, d_token=64, hidden_dims=(256,128), dropout=0.1)`.
- `reconstruct(z) -> Tensor`.

#### `encoders/proteomic_encoder.py` -- ProteomicEncoder
- Gaussian reconstruction with learned per-protein variance.
- `__init__(input_dim, d_token=64, hidden_dims=(1024,256), use_protein_attention=False, attention_chunks=16, attention_heads=4)`.
- Optional protein-aware bottleneck: chunks input into attention_chunks groups, projects each to 128-d, self-attention, flatten.
- `reconstruct(z) -> (mean, log_var)`.
- Standalone: `gaussian_nll(target, params, ...)`.

#### `encoders/phosphoproteomic_encoder.py` -- PhosphoproteomicEncoder
- Gaussian reconstruction in log-magnitude * sign space.
- `__init__(input_dim, d_token=64, hidden_dims=(512,256))`.
- `reconstruct(z) -> (mean, log_var)`.

#### `encoders/clinical_encoder.py` -- ClinicalEncoder
- Handles mixed continuous/categorical/binary features. Missing values encoded as zeros with explicit missing mask appended.
- `__init__(input_dim, d_token=64, hidden_dims=(64,64), append_missing_mask=True)`.
- If `append_missing_mask`, doubles input dim by appending `(1 - isnan)` mask.

#### `encoders/drug_encoder.py` -- DrugEncoder
- Encodes drug context (dose, start_time, end_time, identity, class) + optional chemical embeddings.
- `__init__(n_drugs=32, n_classes=16, d_drug_raw=3, d_token=64, d_chem=0)`.
  - `drug_embed`: Embedding(n_drugs, 16).
  - `class_embed`: Embedding(n_classes, 8).
  - Encoder: Linear(d_drug_raw+16+8+d_chem, 64) -> GELU -> Dropout -> Linear(64, d_token).
- `forward(raw, drug_idx, class_idx, chem_embedding, presence_mask) -> z`.
- `encode_contexts(contexts: Sequence[DrugContext], device) -> Tensor`: inference path.
- **Imports**: `drug_to_index`, `class_to_index` from `resistancemap.data.drug_ontology`; `DrugContext` from `resistancemap.mortfm.schemas`.

#### `encoders/modality_tokenizer.py` -- ModalityTokenizer
- `__init__(self, config: MORTFMConfig)`:
  - Conditionally instantiates encoders based on config flags (`use_rna`, `use_atac`, `use_methylation`, `use_histone_ptm`, `use_proteomics`, `use_phosphoproteomics`, `use_clinical`, `use_drug`).
  - `modality_pos_embed`: Embedding(len(MODALITY_ORDER), d_token) for distinguishing token types.
- `forward(self, batch: MORTBatch) -> (tokens: (N,K,d_token), presence: (N,K), names: List[str])`.
  - Routes each modality through its encoder, adds modality positional embedding, stacks.
  - Missing modalities at runtime yield zero tokens with presence=False.

---

### `identifiable_ode.py` -- Structured, Identifiable Biological ODE (Phase 3)

**Purpose**: Mechanistically interpretable ODE for drug resistance dynamics. K latent dimensions = biological programs (stemness, drug efflux, DNA damage response, immune evasion, metabolic rewiring, anti-apoptosis, epigenetic reprogramming, proliferation).

#### `BiologicalProgram(Enum)`
Values 0-7: DRUG_EFFLUX, DNA_DAMAGE_RESPONSE, STEMNESS, IMMUNE_EVASION, METABOLIC_REWIRING, ANTI_APOPTOSIS, EPIGENETIC_REPROG, PROLIFERATION.

**`PROGRAM_MARKERS`**: Dict mapping each BiologicalProgram to marker gene lists.

#### `IdentifiableODEConfig` (dataclass)
- `n_programs=8`, `state_dim=16`, `hidden_dim=128`, `n_drugs=20`, `drug_embed_dim=32`, `treatment_history_dim=64`.
- Regularization: `sparsity_lambda=0.01`, `diagonal_dominance_lambda=0.1`, `orthogonality_lambda=0.05`, `anchor_lambda=1.0`.
- Lyapunov: `lyapunov_epsilon=0.01`, `lyapunov_lambda=0.5`.
- ODE solver: `ode_solver="dopri5"`, `adjoint=True`, `rtol=1e-5`, `atol=1e-6`.

#### `SparseInteractionMatrix(nn.Module)`
- Parameterized as `A = diag(-exp(log_neg_diag)) + mask * softthresh(W_off, tau)`.
- `__init__(n_programs, init_diagonal=-0.5, prior_mask=None, sign_constraints=None)`.
- `get_matrix(tau=0.0) -> A`: applies prior mask, soft-thresholding, sign constraints.
- `diagonal_dominance_penalty() -> Tensor`: sum of violations where |A_ii| < sum_j!=i |A_ij|.
- `sparsity_loss() -> Tensor`: L1 of off-diagonal.
- `effective_sparsity(threshold=0.01) -> float`.

#### `TreatmentHyperNetwork(nn.Module)`
- Generates time-varying A(t) = A_base + Delta(h(t)).
- `__init__(treatment_history_dim, n_programs, hidden_dim=64, max_perturbation=0.3)`.
- MLP (treatment_history_dim -> hidden -> hidden -> n_programs^2), SiLU activations, clamped to [-max_perturbation, max_perturbation].
- Initialized near zero.

#### `DrugInterventionModule(nn.Module)`
- Drug action as ODE perturbation: g(x, drug) = phi(x, e_drug) * concentration.
- `__init__(state_dim, n_drugs, drug_embed_dim=32, hidden_dim=64)`.
- Drug embeddings: nn.Embedding(n_drugs, drug_embed_dim).
- Interaction network: MLP (state_dim + drug_embed_dim -> hidden -> hidden -> state_dim), SiLU.
- PK model: one-compartment `C(t) = dose * Cmax * exp(-ln2 * t / t_half)` with learnable log_half_life, log_cmax per drug.
- `counterfactual(x0, ode_func, drug_id, dose, t_span) -> trajectory`: solves combined dynamics via torchdiffeq.

#### `LyapunovNet(nn.Module)`
- V(x) = ||phi(x - x*)||^2 where phi is MLP (state_dim -> hidden -> ... -> state_dim) with Tanh activations.
- `forward(x, equilibrium) -> V`: guaranteed V >= 0, V(x*) = 0.
- `lie_derivative(x, equilibrium, f) -> dVdt`: via autograd, should be < 0 for stability.

#### `LyapunovStabilityCertifier(nn.Module)`
- `__init__(state_dim, hidden_dim=64, epsilon=0.01, n_sample_points=64, sample_radius=1.0)`.
- `stability_loss(equilibria, dynamics_fn) -> loss`: mean(max(0, dVdt + epsilon)) over sampled points.
- `certify(equilibrium, dynamics_fn, n_test_points=256, test_radius=0.5) -> (is_stable, worst_dVdt)`.

#### `ProgramEmbeddingLayer(nn.Module)`
- Learnable embeddings for each biological program, initialized near-orthogonal via QR decomposition.
- `orthogonality_penalty() -> Tensor`: ||E E^T - I||_F^2.

#### `StructuredBiologicalODE(nn.Module)`
- Dynamics: `dx/dt = A(t) @ tanh(x_prog) + 0.1*residual_mlp(x) + basal + g(x, drug)`.
- `__init__(config: IdentifiableODEConfig)`:
  - `interaction`: SparseInteractionMatrix(K).
  - `program_embeddings`: ProgramEmbeddingLayer(K, hidden_dim).
  - `residual_mlp`: MLP (D -> hidden -> hidden -> D), SiLU, initialized near zero.
  - Projection layers if D > K: `proj_to_programs`, `proj_from_programs`.
  - `hyper_net`: TreatmentHyperNetwork.
  - `drug_module`: DrugInterventionModule.
  - `lyapunov`: LyapunovStabilityCertifier.
  - `anchor_heads`: ModuleList of Linear(1, 1) per program.
  - `basal`: Parameter(D).
- `set_treatment_context(drug_id, dose, treatment_history)`: sets runtime context for ODE integration.
- `autonomous_dynamics(x) -> dxdt`: without drug perturbation.
- `forward(t, x) -> dxdt`: compatible with torchdiffeq interface.
- `integrate(x0, t_span, method=None) -> trajectory`: uses odeint_adjoint or odeint.
- `find_equilibria(x0, max_iter=500, tol=1e-5, lr=0.01) -> equilibria`: minimizes ||f(x)||^2 via Adam.
- `anchor_loss(x, anchor_targets) -> loss`: MSE for anchored dimensions.

#### `IdentifiableODELoss(nn.Module)`
- 7 components: trajectory_mse, sparsity, diagonal_dominance, orthogonality, anchor, lyapunov, hyper_reg.
- `forward(model, pred_trajectory, true_trajectory, equilibria, anchor_targets) -> (total, loss_dict)`.

---

### `causal_fusion.py` -- Causally Structured Multi-Modal Fusion

**Purpose**: Replaces symmetric cross-attention with fusion grounded in biological causal hierarchy: Genetic -> Epigenetic -> Transcriptomic -> Proteomic.

#### `Modality(IntEnum)`: GENETIC=0, EPIGENETIC=1, TRANSCRIPTOMIC=2, PROTEOMIC=3.

**`BIOLOGICAL_CAUSAL_EDGES`**: 6 directed edges encoding the DAG (including skip connections like GENETIC -> PROTEOMIC).

#### `CausalFusionConfig` (dataclass)
- `modality_dims`: {GENETIC: 128, EPIGENETIC: 256, TRANSCRIPTOMIC: 512, PROTEOMIC: 256}.
- `hidden_dim=128`, `n_heads=4`, `n_layers=2`, `dropout=0.1`.
- `confounder_dim=32`, `n_confounders=4`, `noise_dim=16`, `intervention_noise_std=0.01`.

#### `CausalModalityGraph(nn.Module)`
- Fixed biological DAG. Computes adjacency, transitive closure (ancestor_mask), direct_parent_mask as registered buffers.
- `get_attention_mask(use_ancestors=True)`: (M, M) binary mask.
- `get_intervention_mask(do_modality)`: do-calculus, blocks incoming edges to intervened modality.
- `parents()`, `children()`, `is_ancestor()`.

#### `ConfounderEncoder(nn.Module)`
- `__init__(n_confounders=4, max_categories=32, confounder_dim=32, output_dim=64)`.
- Per-confounder Embedding(max_categories, confounder_dim) + aggregate MLP (n_conf*conf_dim -> output_dim, LayerNorm, SiLU).

#### `SCMLayer(nn.Module)`
- Structural equation: Z_m = f_m(Z_{pa(m)}, U_m, C) with gated residual connection.
- `__init__(modality, hidden_dim, n_parents, noise_dim=16, confounder_dim=64)`.
- Input: own representation + parent representations + noise + confounder.
- `structural_fn`: MLP (input_dim -> 2*hidden -> hidden, LayerNorm, GELU, Dropout).
- `residual_gate`: Linear(2*hidden, 1) -> Sigmoid.

#### `CausalAttentionBlock(nn.Module)`
- Multi-head attention with causal masking (blocked positions -> -inf).
- `__init__(hidden_dim, n_heads=4, dropout=0.1)`.
- Standard Q/K/V projections + FFN (4x expansion) with pre-norm, residual connections.
- `get_attention_weights()` for interpretability.

#### `CausalFusionNetwork(nn.Module)`
- `__init__(config: CausalFusionConfig)`:
  - Per-modality input projections (Linear -> LayerNorm -> GELU).
  - Modality type embeddings: Embedding(4, hidden_dim).
  - Stacked CausalAttentionBlocks.
  - SCM layers (one per modality).
  - ConfounderEncoder.
  - Output projection: Linear(H*4, 2H) -> LayerNorm -> GELU -> Linear(2H, H).
- `forward(modality_inputs, confounder_ids) -> (fused, modality_reprs)`.
  - Project modalities -> add modality embeddings -> stack -> causal attention layers -> SCM refinement in topological order -> concat all -> project.
- `set_intervention()` / `clear_intervention()` / `interventional_query()`: do-calculus interface.
- `causal_effect()`: ACE = E[Y|do(X=x)] - E[Y|do(X=0)].

---

### `disentangled_vae.py` -- Beta-Total Correlation VAE with Anchored Dimensions

**Purpose**: Produces disentangled, interpretable latent factors for single-cell multi-omics. First K_anchor dimensions supervised to predict known biology.

#### `AnchorSpec` (dataclass)
- `dim_index`, `name`, `task` ("classification"/"regression"), `n_classes`, `marker_genes`, `weight`.

#### `DisentangledVAEConfig` (dataclass)
- `input_dim=2000`, `latent_dim=32`, `hidden_dims=[512, 256, 128]`.
- `beta=4.0` (TC weight), `alpha=1.0` (MI weight), `gamma=1.0` (dim-KL weight).
- Default anchors: drug_efflux(regression), cell_cycle(classification, 4 classes), stemness(regression), dna_damage(regression).
- `tc_estimation="minibatch"`, `dropout=0.1`, `activation="gelu"`, `batch_norm=True`.

#### `Encoder(nn.Module)` / `Decoder(nn.Module)`
- Encoder: input_dim -> [hidden layers with BN/Dropout/activation] -> fc_mu + fc_logvar. Logvar clamped [-10, 10], initialized with constant bias -2.0.
- Decoder: latent_dim -> reversed hidden layers -> output_dim.

#### `AnchorHead(nn.Module)`
- Takes single latent dimension -> predicts biological readout.
- Classification: Linear(1, 32) -> ReLU -> Linear(32, n_classes).
- Regression: Linear(1, 32) -> ReLU -> Linear(32, 1).

#### `DisentangledVAE(nn.Module)`
- `__init__(config)`: Encoder, Decoder, AnchorHead per anchor_spec.
- Loss: `recon (MSE) + alpha*MI + beta*TC + gamma*dim_KL + sum(anchor_weight * anchor_loss)`.
- TC estimated via minibatch weighted sampling (Chen et al. 2018): `_log_density_gaussian` -> `estimate_tc_minibatch`.
- `forward(x, anchor_targets=None) -> (total_loss, loss_dict)`.

#### `DCIMetrics`
- Static `compute(z, targets, method="gradient")`: returns disentanglement, completeness, informativeness.
- Methods: `_correlation_importance` (Pearson) or `_gradient_importance` (least squares weights).

#### `TraversalGenerator`
- `generate(vae, reference_z, n_steps=11, traverse_range=3.0, dims)`: varies each dim from -range to +range, decodes.
- `traversal_sensitivity(vae, reference_z)`: Var(decoded traversal) per dim.

---

### `bistability.py` -- Jacobian Eigenvalue Analysis + Dual-Head GNN (Gap 4.3)

**Purpose**: Extends chromatin bistability with Jacobian eigenvalue analysis, dual-head GNN (IC50 + reversibility), reversibility classification, and stability-aware ODE dynamics.

#### `JacobianEigenvalueAnalyzer(nn.Module)`
- `__init__(ode_fn=None, state_dim=2)`.
- `compute_eigenvalues(x, t, method)`: uses `torch.linalg.eig` (NOT eigh -- Jacobians are non-symmetric).
- `compute_basin_depth(x, t)`: basin_depth = clamp(-max_real_part, min=0).
- `forward(x, t, return_jacobians=False)`: returns basin depths.

#### `DualHeadGNN(nn.Module)`
- `__init__(protein_dim=256, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1)`.
- Shared backbone: input_proj + LayerNorm + 4 GATConv layers with residual + GELU + Dropout.
- Head 1 (IC50): Linear(256,128) -> GELU -> Dropout -> Linear(128,64) -> GELU -> Dropout -> Linear(64,1) -> Softplus.
- Head 2 (Reversibility): Linear(256+4, 128) -> GELU -> Dropout -> Linear(128,64) -> GELU -> Dropout -> Linear(64,1) -> Sigmoid. The +4 is eigenvalue statistics (max, min, trace, norm).
- GlobalAttention pooling.
- `forward(x, edge_index, eigenvalues=None, batch_idx=None) -> (ic50_pred, reversibility_pred)`.

#### `ReversibilityClassifier(nn.Module)`
- 4 classes: reversible_fast (S>=0.75), reversible_slow, partially_reversible, irreversible (S<0.25 or TP53 loss).
- `__init__(num_mutation_features=5)`: MLP (1+5 -> 64 -> 32 -> 4).
- `forward(reversibility_score, mutations) -> (logits, predicted_class)`.

#### `BistabilityIntegratedLoss(nn.Module)`
- `Loss = 0.6*L_IC50(MSE) + 0.3*L_reversibility(BCE) + 0.1*L_consistency(MSE)`.
- Consistency: penalizes mismatch between normalized IC50 and reversibility score.

#### `StabilityAwareODEDynamics(nn.Module)`
- `dtheta/dt = f(theta) - alpha(S) * (theta - theta_ref)`.
- `alpha_network`: MLP (1 -> 32 -> 16 -> 1, Softplus). Maps reversibility score to damping coefficient.
- High S (reversible) -> weak damping; low S (irreversible) -> strong damping.

---

### `cell_state_plasticity.py` -- Multi-Scale Resistance Predictor (Gap 4.2)

**Purpose**: Integrates three plasticity scales: transcriptomic noise (fast, SDE), epigenetic bistability (slow, double-well potential), genetic mutation accumulation (irreversible ratchet).

#### `TranscriptomicNoiseModel(nn.Module)`
- SDE: `dX = f(X) dt + sigma(X) dW` with mean-reverting drift.
- `__init__(state_dim=32, hidden_dim=16, noise_scale=0.1, mean_reversion_strength=0.5)`.
- `drift_net`: MLP -> state_dim. `diffusion_net`: MLP -> state_dim, Softplus.
- Learnable `equilibrium` parameter.
- `forward(x0, t_span, n_steps, return_trajectory)`: Euler-Maruyama integration.
- `compute_transition_probability(x_start, x_target, horizon)`: Monte Carlo (100 samples, threshold=0.5).

#### `EpigeneticBistabilityModel(nn.Module)`
- Double-well potential: V(E) = a*E^4 - b*E^2.
- `potential_net`: MLP (4 -> hidden -> hidden -> 2, Softplus). 4 inputs = chromatin modifier abundances (EZH2, DNMT1, TET1, etc.).
- Learnable `log_temperature`.
- `forward(e0, chromatin_modifiers, t_span, n_steps)`: Euler integration with gradient descent on potential.
- `kramers_escape_rate()`: k = omega0 * exp(-barrier / T) where barrier = b^2/(4a).
- `transition_probability(chromatin_modifiers, time_horizon)`: 1 - exp(-k * t).

#### `GeneticMutationAccumulator(nn.Module)`
- Ratchet: `dC/dt = mu * h(C)` where h is fitness landscape (sigmoid saturation).
- `__init__(n_drivers=5, hidden_dim=16)`.
- `fitness_net`: MLP (n_drivers -> hidden -> n_drivers, Sigmoid).
- Learnable `log_mutation_rates` per driver (init -3.0).

#### `WaddingtonLandscape(nn.Module)`
- Potential: phi(E) = -0.5 * E^T W E + lambda * ||E||^2.
- Learnable `grn_weights` (gene_dim x gene_dim) and `log_lambda`.
- `find_attractors(e0, n_iterations=100, lr=0.01)`: gradient descent on potential.
- `compute_landscape_distance(e1, e2)`: geodesic approximation.

#### `ResistanceSignatureBank(nn.Module)` -- FIX #2
- Replaces random target with learned drug-class-specific resistance signatures.
- `__init__(state_dim=32, n_drug_classes=8)`.
- Learnable `signatures` (n_drug_classes, state_dim) + `default_signature` + `context_refinement` MLP.
- `get_signature(expression_state, drug_class_idx=None)`: base_signature + 0.1 * refinement.

#### `MultiScaleResistancePredictor(nn.Module)`
- Combines all three scales: `P_resist = softmax(w) . [P_transcriptomic, P_epigenetic, P_genetic]`.
- `__init__(state_dim=32, n_drivers=5, hidden_dim=16, n_drug_classes=8)`.
- `forward(expression_state, epigenetic_state, mutation_counts, chromatin_modifiers, ...)`: returns resistance probability in [0, 1].

#### `PlasticityDiagnostics` (plain class)
- Static methods: `transcriptomic_noise_sensitivity()`, `epigenetic_stability_index()`, `generalized_lyapunov()`.

---

### `clinical_anchored_ode.py` -- Clinical Anchored Neural ODE (Gap 4.1)

**Purpose**: Bridges cross-sectional multi-omic snapshots to continuous temporal trajectories with treatment conditioning.

#### `PseudotimeEncoder(nn.Module)`
- MLP (input_dim -> hidden -> hidden -> 1, Sigmoid). Output tau in [0, 1].

#### `TreatmentEncoder(nn.Module)`
- MLP (treatment_dim -> conditioning_dim -> conditioning_dim, GELU).

#### `ClinicalAnchoredODEFunc(nn.Module)`
- Neural ODE: `dz/dt = f_theta(z, t_embedding, treatment_embedding)`.
- `__init__(latent_dim=64, treatment_conditioning_dim=64, hidden_dim=128, n_layers=3)`.
- Time embedding: Linear(1, hidden) -> GELU -> Linear(hidden, hidden).
- ODE blocks: MLP (latent_dim + hidden -> hidden, repeated n_layers times).
- FiLM conditioning: `h = gamma(treatment) * h + beta(treatment)`.
- Output: Linear(hidden, latent_dim).

#### Loss Modules:
- **`VelocityConsistencyLoss`**: 1 - cosine_similarity(dzdt, external_velocity).
- **`ClinicalAnchorLoss`**: ||z(tau_clinical) - z_measured||^2 at landmark timepoints, with linear interpolation.
- **`TemporalSmoothnessLoss`**: ||d^2z/dtau^2||^2 via finite differences.

#### `ClinicalAnchoredTrajectoryModule(nn.Module)`
- End-to-end: encoder -> pseudotime encoder -> treatment encoder -> ODE solver -> decoder -> multi-scale losses.
- `__init__(input_dim, latent_dim=64, output_dim=None, treatment_dim=4, encoder_hidden_dims=[256,128], decoder_hidden_dims=[128,256], ode_hidden_dim=128, ode_n_layers=3, integration_steps=50, ...)`.
- Learnable loss weights: `lambda_velocity`, `lambda_anchor`, `lambda_smooth` (wrapped in Softplus).
- Xavier uniform init.
- `forward(x, treatment, x_target, ...)`: returns dict with z_trajectory, predictions, pseudotime, all loss components.
- Uses `torchdiffeq.odeint_adjoint` with `method="dopri"`.

---

### `cell_communication.py` -- Cell-Cell Communication (Gap 4.4)

**Purpose**: Heterogeneous GNN for ligand-receptor communication and spatial decay weighting.

#### `SpatialDecayWeighting(nn.Module)`
- `w = exp(-d / sigma)` where d is L2 distance in embedding space. Learnable sigma.

#### `LRInteractionGraph(nn.Module)`
- Builds heterogeneous graph: protein nodes (PPI edges) + cell-type nodes (L-R edges weighted by spatial decay * CellChat probability).
- `build_heterogeneous_edges(cell_embeddings, cellchat_probabilities) -> dict`.

#### `DualLayerHeteroGNN(nn.Module)`
- Layer 1: GAT for protein PPI (2 GATConv layers, 8 heads).
- Layer 2: R-GCN for cell-type communication (2 RGCNConv layers, num_bases=4).
- Cross-layer: MultiheadAttention(hidden_dim, 4 heads) from cell types to proteins.

#### `CommunicationAwareResistancePredictor(nn.Module)`
- `__init__(hidden_dim=256, num_proteins=1000, num_cell_types=15, num_drugs=300)`.
- Protein encoder + cell encoder + drug embedding + fusion (3*hidden -> hidden -> hidden).
- Resistance head (Sigmoid) + uncertainty head (Softplus).

---

### `clonal_heterogeneity.py` -- Game-Theoretic Clonal Dynamics (Gap 4.5)

**Purpose**: Replicator dynamics ODE for clonal population frequency evolution under therapy.

#### `ClonalProfile` (dataclass)
- Per-clone: clone_id, genotype, frequency, parent_id, phenotypic signatures (HIF-1alpha, UPR, OXPHOS, proliferation, immune_evasion, stemness), fitness_coefficient, drug_sensitivity dict.

#### `CloneSpecificEncoder(nn.Module)`
- `__init__(phenotype_dim=6, genotype_vocab_size=256, genotype_embedding_dim=64, hidden_dim=128, output_dim=128, dropout=0.2)`.
- Genotype embedding + 2-layer MLP with residual connection.

#### `ReplicatorDynamicsODE(nn.Module)`
- `dphi_c/dt = phi_c * (f_c - phi_bar_f)` where f_c is fitness from clone embeddings + drug concentration.
- Fitness network: Linear(embedding_dim+1, hidden) -> ReLU -> Dropout -> Linear -> ReLU -> Linear(1) -> Tanh.
- Supports torchdiffeq ("dopri") or Euler fallback. Post-integration simplex renormalization.

#### `AdaptiveTherapyScheduler(nn.Module)`
- `decision_net`: MLP (n_clones+1 -> hidden -> hidden -> 1, Sigmoid).
- `recommend_dose(frequencies, current_dose) -> next_dose`.

#### `CloneSpecificResistancePrediction(nn.Module)` (CSRP-MM)
- Full pipeline: CloneSpecificEncoder + ReplicatorDynamicsODE + per-clone resistance head (embedding+1 -> hidden -> 1, Sigmoid) + AdaptiveTherapyScheduler.
- `forward(clone_profiles, drug_schedule, genotypes) -> dict` with clone_embeddings, frequencies, resistance_probs, dominant_clone_idx, adaptive_doses.

---

### `domain_adaptation.py` -- Cell-Line-to-Patient Transfer (Gap 4.6)

**Purpose**: Multi-phase adversarial training to transfer drug response models from cell lines (CCLE) to patients (TCGA, PDX).

#### `GradientReversalFunction` / `GradientReversalLayer`
- Same pattern as vae.py but separate implementation.

#### `DomainAdversarialVAE(nn.Module)`
- Encoder: 10000 -> 256 -> 128 -> 64 (latent_dim), with BatchNorm + GELU + Dropout.
- Decoder: 64 -> 128 -> 256 -> 10000.
- Domain classifier with GRL: 64 -> 32 -> 8 -> 1 (LeakyReLU).
- Drug response head: 64 -> 32 -> 1 (ReLU).
- Combined loss: alpha*L_VAE + beta*L_domain + gamma*L_supervised + delta*L_diversity.

#### `WassersteinDomainLoss(nn.Module)`
- WGAN-GP: Wasserstein distance + gradient penalty (weight=10.0).

#### `DriftDetector(nn.Module)`
- Monitors domain classifier confidence/entropy. Triggers retraining when drift detected.
- Rolling window (100), thresholds: confidence > 0.8, entropy < 0.3.

#### `MultiPhaseDomainAdaptationTrainer`
- Phase 1: CCLE supervised (VAE + response head).
- Phase 2: Patient DANN (domain-adversarial loss).
- Phase 3: PDX bridge (optional intermediate).
- Phase 4: Validation on held-out patient cohort.
- Uses `torch.amp.autocast` + `GradScaler`, gradient clipping.

---

### `per_drug_head.py` -- Per-Drug Specialized Regression Head

**Purpose**: Replaces shared `Linear(64, n_drugs)` with independent `Linear(64, 1)` per drug + inverse-variance loss weighting to prevent HDAC-class gradient dominance.

#### `variance_weights(drug_sensitivity_train, eps, clip) -> Tensor`
- Computes `1/Var(y_d_train)` per drug, clipped to [0.1, 10.0], normalized to mean 1.

#### `PerDrugHead(nn.Module)`
- `__init__(fusion_dim, n_drugs, hidden=64, dropout=0.1, loss_weights=None)`.
- `trunk`: Linear(fusion_dim, hidden) -> ReLU.
- `heads`: ModuleList of n_drugs independent (Dropout -> Linear(hidden, 1)).
- `loss_weights`: registered buffer, travels with `.to(device)`.
- `forward(fused) -> (B, n_drugs)`: concatenates per-drug outputs.
- `masked_weighted_mse(pred, target) -> loss`: NaN-aware per-drug MSE weighted by inverse-variance. Uses `torch.where(mask, target, pred.detach())` to avoid NaN gradients.

---

### `rl_treatment_optimization.py` -- Unified RL Treatment Optimization (Gap 7.x)

**Purpose**: Offline RL for personalized MM treatment selection. 21 treatment actions, ~500-dim state space, multi-component reward.

#### Constants
- `TREATMENT_ACTIONS`: 21 treatments (VRd, Dara-VRd, ..., Refer_trial).
- `ACTION_IDX`: name -> index mapping.
- `RLBatch(NamedTuple)`: states, actions, rewards, next_states, dones, feasibility_masks.

#### `RLStateEncoder(nn.Module)`
- `__init__(input_dim=500, hidden_dim=256, output_dim=256, dropout=0.1)`.
- 3-layer MLP with LayerNorm + GELU + Dropout.

#### `FeasibilityMask(nn.Module)`
- Clinical constraints: CAR-T once only, eGFR < 30 excludes nephrotoxic, ECOG >= 3 excludes intensive.
- `forward(batch_size, device, egfr, ecog, prior_cart_use) -> (B, num_actions) binary mask`.

#### `TreatmentActor(nn.Module)`
- `__init__(state_dim=256, hidden_dim=256, num_actions=21, dropout=0.1)`.
- 2-layer MLP with LayerNorm + GELU + action_head Linear.
- `forward(states, feasibility_mask) -> (logits, log_probs)`: masks infeasible actions to -inf.
- `sample_action()`: multinomial sampling with mask.

#### `TreatmentCritic(nn.Module)`
- Ensemble of MLPs (state_dim -> hidden -> hidden -> 1).
- `compute_disagreement(states) -> uncertainty`: std across ensemble.

#### `IQLTrainer(nn.Module)`
- Implicit Q-Learning (Kostrikov et al. 2022) + CQL regularization.
- `__init__(state_dim=256, hidden_dim=256, num_actions=21, lr_actor=1e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, beta=3.0, cql_weight=1.0)`.
- Actor + critic_v + critic_q + critic_q_target.
- `train_step(batch) -> losses dict`: V-loss (TD), Q-loss (TD + CQL), Policy-loss (advantage-weighted log-likelihood).
- Soft target update: `theta_target <- tau*theta + (1-tau)*theta_target`.

#### `RewardCalculator(nn.Module)`
- 8-component reward: `w1*deltaOS + w2*MRD_neg - w3*toxicity - w4*EMD - w5*CNS + w6*QoL - w7*fairness - w8*uncertainty`.
- Default weights: w_os=10, w_mrd=5, w_tox=3, w_emd=2, w_cns=2, w_qol=2, w_fair=1, w_unc=0.5.

#### `RLTreatmentOptimizer(nn.Module)`
- Top-level: state_encoder + actor + critic + feasibility_mask_module.
- `forward(patient_features, egfr, ecog, prior_cart_use, deterministic=True) -> dict`: returns action, action_name, value, log_prob, advantage, feasibility_mask, state.
- `explain_action(action_idx, patient_features) -> dict`: human-readable explanation.

---

### `rl/ope.py` -- Off-Policy Evaluation

**Purpose**: Safety gate for RL module. OPE must pass before recommendations are emitted.

#### `PolicyViolation(RuntimeError)`: positivity violated.

#### `OPEResult` (dataclass): wis, doubly_robust, positivity_violation_rate, agreement_gap, passed, notes.

#### Functions (numpy-based):
- `weighted_importance_sampling(behavior_logp, target_logp, rewards, clip=10.0) -> float`: self-normalised WIS.
- `doubly_robust(behavior_logp, target_logp, rewards, q_hat, v_hat, clip=10.0) -> float`.
- `positivity_check(behavior_logp, min_prob=1e-3) -> float`: fraction below min_prob.
- `evaluate(behavior_logp, target_logp, rewards, q_hat, v_hat, positivity_threshold=0.05, agreement_tolerance=0.1) -> OPEResult`: gate passes iff positivity_ok AND agreement_ok AND both estimators finite.

---

### Model Composition / Cross-References

1. **MORT-FM Architecture** (foundation model):
   - `ModalityTokenizer` instantiates per-modality encoders (RNA, ATAC, methylation, histone PTM, proteomics, phosphoproteomics, clinical, drug) based on `MORTFMConfig` flags.
   - `MultiOmicFoundationFusion` wraps the tokenizer, adds [CLS]/patient/time tokens, runs TransformerEncoder, produces z0 + modality gates.
   - The fusion imports `MORTFMConfig`, `FoundationState`, `MORTBatch` from `resistancemap.mortfm.schemas`.

2. **Legacy v6 Architecture** (pre-MORT-FM):
   - `ProteomeToEpigenomeVAE` (vae.py) -> `MemoryStabilityScorer` / `TrajectoryForecaster` (trajectory.py) -> `ProteinNetworkPropagator` (protein_network.py) -> `ResistanceMapFusion` (fusion.py).

3. **Phase 3 additions** (identifiable ODE + causal fusion + disentangled VAE):
   - `StructuredBiologicalODE` contains `DrugInterventionModule`, `SparseInteractionMatrix`, `TreatmentHyperNetwork`, `LyapunovStabilityCertifier`, `ProgramEmbeddingLayer`.
   - `CausalFusionNetwork` contains `CausalModalityGraph`, `SCMLayer`, `CausalAttentionBlock`, `ConfounderEncoder`.
   - `DisentangledVAE` contains `Encoder`, `Decoder`, `AnchorHead`.

4. **Gap modules** (4.1-4.6, 7.x):
   - `ClinicalAnchoredTrajectoryModule` (4.1) is standalone encoder-ODE-decoder pipeline.
   - `MultiScaleResistancePredictor` (4.2) composes `TranscriptomicNoiseModel`, `EpigeneticBistabilityModel`, `GeneticMutationAccumulator`, `WaddingtonLandscape`, `ResistanceSignatureBank`.
   - `DualHeadGNN` + `JacobianEigenvalueAnalyzer` + `StabilityAwareODEDynamics` (4.3) extend bistability.
   - `DualLayerHeteroGNN` + `CommunicationAwareResistancePredictor` (4.4) for cell communication.
   - `CloneSpecificResistancePrediction` (4.5) wraps `CloneSpecificEncoder` + `ReplicatorDynamicsODE` + `AdaptiveTherapyScheduler`.
   - `DomainAdversarialVAE` + `MultiPhaseDomainAdaptationTrainer` (4.6) for domain adaptation.
   - `RLTreatmentOptimizer` + `IQLTrainer` (7.x) for treatment optimization with `rl/ope.py` safety gate.

5. **Dependency on external packages**:
   - `torchdiffeq`: trajectory.py (odeint), clinical_anchored_ode.py (odeint_adjoint), identifiable_ode.py (odeint, odeint_adjoint), clonal_heterogeneity.py (odeint). All with graceful ImportError fallback.
   - `torch_geometric`: bistability.py (GATConv, GlobalAttention), cell_communication.py (GATConv, RGCNConv, GlobalAttention), protein_network.py (GATConv, GlobalAttention). All with HAS_PYG guard.
   - `transformers` (HuggingFace): protein_network.py ESM2Embedder.
   - `resistancemap.config`: trajectory.py (StabilityConfig), vae.py (VAEConfig).
   - `resistancemap.utils.checkpoint`: trajectory.py, vae.py (CheckpointManager).
   - `resistancemap.mortfm.schemas`: foundation_fusion.py, drug_encoder.py, modality_tokenizer.py (MORTFMConfig, MORTBatch, FoundationState, ModalityName, MODALITY_ORDER, DrugContext).
   - `resistancemap.data.drug_ontology`: drug_encoder.py (drug_to_index, class_to_index, materialise_drug_context).

---

## Appendix C: MORT-FM Foundation Model (mortfm/)

# mortfm/ Directory Report

> `resistancemap/mortfm/` -- 59 Python files, ~8000 lines
> Multi-Omic Resistance Trajectory Foundation Model (MORT-FM), the v15+ research-track namespace.

---

## 1. Top-Level Module Files

### `__init__.py`
- Re-exports all core schema types from `schemas.py`.
- Exports: `DrugContext`, `FoundationState`, `MORTBatch`, `MORTFMConfig`, `ModalityName`, `ModalityTensor`, `PatientCellSnapshot`, `ResistanceOutcome`, `TemporalTrainingPair`, `TrajectoryPrediction`.

### `schemas.py`
Central typed primitives. Minimal-dependency (only torch + stdlib).

- **`ModalityName(str, Enum)`** -- Canonical modality identifiers: RNA, ATAC, METHYLATION, HISTONE_PTM, PROTEOMICS, PHOSPHOPROTEOMICS, CLINICAL, DRUG.
- **`MODALITY_ORDER`** -- Stable tuple of `ModalityName` values in the canonical order the foundation encoder emits tokens.
- **`ModalityTensor` (dataclass)** -- Wraps a single-modality feature tensor with provenance. Fields: `name: str`, `values: Tensor` (2D or 3D), `feature_names: List[str]`, `mask: Optional[Tensor]`, `batch_id: Optional[Tensor]`. Validates shape consistency in `__post_init__`. Has `.to(device)`, `.n_samples`, `.feature_dim` properties.
- **`DrugContext` (dataclass)** -- Treatment-pressure descriptor. Fields: `drug_name`, `drug_class`, `target_genes: List[str]`, `target_proteins: List[str]`, `dose`, `start_time`, `end_time`, `combination_id`, `molecular_embedding: Optional[Tensor]`. Has `.to(device)`.
- **`PatientCellSnapshot` (dataclass)** -- One patient/cell at one timepoint, input-only (no labels). Fields: `patient_id`, `sample_id`, `disease`, `timepoint`, `treatment_id`, `cell_id`, plus one `Optional[ModalityTensor]` per modality and `drug: Optional[DrugContext]`. Method `available_modalities() -> List[ModalityName]`.
- **`ResistanceOutcome` (dataclass)** -- Patient-level supervision target. Fields: `patient_id`, `baseline_time: float`, `event_time: Optional[float]`, `censored: bool` (explicit flag, never silent), `resistance_label: Optional[int]`, `future_state: Optional[Tensor]`, `drug_response: Optional[Tensor]`, `pathway_labels: Optional[List[str]]`. Methods: `has_survival_label()`, `has_future_state()`.
- **`TemporalTrainingPair` (dataclass)** -- `(x_t: PatientCellSnapshot, x_t_delta: Optional[PatientCellSnapshot], outcome: ResistanceOutcome)`. Method `has_any_supervision()`.
- **`MORTBatch` (dataclass)** -- THE canonical batch object for the model. Per-modality `Optional[Tensor]` fields (rna, atac, methylation, histone_ptm, proteomics, phosphoproteomics, clinical, drug), `modality_mask: Dict[str, Tensor]`, provenance (patient_ids, cell_ids, time), supervision (future_state, resistance_label, event_time, event_observed, pathway_targets, drug_response). **v19 additions**: `future_snapshot_batch: Optional[MORTBatch]` (encoder-consistent follow-up), `delta_t_days: Optional[Tensor]`, `graph: Any`. Methods: `batch_size`, `available_modalities()`, `to(device)`, `validate_for_loss(loss_name)` -- validates survival/trajectory/resistance_state/drug_response/latent_future_state supervision, raises ValueError on missing.
- **`FoundationState` (dataclass)** -- Encoder output: `z0: Tensor` (N, d_latent), `modality_embeddings: Dict[str, Tensor]`, `attention_maps: Dict[str, Tensor]`, `modality_gates: Optional[Tensor]`, `uncertainty: Optional[Tensor]`.
- **`TrajectoryPrediction` (dataclass)** -- Full inference payload: `z_path`, `z_samples`, `resistance_state_logits`, `hazard`, `survival_curve`, `trajectory_mean`, `trajectory_cov`, `pathway_protein_scores`, `pathway_edge_scores`, `drug_specific_risk`, `counterfactual_rankings`, `uncertainty`.
- **`MORTFMConfig` (dataclass)** -- Typed mirror of `configs/mortfm.yaml`. Identity (model_name, version). Modality flags (use_rna, use_atac, etc.). Encoder dimensions (rna_input_dim=5000, atac_input_dim=50000, etc.). Foundation latent (d_latent=128, d_token=64, fusion_layers=4, fusion_heads=4). Dynamics (dynamics_kind="neural_sde", integration_time=12.0, n_time_grid=25, sde_n_mc_samples=16). Potential (use_potential=True, n_attractors=4). Survival (survival_kind="discrete", survival_n_bins=12). 13 loss weights (lambda_*). Training (batch_size=64, pretrain/finetune/trajectory/survival epochs, lr=1e-4, modality_dropout_p=0.25). Guardrails (min_paired_n_for_longitudinal_claim=100, min_patient_n_for_survival_claim=200).

### `model.py`
- **`class MORTFM(nn.Module)`** -- End-to-end model composing: encoder+fusion -> dynamics -> heads.
  - **`__init__(config, n_pathway_proteins, n_pathway_edges, n_pathways, n_drug_candidates, n_resistance_states, return_attention)`**: Builds `MultiOmicFoundationFusion`, `WaddingtonPotential`, legacy `TrajectorySampler`, 7 legacy heads (ResistanceStateHead, TimeToResistanceHead, TrajectoryDistributionHead, PathwayRouteHead, DrugSpecificTrajectoryRiskHead, EvidentialUncertaintyHead, CounterfactualInterventionHead). Builds v16 LENS modules: `GraphEnergyResistanceSDE`, `ResistanceBasin`, `HittingTime`, `LatentToGraphProjector`, `CompetingRiskHead`. Uses a shared `CanonicalTimeGridConfig` (v19 Phase 7). Bug B19 verification: asserts d_graph consistency between SDE and projector at construction time.
  - **`encode(batch) -> FoundationState`**: Foundation forward pass.
  - **`rollout(state, time_grid, drug, graph, n_samples)`**: TrajectorySampler.sample_paths.
  - **`forward_foundation(batch) -> FoundationState`**: Lightweight encode-only for stages A/B/C.
  - **`forward_legacy(batch, time_grid, n_traj_samples, graph, compute_counterfactuals) -> TrajectoryPrediction`**: Full legacy path through TrajectorySampler + all 7 heads.
  - **`forward(batch, clinical, drug, graph_emb, use_graph_projector) -> dict`**: CANONICAL v18 path. LENS-style: SDE rollout + competing-risk hazard + basin probs + hitting time. Returns dict with keys: z0, z_traj, z_samples, t_grid, hazard, survival_curve, cif_per_event, basin_probs, hitting_cdf, hitting_mean_tau, hitting_frac_hit.

### `trainer.py`
- **`TrainingMetrics` (dataclass)** -- epoch, stage, train_loss, val_loss, components, wall_time_s.
- **`MissingSupervisionError(RuntimeError)`** -- v18.2 strict-supervision policy; raised when supervision coverage < threshold.
- **`StageSupervisionReport` (dataclass)** -- Per-stage supervision audit: stage, n_batches, n_supervised_batches, missing_reasons, coverage property.
- **`STRICT_SUPERVISION_COVERAGE`** -- Dict mapping stages E->0.80, F->0.90, G->0.70.
- **`VALID_STAGES`** -- ("A", "B", "C", "D", "E", "F", "G", "H").
- **`stage_weight_overrides(stage, config) -> Dict[str, float]`** -- Per-stage loss-weight selection. Stage A: recon. Stage B: recon+masked+contrastive. Stage C: drug. Stage D: contrastive. Stage E: trajectory+landscape. Stage F: survival+state. Stage G: pathway. Stage H: survival+calibration.
- **`class MORTFMTrainer`** -- One trainer for all stages.
  - **`__init__`**: Takes model, config, train_loader, val_loader, graph, device. Creates AdamW optimizer, GradScaler.
  - **`with_canonical` (classmethod)**: Factory that constructs a MORTFMTrainer with a CanonicalMORTFMTrainer attached, routing stages E/F/G/H through canonical forward().
  - **`_losses_for_batch(batch, stage)`**: Computes per-stage losses. v19: routes to canonical_trainer for E/F/G/H if attached. Legacy path: forward_legacy, then per-stage loss assembly (recon via NB, masked modality, contrastive, drug response, trajectory via future_snapshot_batch or future_state, survival, resistance state, pathway, landscape regularization).
  - **`train_epoch(stage, epoch) -> TrainingMetrics`**: Full training loop with AMP, gradient clipping, strict supervision enforcement.
  - **`validate(stage) -> float`**: No-grad validation loop.
  - **`fit_stage(stage, n_epochs, acceptance_level, acceptance_pairs)`**: Runs acceptance gate, then train loop.
  - **`fit_all(stages)`**: Multi-stage curriculum with per-stage epoch counts and checkpoint saves.
  - **`save_checkpoint / load_checkpoint`**.

### `canonical_trainer.py`
- **`STAGE_VALIDATE_FOR_LOSS_KEY`** -- Maps stages E/F/G/H to their `validate_for_loss` key.
- **`CanonicalEpochMetrics` (dataclass)** -- epoch, stage, loss_total, n_batches, per_term_n_supervised, wall_time_s, val_loss. Has `.to_dict()`.
- **`class CanonicalMORTFMTrainer`** -- Drives MORT-FM through the canonical LENS dict surface exclusively (no forward_legacy).
  - **`__init__`**: Takes model, cfg, train_loader, val_loader, device, strict/strict_supervision, weights, graph_emb_fn.
  - **`_default_weights(cfg)`**: Pulls lambda_trajectory, lambda_pathway, lambda_survival from config.
  - **`_canonical_forward(batch) -> Dict`**: Calls `model(batch, clinical=..., drug=..., graph_emb=...)`. Cleans NaN in drug/clinical tensors.
  - **`_drug_padded(batch)`**: Clips/pads drug tensor to width 8, zeros NaN.
  - **`_clean_clinical(clinical)`**: Clips/pads clinical tensor to width 5, zeros NaN.
  - **`_trajectory_step(batch, outputs)`**: Phase 1 latent trajectory step. Re-encodes future_snapshot_batch through same encoder (target-network trick), computes latent_future_state_loss. Returns n_supervised=0 if no follow-up.
  - **`_losses_for_batch(batch, stage) -> Dict`**: Calls validate_for_loss, canonical_forward, assemble_canonical_loss. Phase 1 override for stage E trajectory. Strict per-batch supervision enforcement.
  - **`fit_stage(stage, n_epochs) -> List[Dict]`**: Epoch loop with per-epoch metrics.
  - **`_train_one_epoch(stage, epoch) -> CanonicalEpochMetrics`**: AdamW optimizer, gradient clipping, strict epoch-boundary supervision check.
  - **`validate(stage) -> float`**: No-grad validation.

### `data_module.py`
- **`MORTFMSplits` (dataclass)** -- train/val/test lists of TemporalTrainingPair. Has `.summary()`.
- **`split_patients(pairs, val_fraction, test_fraction, seed) -> MORTFMSplits`**: Patient-disjoint split using torch.Generator manual seed. Validates no overlap across splits.
- **`make_loader(pairs, batch_size, ...) -> DataLoader`**: Wraps MORTFMDataset + mort_collate.
- **`make_data_module(pairs, ...) -> (MORTFMSplits, DataLoader, DataLoader, DataLoader)`**: End-to-end split + loader construction.

### `acceptance_gate.py`
- **`VALID_LEVELS`** -- ("debug", "technical", "research", "strong").
- **`AcceptanceReport` (dataclass)** -- level_requested, level_achieved, passed, n_samples, n_unique_patients, n_modalities, n_longitudinal_pairs, n_survival_rows, n_events_observed, has_patient_disjoint_split, has_drug_response_labels, has_survival_labels, has_temporal_pairs, modality_coverage, blocking_reasons. Has `.summary()`.
- **`evaluate_cohort(pairs, level, config) -> AcceptanceReport`**: Evaluates cohort against claim-level thresholds.
  - debug: >=10 patients, >=1 modality, labels present.
  - technical: >=100 samples, >=1 modality, labels present.
  - research: >=200 patients, >=50 events, >=2 modalities, trajectory needs >=25 patients with >=2 timepoints.
  - strong: research + explicit operational evidence required.
- **`_max_achieved_level(pairs, ...)`**: Best level cohort satisfies.
- **`require_level(pairs, level, config) -> AcceptanceReport`**: Raises AcceptanceError if gate fails.
- **`AcceptanceError(RuntimeError)`** -- Carries the AcceptanceReport.

---

## 2. `blocks/` -- Block Adapters (5 blocks + common + init)

### `_common.py`
Shared helpers for block adapters:
- **`read_json(path) -> Optional[Dict]`**: Safe JSON reader.
- **`utcnow_iso() -> str`**: UTC ISO timestamp.
- **`git_head_short() -> str`**: Short git SHA of HEAD, no-fail.
- **`load_torch_payload(path, map_location) -> Optional[dict]`**: Safe torch.load.

### `block_a_cellline.py` -- Block A: Cell-line foundation
- **`load_block_a(checkpoint_path) -> (model, config, payload)`**: Loads Block A checkpoint, constructs MORTFM, loads state_dict (strict=False).
- **`block_a_artifact_record(...) -> ArtifactRecord`**: Documents data sources (DepMap RNA 24Q2 1775 cell-lines, GDSC 241,578 rows, STRING v12 PPI 473,618 edges, Reactome 156,329 edges, ChEMBL 6,577 edges). Feature space: rna_block_a_top2000. Allowed claims: technical, static_drug_response.

### `block_b_beataml.py` -- Block B: BeatAML fine-tuned-from-A
- **`load_block_b(checkpoint_path) -> (model, config, payload)`**: Similar to Block A loader.
- **`block_b_artifact_record(...) -> ArtifactRecord`**: Data: BeatAML 1.0 expression (328 specimens), drug-response AUC (95,300 rows x 122 drugs). Feature space: rna_block_b_aligned_to_a. Allowed claims: technical.

### `block_c_scrna.py` -- Block C: Single-cell state encoder
- **`_infer_head_dims(state_dict) -> dict`**: Reads head output dims from saved state-dict (pathway_proteins, drug_candidates, resistance_states).
- **`load_block_c(checkpoint_path) -> (model, config, payload)`**: Constructs MORTFM with inferred head dims.
- **`block_c_artifact_record(...) -> ArtifactRecord`**: Data: GSE124310 (26,321 cells), GSE271107 (125,676 cells), 51 pseudobulked snapshots. Feature space: scrna_block_c_hvg_union_3535. Allowed claims: technical, single_cell_state. Blocked: trajectory, drug_response.

### `block_d_graph.py` -- Block D: Graph + ESM-2 cache
- **`load_block_d(cache_path)`**: Loads ESMEmbeddingCache.
- **`block_d_artifact_record(...) -> ArtifactRecord`**: Data: UniProt human proteome, ESM-2 8M embeddings, biological graph edges, identifier map. Artifact type: embedding_cache. Allowed claims: technical, (conditionally) sequence_aware, pathway_context.

### `block_e_integrated.py` -- Block E: Integrated checkpoint
- **`load_block_e(checkpoint_path, manifest_path) -> (state_dict, config, manifest)`**: Loads merged A+B+C+D checkpoint.
- **`block_e_artifact_record(...) -> ArtifactRecord`**: Inherits union of allowed_claims from constituent blocks. Does NOT introduce new claims.

---

## 3. `causal/` -- Causal Counterfactual Primitives (7 files)

### `__init__.py`
Exports: `InterventionGraph`, `InterventionType`, `CounterfactualRunner`, `EdgeEffect`, `PathwayCausalValidator`, `join_crispr_essentiality`, `join_drug_target_support`, `join_pathway_support`, `top_k_pathway_enrichment`, `build_edge_evidence_report`.

### `intervention_graph.py`
- **`InterventionType`** = Literal["knockout", "amplify", "identity"].
- **`_deterministic_edge_seed(eid) -> int`**: SHA-256 truncated to 64 bits. Deterministic, not affected by PYTHONHASHSEED.
- **`deterministic_edge_basis(eid, d_graph) -> Tensor`**: Per-edge cosine basis vector, L2-normalized.
- **`EdgeIntervention` (dataclass)** -- edge_id, intervention, factor.
- **`class InterventionGraph`**: Wraps precomputed graph embedding, supports do(e=...) edits.
  - **`__init__(edges_parquet, nodes_csv, d_graph, attach_id_bridge)`**: Loads biological_edges.parquet, protein_nodes.csv. Builds deterministic per-edge basis vectors (SHA-256 seeded cosine patterns, NOT random). Optionally attaches IDHarmonizer.
  - **`has_edge(edge_id)`, `list_edges(edge_type, limit)`**, **`edge_to_hgnc_pair(edge_id)`**.
  - **`base_embedding(batch_size) -> Tensor`**: Average-pooled per-row basis.
  - **`intervene(base_emb, interventions) -> Tensor`**: Applies knockout (zeros contribution) or amplify (scales contribution). Raises KeyError for unknown edges.

### `counterfactual_runner.py`
- **`EdgeEffect` (dataclass)** -- edge_id, p_resistance_baseline, p_resistance_under_knockout, p_resistance_under_amplify, delta, note.
- **`class CounterfactualRunner`**: Scores each edge's effect on resistance probability.
  - **`__init__(sde, basin, hitting, intervention_graph)`**.
  - **`_roll(z0, graph_emb, drug, clinical) -> float`**: Runs SDE + HittingTime, returns P(tau^R <= T_max).
  - **`run(z0, drug, clinical, edge_ids, amplify_factor) -> DataFrame`**: For each edge: compute baseline, knockout, amplify resistance probabilities. Returns DataFrame sorted by delta descending.

### `crispr_evidence_joiner.py`
- **`join_crispr_essentiality(edge_effects, crispr_summary_csv, target_col) -> DataFrame`**: Annotates edges with DepMap CRISPR essentiality (median_effect, frac_essential, is_common_essential). Returns NaN columns if file missing.

### `drug_target_evidence_joiner.py`
- **`join_drug_target_support(edge_effects, drug_targets_csv, target_col, source_col) -> DataFrame`**: Annotates edges with ChEMBL drug-target support (target_support, source_support, either_endpoint).

### `pathway_evidence_joiner.py`
- **`_build_hgnc_to_pathways(reactome_parquet) -> dict[str, Set[str]]`**: Builds HGNC -> pathway set map via IDHarmonizer.
- **`join_pathway_support(edge_effects, target_col, reactome_parquet) -> DataFrame`**: Adds target_pathway_count and target_top_pathway columns.
- **`top_k_pathway_enrichment(edge_effects, k, n_permutations, seed) -> dict`**: Empirical enrichment test: does top-k mean pathway count exceed random expectation? Returns p_value and enriched_at_p_lt_0_05.

### `edge_evidence_report.py`
- **`DEFAULT_K_OF_N = 2`**, **`N_EVIDENCE_CHANNELS = 3`**.
- **`_evaluate_k_of_n_gate(crispr_support, drug_target_support, reactome_support, k) -> dict`**: k-of-N gate verdict (v19 replaces hard-AND from v17.8). Returns n_supporting_channels, gate_passes, channels_supported, channels_not_supported.
- **`evaluate_edge_evidence_gate(...) -> dict`**: Public helper for single evidence row.
- **`build_edge_evidence_report(edge_effects, top_k, k) -> dict`**: Full v17.8 evidence package. Composes all three joiners. Returns {table, enrichment, summary} with causal_mechanism_gate_pass.

### `pathway_causal_validator.py`
- **`CausalValidationResult` (dataclass)** -- n_edges_scored, median_delta, top_k_drug_target_recall, top_k_reactome_enrichment, negative_control_distribution, causal_mechanism_gate_pass, reasons_blocked.
- **`class PathwayCausalValidator`**: Cross-checks counterfactual edge effects against external evidence (CRISPR DepMap, Open Targets drug-target, Reactome). Always reports negative-control distribution. Never grants causal_mechanism from agreement alone.
  - **`__init__(drug_targets_csv, reactome_parquet, crispr_dir)`**: Loads drug targets, Reactome, CRISPR essentiality.
  - **`validate(edge_effects, ks) -> CausalValidationResult`**: Computes drug-target recall, Reactome enrichment, CRISPR cross-check (via IDHarmonizer), negative-control distribution (middle 10% ranking), causal_mechanism gate pass/fail.

---

## 4. `drug_response/` -- Drug-Conditioned Response Head (3 files)

### `__init__.py`
Exports: `DrugConditionedSpecimenResponseHead`, `PerDrugPairSampler`.

### `drug_conditioned_head.py`
- **`class DrugConditionedSpecimenResponseHead(nn.Module)`**: Per-drug response head replacing v16 mean-aggregated DrugSpecificTrajectoryRiskHead.
  - `__init__(d_latent, n_drugs, n_drug_families, d_drug_embed=32, d_family_embed=8, d_hidden=128, dropout=0.2)`: Drug embedding + family embedding + 3-layer MLP -> scalar.
  - `forward(z_patient, drug_idx, family_idx) -> Tensor`: (B,) scalar response per (patient, drug) pair.

### `per_drug_sampler.py`
- **`class PerDrugPairSampler` (dataclass)**: Yields batches of indices into a flat (specimen, drug) pair list, stratified by drug_family so each batch sees every family at least once.
  - Fields: drug_families, batch_size, seed.
  - `__iter__` -> Iterator[List[int]], `__len__` -> int.

---

## 5. `graph/` -- ID Harmonization (2 files)

### `__init__.py`
Exports: `IDHarmonizer`.

### `id_harmonizer.py`
- **`_PREFERRED_SOURCES`** -- Priority mapping for STRING aliases: Ensembl_HGNC_symbol (prio 1), UniProt_AC (prio 1), Ensembl_HGNC (prio 2), UniProt_GN_Name (prio 3), Ensembl_UniProt (prio 2), BioMart_HUGO (prio 4).
- **`class IDHarmonizer` (dataclass)**: Bidirectional bridge between STRING ENSP, UniProt, HGNC symbol.
  - Fields: `string_to_hgnc`, `string_to_uniprot`, `hgnc_to_string`, `uniprot_to_string`, `alias_to_string` (ALL aliases -- PDB, Entrez, KEGG, etc.), `n_rows`.
  - **`build_or_load(aliases_gz, cache_parquet) -> IDHarmonizer`**: Builds from `9606.protein.aliases.v12.0.txt.gz` or loads cached parquet. Indexes preferred sources by priority. Caches two parquets: string_id_bridge.parquet and string_alias_to_id.parquet.
  - **`any_alias_to_hgnc(aliases) -> List[Optional[str]]`**: Maps any STRING-known alias to HGNC symbol via alias_to_string -> string_to_hgnc chain.
  - **`string_to_ensp`, `string_to_hgnc_symbol`, `string_to_uniprot_acc`, `hgnc_to_string_id`**: Bulk lookups.
  - **`coverage_against_hgnc_set(hgnc_symbols) -> dict`**: Reports coverage stats.

---

## 6. `longitudinal/` -- Longitudinal/Resistance Modelling (10 files)

### `__init__.py`
Exports all schemas, builders, dataset, cohort_registry, endpoint_mapper, censoring_policy, temporal_splitter, cohort_harmonizer.

### `schemas.py`
- **`EvidenceLevel`** = Literal["clinical", "molecular", "ex_vivo", "weak"].
- **`EndpointType`** = Literal["progression", "relapse", "refractory_status", "mrd_conversion", "ex_vivo_resistance_shift", "molecular_resistance_state", "death", "censoring"].
- **`TreatmentExposure` (dataclass)** -- drug_name, drug_class, start_day, end_day, dose, line, regimen_id.
- **`MolecularSample` (dataclass)** -- sample_id, aliquot_id, timepoint_label, timepoint_day, assay, parquet_path, notes.
- **`ClinicalEvent` (dataclass)** -- event_name, day, free_text.
- **`ResponseEvent` (dataclass)** -- response (CR/VGPR/PR/MR/SD/PD/unknown), day, drug_context.
- **`ResistanceEndpoint` (dataclass)** -- endpoint_name, endpoint_type, event_time_days, event_observed, drug_context, evidence_level, notes. evidence_level is mandatory.
- **`PatientTimeline` (dataclass)** -- patient_id, diagnosis_day, treatment_lines, molecular_samples, clinical_events, response_events, resistance_events, cohort.
- **`LongitudinalPair` (dataclass)** -- patient_id, x_t_sample_id, x_future_sample_id, delta_t_days (MUST be real calendar days, NOT pseudotime), treatment_between, endpoint, cohort. Has `validate() -> List[str]` checking delta_t_days > 0 and both sample_ids present.

### `resistance_endpoint_builder.py`
- **`build_endpoints_from_mmrf_outcomes(outcomes_tsv) -> dict[str, List[ResistanceEndpoint]]`**: Builds TT2L (time_to_next_treatment, progression type) and OS (death type, NOT a resistance endpoint) from mmrf_outcomes_treatment.tsv.

### `temporal_pair_builder.py`
- **`build_pairs_from_mmrf(paired_tsv, outcomes_tsv) -> List[LongitudinalPair]`**: Builds pairs from 29 MMRF patients with t0+t1 aliquots. Uses TT2L event_time_days as delta_t_days proxy. Pseudotime never inserted as delta_t_days.
- **`summarise_pairs(pairs) -> dict`**: Reports n_pairs, n_with_endpoint, n_with_real_calendar_time, n_with_observed_event, n_unique_patients, validation_issues.

### `longitudinal_dataset.py`
- **`class LongitudinalDataset` (dataclass)**: Wrapper over List[LongitudinalPair].
  - `__len__()`, `patient_ids()`, `to_parquet(path)` (serializes to parquet with all fields).

### `cohort_registry.py`
- **`AccessTier`** = Literal["public", "registered", "controlled_dbgap", "controlled_ega"].
- **`Modality`** = Literal["bulk_rna", "scrna", "bulk_proteomics", "wes", "wgs", "methylation"].
- **`LongitudinalCohortSpec` (dataclass)** -- name, disease, molecular_modality, baseline_available, relapse_available, event_time_available, treatment_available, public_or_controlled, expected_pairs, feature_id_type, on_disk_path, ingested_at, notes. Property `contributes_to -> List[str]` -- which claim levels this cohort advances.
- **`CANONICAL_COHORTS`** -- 7 cohorts:
  - MMRF_IA12_paired (MM, 29 pairs, controlled_dbgap, ingested)
  - PADIMAC_GSE116324 (MM, 0 pairs, public, ingested -- baseline-only)
  - BeatAML_1.0 (AML, 0 pairs, public, ingested)
  - GSE124310_GSE271107_scRNA (MM, 0 pairs, public, ingested -- pseudotime only)
  - MMRF_IA19_planned (MM, 80 pairs, planned -- not ingested)
  - GSE39754_Brioli_paired (MM, 25 pairs, planned)
  - Tirier2021_EGA_scRNA_paired (MM, 14 pairs, controlled_ega, planned)
- **`lookup_cohort(name)`, `write_cohort_registry(out_path)`, `total_pairs_in_state(state)`**.

### `endpoint_mapper.py`
- **`EndpointMapSpec` (dataclass)** -- cohort_name, time_column, event_column, endpoint_type, endpoint_name, drug_column, notes.
- **`COHORT_ENDPOINT_MAPS`** -- Maps "MMRF_IA12_paired::TT2L" -> progression endpoint, "MMRF_IA12_paired::OS" -> death endpoint, "GSE39754_Brioli_paired::relapse_pair" -> molecular_resistance_state.
- **`map_outcomes_to_endpoints(cohort_name, outcomes_df, ...) -> dict[str, List[ResistanceEndpoint]]`**: Converts raw cohort columns to canonical ResistanceEndpoint objects.

### `censoring_policy.py`
- **`apply_censoring_policy(pairs) -> dict`**: Single source of truth for right-censoring treatment. Rules: uncensored (event_observed=True, finite time > 0), administratively censored (event_observed=False, finite time), dropped (missing time). No best-effort backfill. Returns {n_input, n_kept, n_dropped_no_event_time, n_dropped_zero_or_negative, n_event_observed, n_administratively_censored, kept: List[LongitudinalPair]}.

### `clinical_features.py`
- **`CLINICAL_FEATURE_NAMES`** = ["iss_stage_ordinal", "age_at_dx_zscore", "gender_is_male", "bort_1L", "n_treatments_norm"].
- **`ClinicalEncoding` (dataclass)** -- features: ndarray (n,5), mask: ndarray (n,5), feature_names, patient_ids, age_mean_days, age_std_days.
- **`encode_for_lens(outcomes_tsv, patient_ids) -> ClinicalEncoding`**: Deterministic encoding from mmrf_outcomes_treatment.tsv. ISS -> ordinal 0/1/2. Age -> z-score. Gender -> binary. Bort_1L -> binary. n_treatments -> normalized [0,1]. Missing values stay NaN; zero-filled with mask.

### `cohort_harmonizer.py`
- **`_ensg_to_hgnc_lookup(map_tsv) -> Dict[str, str]`**: ENSG to HGNC symbol lookup from mmrf_ensembl_to_symbol.tsv.
- **`harmonize_to_block_a(cohort_expression, cohort_feature_id_type, block_a_reference_features) -> (aligned, mask, report)`**: Projects cohort expression onto Block-A feature space. Handles ENSG->HGNC conversion (sum-aggregation for duplicates). Uses `align_to_reference_features` from data module.

### `temporal_splitter.py`
- **`TemporalSplit` (dataclass)** -- train, val, test lists + mode. Has `.summary()`.
- **`split_patient_disjoint(pairs, val_fraction, test_fraction, seed) -> TemporalSplit`**: Random patient shuffle with fixed seed.
- **`split_temporal_holdout(pairs, diagnosis_day_lookup, ...) -> TemporalSplit`**: Earliest 70% train, next 15% val, latest 15% test (by diagnosis_day).

---

## 7. `registries/` -- Meta-Layer Registries (5 files)

### `__init__.py`
Exports: `ArtifactRecord`, `ArtifactRegistry`, `CANONICAL_CLAIM_LEVELS`, `CLAIM_LEVELS`, `ClaimLevel`, `lookup_claim_level`, `CANONICAL_ENDPOINTS`, `EndpointSpec`, `lookup_endpoint`, `CANONICAL_FEATURE_SPACES`, `FeatureSpace`, `lookup_feature_space`.

### `artifact_registry.py`
- **`ArtifactRecord` (dataclass)** -- block_id, artifact_type (checkpoint|embedding_cache|graph|manifest|report), path, data_source, n_samples, n_features, feature_space_id, endpoint_name, allowed_claims, blocked_claims, checkpoint_compatible, created_at, git_commit, gate_report_path, notes. Property `status` -> "present" or "missing".
- **`class ArtifactRegistry`**: Container for ArtifactRecord objects.
  - `add(record)`, `get(block_id)`, `__contains__`, `__len__`, `items()`, `to_dict()`, `save(out_path)`, `load(path)` (classmethod), `all_present(required_block_ids)`, `claims_unlocked() -> List[str]` (union of allowed_claims across present artifacts).

### `claim_gate_registry.py`
- **`CLAIM_LEVELS`** -- Ordered list of 12 claim levels from weakest to strongest: technical, static_drug_response, hematologic_specimen_drug_response, single_cell_state, sequence_aware, pathway_context, drug_target_mechanism, survival_prediction, patient_level_clinical_prediction, longitudinal_trajectory, resistance_emergence, causal_mechanism.
- **`ClaimLevel` (dataclass)** -- name, required_artifacts, required_gates, required_endpoint_types, forbidden_endpoint_types, blocks_until, description.
- **`CANONICAL_CLAIM_LEVELS`** -- 12 entries, each specifying required artifacts, gates, allowed/forbidden endpoint types. Notable constraints:
  - OS forbidden for resistance_emergence, patient_level_clinical_prediction.
  - Pseudotime forbidden for longitudinal_trajectory, resistance_emergence.
  - causal_mechanism requires E_integrated + D_esm2_embeddings + gate_causal_pathway_mechanism.
- **`lookup_claim_level(name)`, `write_claim_gate_registry(out_path)`**.

### `endpoint_registry.py`
- **`EndpointType`** = Literal of 9 types including drug_response, overall_survival, progression_free_survival, disease_stage_ordinal, etc.
- **`EndpointSpec` (dataclass)** -- endpoint_name, endpoint_type, event_time_column, event_observed_column, allowed_claims, forbidden_claims, notes.
- **`CANONICAL_ENDPOINTS`** -- 9 entries:
  - ln_ic50: drug_response, allowed for technical + static_drug_response.
  - auc: drug_response, allowed for technical + static + hematologic.
  - overall_survival: allowed for technical + survival_prediction ONLY. Forbidden for resistance_emergence and patient_level.
  - progression_free_survival: allowed for resistance_emergence.
  - relapse_time, refractory_status, mrd_conversion: allowed for resistance_emergence.
  - ex_vivo_resistance_shift: allowed for resistance_emergence.
  - longitudinal_molecular_state: allowed for longitudinal_trajectory + resistance_emergence. Requires real calendar-time labels.
  - disease_stage_pseudotime: allowed ONLY for technical + single_cell_state. Forbidden for trajectory, resistance, survival, patient-level.
- **`lookup_endpoint(name) -> Optional[EndpointSpec]`**, **`write_endpoint_registry(out_path)`**.

### `feature_space_registry.py`
- **`FeatureSpace` (dataclass)** -- feature_space_id, modality, n_features, source, feature_names_path, notes, derived_from, block_owner. Property `status` -> "present"/"missing"/"implicit".
- **`CANONICAL_FEATURE_SPACES`** -- 8 entries:
  - rna_block_a_top2000 (2000 genes, Block A)
  - rna_block_b_aligned_to_a (2000, same columns as A, Block B)
  - scrna_block_c_hvg_union_3535 (3535 HVGs, Block C -- DIFFERENT axis from A)
  - protein_uniprot_human (20,659 proteins, Block D)
  - protein_esm2_8m_cached (1,781 cached sequences, 320-dim embeddings, Block D)
  - drug_chembl_v34_filtered (5,092 drugs, Block D)
  - pathway_reactome (2,845 pathways, Block A)
  - graph_node_string_v12 (20,919 nodes, Block A)
- **`lookup_feature_space(fs_id)`, `write_feature_space_registry(out_path, extras)`**.

---

## 8. `single_cell/` -- Single-Cell Objectives for Block C (4 files)

### `__init__.py`
Exports: `supcon_loss`, `pseudotime_ordinal_loss`, `StratifiedStageSampler`.

### `state_contrastive_loss.py`
- **`supcon_loss(z, labels, temperature, eps) -> Tensor`**: Supervised contrastive loss (Khosla et al. 2020). L2-normalizes z, computes (B,B) similarity matrix, masks diagonal, applies log-softmax + positive-pair weighting.

### `pseudotime_regularizer.py`
- **`pseudotime_ordinal_loss(proj_scores, stage_labels, margin) -> Tensor`**: Ranking loss enforcing stage_i > stage_j implies proj(z_i) > proj(z_j) + margin. NOT calendar time -- ordinal only. Returns mean hinge violation across all valid pairs.

### `disease_stage_sampler.py`
- **`class StratifiedStageSampler` (dataclass)**: Iterates indices in stage-balanced mini-batches. Shuffles within each stage per epoch (deterministic for seed), draws round-robin across stages. Fields: stage_labels, batch_size, seed. Has `__iter__` and `__len__`.

---

## 9. `survival/` -- Survival / Time-to-Event (6 files)

### `__init__.py`
Exports: `CompetingRiskHead`, `discrete_time_grid`, `survival_curve_from_hazards`, `SurvivalCalibration`, `concordance_index`, `concordance_index_bootstrap_ci`, `integrated_brier_score`, `km_strata_separation`, `ClinicalOnlyCox`, `cox_fit_predict_loo`, `permutation_null_cindex`, `RegularizedLENS`, `RegularizedLENSConfig`, `bootstrap_optimism_correction`, `fit_predict_loo_regularized`.

### `competing_risk_head.py`
- **`discrete_time_grid(t_max_days, n_bins) -> Tensor`**: Equal-width bins from 0 to t_max_days.
- **`class CompetingRiskHead(nn.Module)`**: Discrete-time hazard head with K competing events.
  - `__init__(d_latent, n_bins, event_names, time_grid_config)`: v19 Phase 7: when CanonicalTimeGridConfig supplied, n_bins forced to match canonical grid. One nn.Linear per event type.
  - `forward(z) -> dict`: Returns {hazard: (B,K,R), survival_curve: (B,K), cif_per_event: (B,K,R)}.
- **`survival_curve_from_hazards(hazard) -> Tensor`**: S_k = cumprod of (1 - h_{r,j}) over events and time.
- **`nll_competing_risk(hazard, survival, event_bin, event_observed, event_type, eps) -> Tensor`**: Discrete-time competing-risk NLL. Uncensored: -log h + log S_{k-1}. Censored: -log S_k.

### `baselines.py`
- **`class ClinicalOnlyCox(nn.Module)`**: Cox PH with L2 regularization, fit by Adam.
  - `__init__(n_features, l2)`, `linear_predictor(X)`, `_neg_partial_log_likelihood(X, time, event)` (Breslow ties), `fit(X, time, event, epochs, lr)`, `predict_risk(X)`.
- **`cox_fit_predict_loo(X, event_time, event_observed, ...) -> List[float]`**: Leave-one-out Cox: fit on n-1, predict held-out risk.
- **`permutation_null_cindex(risk_scores, event_time, event_observed, ...) -> dict`**: Empirical null distribution by permuting labels. Returns observed_cindex, null_mean, null_std, p_value.

### `regularized_lens.py`
- **`RegularizedLENSConfig` (dataclass)** -- d_latent, n_bins, d_hidden, dropout=0.3, l1=1e-4, l2=1e-3, feature_bag_frac.
- **`class RegularizedLENS(nn.Module)`**: Small encoder + CompetingRiskHead with L1+L2, dropout, feature bagging. For n<50 regime.
  - `forward(x) -> dict` (hazard + survival_curve + z).
  - `l1_penalty() -> Tensor`.
- **`fit_predict_loo_regularized(X, event_time, event_observed, ...) -> List[float]`**: LOO with feature bagging, NLL + L1 penalty training.
- **`bootstrap_optimism_correction(observed_cindex, X, ...) -> dict`**: Harrell 1996 optimism correction. Returns observed, mean_optimism, corrected_cindex, n_bootstrap.

### `survival_calibration.py`
- **`class SurvivalCalibration(nn.Module)`**: Single-parameter temperature scaling on competing-risk hazards (Guo et al. 2017 generalization).
  - `fit(hazard, event_bin, event_observed, event_type, max_iter, lr)`: LBFGS optimizer. Adjusts logit(h) by -log_temperature.
  - `forward(hazard) -> Tensor`: Applies learned temperature to hazards.

### `time_to_event_metrics.py`
- **`concordance_index(event_times, event_observed, risk_scores) -> float`**: Harrell's C-index. Pure numpy, no lifelines. O(n^2) pairwise comparison.
- **`concordance_index_bootstrap_ci(event_times, event_observed, risk_scores, n_boot, seed) -> (point, lo, hi)`**: 95% bootstrap CI over real data.
- **`integrated_brier_score(survival_curve_pred, t_grid, event_times, event_observed) -> float`**: IBS with IPCW (KM censoring distribution).
- **`km_strata_separation(event_times, event_observed, risk_scores, quantiles) -> dict`**: Stratify by risk-score tertiles. Reports per-stratum KM (n, n_events, median time, S at 180d/365d) + log-rank chi2 (Mantel-Haenszel).

---

## 10. `trajectory/` -- LENS Trajectory Primitives (6 files)

### `__init__.py`
Exports: `GraphConditionedDrift`, `GraphEnergyResistanceSDE`, `WaddingtonPotential`, `LatentToGraphProjector`, `CanonicalTimeGridConfig`, `canonical_time_grid`, `time_grid_meta`, `HittingTime`, `ResistanceBasin`.

### `grid.py`
- **`canonical_time_grid(t_max_months, n_steps, device, dtype) -> Tensor`**: Creates the single canonical time grid in months. Validates t_max > 0, n_steps >= 2.
- **`time_grid_meta(grid) -> dict`**: Returns t_max_months, n_steps, dt_months from a 1D grid.
- **`CanonicalTimeGridConfig` (dataclass)** -- t_max_months=12.0, n_steps=25. Methods: `to_dict()`, `build(device, dtype) -> Tensor`. Passed to both SDE and CompetingRiskHead constructors to ensure shared time axis.

### `graph_energy_sde.py`
- **`_mlp(d_in, d_hidden, d_out, n_hidden, act) -> nn.Sequential`**: Helper MLP builder.
- **`class WaddingtonPotential(nn.Module)`**: Scalar U_theta(z, d) with K learned basin centroids.
  - `__init__(d_latent, d_drug, n_basins, d_hidden)`: Basin centroids (nn.Parameter, randn*0.3), basin_width, drug_to_basin_weight (Linear), residual MLP.
  - `forward(z, drug) -> Tensor (B,)`: Quadratic-around-centroids energy + drug-modulated weights + MLP residual.
- **`class GraphConditionedDrift(nn.Module)`**: f_theta(z, g, d, c). MLP: (d_latent + d_graph + d_drug + d_clinical) -> d_latent.
- **`class _Diffusion(nn.Module)`**: Diagonal, positive, state+drug dependent sigma. MLP with sigmoid output clamped to [min_sigma, max_sigma].
- **`class GraphEnergyResistanceSDE(nn.Module)`**: Full latent dynamics module.
  - `__init__(d_latent, d_graph, d_drug, d_clinical, n_basins, integration_time, n_time_grid, n_mc_samples, time_grid_config)`: Composes WaddingtonPotential + GraphConditionedDrift + _Diffusion. v19 Phase 7: uses CanonicalTimeGridConfig when provided; warns on legacy linspace fallback.
  - `_grad_u(z, drug)`: Computes grad_z U using torch.autograd.grad with enable_grad.
  - `_step(z, graph_emb, drug, clinical, dt, deterministic)`: Euler-Maruyama step: z_{t+dt} = z_t + (-grad_U + f)*dt + sigma*sqrt(dt)*noise.
  - `forward(z0, graph_emb, drug, clinical, return_samples) -> dict`: Rolls out over time grid. Returns z_traj (B,T,d_latent), z_samples (S,B,T,d_latent), t_grid (T,), drift_norm (B,T-1).

### `graph_projector.py`
- **`class LatentToGraphProjector(nn.Module)`**: Projects patient latent z0 into graph-embedding space.
  - `__init__(d_latent, d_graph, d_hidden, dropout)`: Linear -> SiLU -> Dropout -> Linear -> LayerNorm.
  - `forward(z0) -> Tensor`: Returns (B, d_graph).

### `resistance_basin.py`
- **`class ResistanceBasin(nn.Module)`**: Soft assignment to K resistance basins.
  - `__init__(d_latent, n_basins, basin_names, share_centroids_with)`: Basin centroids + widths. Can share parameters with WaddingtonPotential. Default names: sensitive, persister, MRD_like, relapse_like, drug_specific_resistant.
  - `forward(z) -> Tensor (B, K)`: Softmax over negative squared distances.
  - `trajectory_probs(z_traj) -> Tensor (B, T, K)`: Basin probabilities along time axis.

### `hitting_time.py`
- **`class HittingTime(nn.Module)`**: Estimates tau^R = inf{t : z_t in B_R}.
  - `__init__(basin_module, resistant_basin_index=4, prob_threshold=0.5)`.
  - `forward(z_samples, t_grid) -> dict`: From MC samples (S,B,T,d_latent), computes basin probs, first-hit time per sample, returns:
    - cdf: (B, T) empirical P(tau^R <= t_k)
    - mean_tau: (B,) conditional mean hitting time
    - frac_hit: (B,) fraction of MC samples reaching B_R

---

## 11. Connections to `models/` Directory

- `model.py` imports `MultiOmicFoundationFusion` from `resistancemap.models.foundation_fusion` -- the main encoder+fusion module.
- `model.py` imports legacy heads from `resistancemap.legacy.v15_planned_mortfm.models.heads.*` -- ResistanceStateHead, TimeToResistanceHead, TrajectoryDistributionHead, PathwayRouteHead, DrugSpecificTrajectoryRiskHead, EvidentialUncertaintyHead, CounterfactualInterventionHead.
- `model.py` imports `WaddingtonPotential` from `resistancemap.landscape.potential` for the legacy path.
- `model.py` imports legacy `TrajectorySampler` from `resistancemap.legacy.v15_planned_mortfm.models.dynamics.trajectory_sampler`.
- The canonical v18 forward() uses the LENS modules from `mortfm/trajectory/` and `mortfm/survival/` directly (not legacy heads).
- `trainer.py` imports loss functions from `resistancemap.training.mortfm_losses`.
- `canonical_trainer.py` imports from `resistancemap.training.canonical_mortfm_losses`.
- `data_module.py` imports `MORTFMDataset` and `mort_collate` from `resistancemap.data.mortfm_dataset`.
- `block_d_graph.py` imports `ESMEmbeddingCache` from `resistancemap.data.protein_sequence_cache`.
- `cohort_harmonizer.py` imports `align_to_reference_features` from `resistancemap.data.feature_alignment`.

---

## 12. Key Design Invariants

1. **No data fabrication**: Loaders raise FileNotFoundError; outcomes default to None; MORTBatch.validate_for_loss raises ValueError on missing supervision.
2. **Explicit censoring**: ResistanceOutcome.censored is a mandatory flag; survival losses require it.
3. **Patient-disjoint splits**: split_patients / split_patient_disjoint enforce no overlap with runtime assertions.
4. **Acceptance gate**: 4-tier (debug/technical/research/strong) claim levels with explicit blocking_reasons.
5. **Strict supervision policy** (v18.2): MissingSupervisionError raised when supervised stages have insufficient coverage (E: 80%, F: 90%, G: 70%).
6. **Canonical time grid** (v19 Phase 7): Single CanonicalTimeGridConfig shared by SDE and CompetingRiskHead; d_graph pinned by Bug B19 assertion.
7. **OS is NOT a resistance endpoint**: Endpoint registry explicitly forbids OS for resistance_emergence claims.
8. **Pseudotime is NOT calendar time**: disease_stage_pseudotime forbidden for trajectory/resistance/survival claims.
9. **k-of-N causal gate** (v19): 2-of-3 evidence channels (CRISPR, drug-target, Reactome) required for causal_mechanism claim, replacing hard-AND.
10. **Target-network trick**: canonical_trainer encodes future_snapshot_batch through same encoder under torch.no_grad to prevent encoder collapse.

---

## Appendix D: Evaluation & Governance Layer

# Evaluation & Governance Layer

## Architecture Overview

The evaluation layer is a **multi-agent, tiered DAG** that audits the ResistanceMap pipeline against three testable scientific claims (state, when, through-which-pathway). It is orthogonal to the training DAG -- it consumes configs and outputs and produces a graded, auditable assessment without modifying the training pipeline.

**Tier hierarchy:**
- **Tier A** -- Foundational fitness gates (data adequacy, measurement integration, bias/fairness/shift). Hard-stop tier: a Tier A FAIL marks every downstream agent BLOCKED.
- **Tier B** -- Scientific coherence (architecture audit, cell-state biology, pathway/PPI grounding).
- **Tier C** -- Forecasting, calibration, baseline adversary, clinical safety.
- **Tier D** -- Governance chair that consumes all findings and emits the final EvalReport.

**Key invariant:** No agent fabricates data. If inputs are missing, agents return SKIPPED with explicit documentation of what is absent.

---

## 1. Core Framework Files

### `resistancemap/evaluation/__init__.py`
Public API surface for the evaluation layer. Exports:
- `EvalAgent`, `EvalFinding`, `EvalVerdict`, `Verdict`
- `EvalOrchestrator`, `EvalReport`
- `Charter`, `Claim`, `load_default_charter`
- `Rubric`

### `resistancemap/evaluation/base.py`
Base class and contracts for all evaluation agents.

**Enum `EvalVerdict`**: PASS, FAIL, CONDITIONAL, SKIPPED, BLOCKED

**Dataclass `EvalFinding`**: The cross-agent contract. Every evaluation agent returns one.
- Fields: `agent_name`, `tier` (A/B/C/D), `verdict`, `score` (0-1), `criteria_results`, `evidence`, `failure_modes`, `required_changes`
- `__post_init__()`: Coerces verdict from string, clamps score to [0,1], coerces None score to 0.0
- `to_dict()`: JSON-safe serialization

**Class `EvalAgent(BaseAgent)`**: Base class for evaluation agents.
- Class attribute `tier` (must be A/B/C/D)
- `assess(intake, config) -> EvalFinding`: Abstract method subclasses override
- `execute(inputs, config) -> AgentResult`: Adapter wrapping assess() for BaseAgent compatibility; catches all exceptions and wraps them as FAIL findings
- `skip_if_data_missing(paths, memo)`: Returns SKIPPED finding if any path missing (canonical "no fabricated data" escape hatch)
- `record_evidence(key, value)`: Buffer evidence for the next finding
- `fail_closed_gate(reason, required_changes)`: Build a FAIL finding for hard-gate violation

**Helper `_stringify(obj)`**: Recursively coerces Path and non-JSON values to strings.

### `resistancemap/evaluation/charter.py`
Scientific charter encoding three testable claims.

**Dataclass `Claim`**: A single testable scientific claim.
- Fields: `name`, `operational_meaning`, `falsifiers` (list), `epistemic_limits` (list), `required_evidence` (list)

**Dataclass `Charter`**: Top-level evaluation charter.
- Fields: `claims` (list[Claim]), `non_goals` (list[str]), `stop_rules` (list[str]), `notes` (list[str])
- `get_claim(name)`: Look up a claim by name

**Function `load_default_charter()`**: Builds the default MM resistance charter with three claims:
1. **state** -- Discrete or continuous resistance phenotype label per drug. Falsifiers: Brier/ECE no better than prevalence baseline; AUROC collapses on patient-level split; calibration drift across cohorts; label definition leaks outcome.
2. **when** -- Time-to-event or horizon-bucket prediction at 3/6/12 months. Falsifiers: C-index no better than KM/LOCF baseline; pinball/CRPS no better than constant-hazard; no timestamp field (temporal leakage); right-censoring mishandled.
3. **pathway** -- PPI graph attributions identifying mechanistic drivers. Falsifiers: attributions unstable under perturbation (rank corr < 0.5); holdout pathway recovery no better than random; top-k dominated by hubs (graph-prior leakage).

Non-goals: not a diagnostic device; no causal inference; no generalization outside MM.
Stop rules: Tier A FAIL -> hard-stop; missing audit trail -> hard-stop; unlisted claims must be added or removed.

### `resistancemap/evaluation/rubric.py`
Weighted scoring rubric for evaluation governance.

**Class `Rubric`**: 3-level table `tier -> agent -> criterion -> {weight, evidence_type, required, ...}`
- `from_dict(spec)` / `from_yaml(path)` / `to_yaml(path)` / `load_default()`: I/O methods
- `_validate()`: Checks weight non-negativity, sum-to-one per agent, evidence_type in valid set
- `score(agent, criterion, value, evidence)`: Pointwise mode records one criterion. Also supports bulk mode with dict-of-dicts returning weighted means.
- `_coerce(value)`: Coerces arbitrary values to [0,1] scores (bool, numeric, "pass"/"fail" strings)
- `agent_total(agent)`: Sum weighted scores for one agent
- `tabulate()`: Returns DataFrame (or dict-of-dicts if pandas unavailable) of all recorded scores

Valid evidence types: `file, metric, schema, document, hash, todo, report, table, plot, manifest`

### `resistancemap/evaluation/orchestrator.py`
Tiered orchestrator that runs agents in strict tier order.

**Dataclass `EvalReport`**: Aggregated evaluation report.
- Fields: `run_id`, `per_tier_verdicts`, `per_tier_agent_verdicts`, `findings` (list[EvalFinding]), `overall_verdict`, `maturity_grade` (TRL 1-9), `required_changes`, `verification_chain` (SHA-256 hashes), `timing`, `charter_notes`, `metadata`
- `to_dict()`: JSON-safe serialization

**Class `EvalOrchestrator`**:
- `add_agent(agent)`: Register an EvalAgent under its tier
- `run(config, run_id) -> EvalReport`: Execute all agents in tier order (A->B->C->D)
  - Tiers run in parallel within tier via `asyncio.gather`
  - **Hard-stop logic**: Any Tier A FAIL blocks all downstream tiers (agents marked BLOCKED)
  - **Adversarial round**: Between Tier B and C, compares architecture vs baseline-adversary findings; architecture must beat baseline by >= 0.05 to "win"
  - Writes JSON audit trail: `logs/evaluation/<run_id>/report.json` and `findings.jsonl`
- `_run_tier()`: Runs all agents in a tier in parallel, wraps in timing/hashing
- `_record()`: Appends finding to report, records timing, updates per-tier verdicts, computes SHA-256 verification hash
- `_tier_rollup()`: Reduces per-agent verdicts to single worst-case verdict (FAIL > BLOCKED > SKIPPED > CONDITIONAL > PASS)
- `_finalize()`: Computes overall verdict roll-up, aggregates de-duplicated required changes, assigns TRL maturity grade
- `_maturity_grade()`: Conservative TRL 1-9 grading. Maximum reachable from evaluation alone is TRL 4. Mapping: all PASS -> TRL 4; PASS overall but not unanimous -> TRL 3; CONDITIONAL -> TRL 2; FAIL/BLOCKED -> TRL 1.
- `adversarial_round()`: Placeholder debate hook; architecture must beat baseline by >= 0.05 margin

---

## 2. Evaluation Metrics (`resistancemap/evaluation/metrics/`)

### `metrics/__init__.py`
Re-exports all metric functions from submodules: calibration, survival, forecasting, pathway_stability, shift.

### `metrics/calibration.py`
Pure-numpy calibration metrics.

- `expected_calibration_error(probs, labels, n_bins=15) -> float`: ECE with equal-width confidence bins. Supports binary 1D and multiclass (N,K) probs.
- `maximum_calibration_error(probs, labels, n_bins=15) -> float`: Worst-case |accuracy - confidence| across bins.
- `reliability_curve(probs, labels, n_bins=15) -> (centers, accs, confs)`: Per-bin data for reliability diagrams. Empty bins get NaN.
- `brier_score_multiclass(probs, labels) -> float`: Mean squared error vs one-hot labels. Handles both binary 1D and multiclass (N,K).
- `_validate_probs_labels()`: Validates probs in [0,1], rows sum to 1 for multiclass, integer labels.
- `_confidences_and_correct()`: Reduces probs/labels to (confidence, correct) per sample.

### `metrics/survival.py`
Survival-analysis metrics with IPCW (inverse probability of censoring weighting). scipy optional.

- `concordance_index(times, events, risk_scores) -> float`: Harrell's C-index with tie handling (0.5 credit). Vectorized inner loop.
- `integrated_brier_score(times, events, surv_pred, eval_times) -> float`: IPCW-weighted IBS (Graf et al. 1999). Uses KM censoring estimator G(t). Trapezoidal time integration.
- `time_dependent_auc(times, events, risk_scores, eval_times) -> ndarray`: Cumulative/dynamic td-AUC (Uno et al. 2007). Returns per-eval-time AUC array. IPCW weights for cases.
- `_kaplan_meier_censoring()`: KM estimator of censoring survival G(t).
- `_step_eval()`: Right-continuous step function evaluator.
- `_validate_survival_inputs()`: Checks 1D, non-negative times, binary events.

### `metrics/forecasting.py`
Probabilistic forecasting metrics.

- `pinball_loss(y_true, y_pred_quantile, tau) -> float`: Quantile loss for a single level tau in (0,1). At tau=0.5 equals 0.5*MAE.
- `crps_ensemble(y_true, ensemble_preds) -> float`: CRPS from ensemble using energy-form estimator. `ensemble_preds` shape (N, M). Uses sorted-ensemble closed-form for E|X-X'| without forming (M,M) matrix.
- `mean_absolute_calibration_error(pred_quantiles, y_true, quantile_levels) -> float`: Mean absolute deviation of empirical coverage from nominal quantile levels.

### `metrics/pathway_stability.py`
Pathway attribution stability metrics for graph-grounded reasoning audits.

- `attribution_overlap(attr_a, attr_b, top_k=None)`: Two modes:
  - **Set mode**: Both inputs are feature-id lists -> returns Jaccard index (float)
  - **Score mode**: At least one is numeric -> returns dict with `jaccard`, `kendall_tau`, `top_k_overlap`, `n_shared`
- `edge_dropout_stability(attribute_fn, ppi_edges, dropout_rates, n_trials, seed)`: Stability under random PPI edge dropout. For each rate, drops edges, re-runs attribute_fn, measures Jaccard/top-k overlap. Returns bootstrap CIs.
- `pathway_holdout_generalization(model_fn, heldout_pathway_ids, predictions) -> float`: Pearson correlation between model_fn(held-out) and reference predictions.
- `_kendall_tau(x, y)`: Kendall-tau-b without scipy, O(N^2).
- `_bootstrap_ci()`: Percentile bootstrap CI for mean.

### `metrics/per_drug.py`
Per-drug metrics with bootstrap CIs.

**Dataclass `DrugMetric`**: drug, n, n_pos, auroc, auroc_lo/hi, auprc, auprc_lo/hi, note.

- `per_drug_metrics(y_true, y_score, drug, n_boot=2000, seed=0, min_each_class=3) -> list[DrugMetric]`: Per-drug AUROC/AUPRC with 95% percentile bootstrap CIs. Flags drugs with insufficient class balance.
- `pooled_with_ci(y_true, y_score, n_boot, seed) -> dict`: Pooled AUROC with bootstrap CI.
- `require_per_drug_with_pooled(y_true, y_score, drug)`: Returns both pooled and per-drug together. **Reporting rule enforced**: no pooled AUROC may be quoted without the per-drug table.

### `metrics/shift.py`
Distribution-shift detection metrics. scipy optional.

- `maximum_mean_discrepancy(x_src, x_tgt, kernel='rbf', bandwidth=None) -> float`: Biased V-statistic MMD^2 with RBF or linear kernel. Median heuristic bandwidth.
- `kolmogorov_smirnov_shift(x_src, x_tgt) -> dict`: Per-feature two-sample KS test. Returns `stat` and `p_value` arrays. Falls back to numpy if scipy unavailable.
- `population_stability_index(expected, actual, n_bins=10) -> float`: PSI with quantile-based bins and smoothing. Thresholds: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
- `shift_alarm(scores, threshold=0.25) -> bool`: True if max(scores) > threshold.
- `_ks_pvalue_fallback()`: Numpy-only KS p-value via Smirnov series.
- `_median_heuristic_bandwidth()`: Median pairwise distance bandwidth.

---

## 3. Tier A Agents (Foundational Fitness Gates)

### `tier_a/__init__.py`
Exports: `DataAdequacyAgent`, `MeasurementIntegrationAgent`, `BiasFairnessShiftAgent`

### `tier_a/data_adequacy.py`
**Class `DataAdequacyAgent(EvalAgent)`**: Foundational data-fitness audit for MM resistance.

Checks:
1. **Indication match**: Filename keywords for MM vs off-target (AML, lymphoma, etc.). Off-target -> fail_closed_gate.
2. **Treatment line**: NDMM vs RRMM documented in patient manifest.
3. **Sample site**: Bone marrow vs peripheral blood.
4. **Modality alignment**: Single-cell vs bulk patient overlap.
5. **Resistance label definition**: Primary vs secondary documented.
6. **Patient-split structural leakage**: Via `audit_patient_split()`.
7. **Temporal leakage**: Via `detect_temporal_leakage()`.

MM keywords: mm, myeloma, mmrf, commpass, gse124310, gse271107.
Off-target keywords: aml, lymphoma, cll, breast, lung, melanoma.

Never PASS without numeric loader; always returns CONDITIONAL at best with explicit required_changes to wire real data.

### `tier_a/measurement_integration.py`
**Class `MeasurementIntegrationAgent(EvalAgent)`**: Audits cross-modality alignment and PPI coverage of drug targets.

Checks:
1. **Cell-type vs state resolution**: TODO until single-cell loader wired
2. **Batch effects**: TODO (kBET/iLISI not yet computed)
3. **Cross-modality alignment**: TODO (same-cell vs imputed-bridge)
4. **PPI graph version + drug-target coverage**: Scans STRING file for protein count/edge count; checks if PPI contains surface targets for configured drugs. Requires >= 80% coverage.

Drug surface targets map: Daratumumab->CD38, Bortezomib->PSMB5/PSMB1, Lenalidomide->CRBN/IKZF1/IKZF3, Talquetamab->GPRC5D, etc. (12 drugs mapped).

`_scan_ppi(path)`: Streams PPI file without loading full graph into memory.
`_drug_target_coverage()`: Streaming substring scan for targets in PPI file.

### `tier_a/bias_fairness.py`
**Class `BiasFairnessShiftAgent(EvalAgent)`**: Audits selection bias, ancestry, SES proxies, train/deploy shift.

Checks:
1. **Selection bias audit**: Ancestry distribution memo present
2. **Ancestry representation**: Patient manifest with demographic columns
3. **SES proxy handling**: Documented or explicitly absent
4. **Train/deploy shift**: Defers to `metrics/shift.py` if importable
5. **Subgroup calibration**: Declares requirement for Tier C

Returns SKIPPED if both ancestry memo and patient manifest absent; CONDITIONAL otherwise.

---

## 4. Tier B Agents (Scientific Coherence)

### `tier_b/__init__.py`
Exports: `ArchitectureAuditorAgent`, `CellStateTrajectoryAgent`, `PathwayPPIReasoningAgent`

### `tier_b/architecture_auditor.py`
**Class `ArchitectureAuditorAgent(EvalAgent)`**: Static audit of architecture for scientific coherence.

Audits four modules:
1. **VAE latent**: Scientific function = state summary. Identifiability gap: latents only identified up to rotation/sign. Ablation plan: drop_h3k27me3_decoder, shuffle_protein_to_epigenome_pairs, freeze_latent_to_PCA.
2. **ODE/trajectory**: Scientific function = latent dynamics. Key issue: `integration_time` is arbitrary units, not calendar time. Ablation plan: replace_ode_with_linear_drift, permute_initial_conditions, halve_integration_time.
3. **GNN (PPI)**: Scientific function = pathway context. Key issue: attention attributions confounded with node degree. Ablation plan: swap_ppi_for_degree_preserving_rewire, drop_esm2_embeddings_use_one_hot, edge_confidence_threshold_sweep.
4. **Fusion**: Scientific function = combine modalities. Key issue: cross-attention can collapse onto dominant modality. Ablation plan: leave_one_modality_out, shuffle_modality_alignment, swap_cross_attention_for_concat.

Each module records: identified (what data+loss recover), assumed (baked into architecture), identifiability gap, ablation_plan (what should break).

Score: fraction of clean modules. All clean -> PASS; >= 0.5 -> CONDITIONAL; < 0.5 -> FAIL.

### `tier_b/cell_state_specialist.py`
**Class `CellStateTrajectoryAgent(EvalAgent)`**: Evaluates biological coherence of recovered cell states.

Required intake: `vae_latents`, `cell_metadata`, `state_assignments`, `program_scores`.

Sub-checks:
1. **Lineage consistency**: Each lineage should concentrate in <= 50% of states. Reports purity per lineage.
2. **Batch robustness**: Per-batch re-fits compared via pair agreement (threshold >= 0.6).
3. **Program enrichment**: Each state must be enriched for at least one known program. Expected programs: stress_response, emt_like, metabolic_reprogramming, interferon_response, dna_damage_repair, plasma_cell_identity, proliferation, unfolded_protein_response.
4. **Transition plausibility**: Checks row-stochasticity, non-negativity, irreducibility of transition matrix.

### `tier_b/pathway_ppi.py`
**Class `PathwayPPIReasoningAgent(EvalAgent)`**: Audits pathway attributions for graph-grounded structure.

Required intake: `ppi_edges`, `attribute_fn`, `baseline_attribution`, `alternative_attribution`, `heldout_pathway_ids`, `heldout_predictions`, `heldout_model_fn`.

Sub-checks:
1. **Graph-walk provenance**: Pathways must originate from graph walks (walks_traced/walks_total >= 0.8)
2. **Cross-PPI-build consistency**: Jaccard >= 0.4 across alternative PPI builds
3. **Edge-dropout stability**: Jaccard CI lower bound >= 0.5 at 10% dropout
4. **Held-out pathway generalization**: Pearson >= 0.3 on held-out pathways

---

## 5. Tier C Agents (Domain-Specific Evaluation)

### `tier_c/__init__.py`
Exports: `ForecastingUncertaintyAgent`, `BaselineAdversaryAgent`, `ClinicalTranslationSafetyAgent`

### `tier_c/forecasting_uncertainty.py`
**Class `ForecastingUncertaintyAgent(EvalAgent)`**: Probabilistic forecasting quality and uncertainty audit.

**Always emits**: ODE time-axis calibration warning (arbitrary units -> cannot claim 3/6/12 months).

Metrics computed:
- **Time-to-event**: Integrated Brier Score (threshold > 0.25 = fail), time-dependent AUC (per horizon), concordance index
- **Calibration**: ECE per horizon (threshold > 0.10 = fail), reliability curves
- **Distributional**: Pinball loss (per quantile level), CRPS (ensemble)
- **Rare mode**: ECE for rare-phenotype patients (threshold > 0.15 = fail)
- **Perturbation budget**: `n_perturbation >= 200` required to resolve tail quantiles for rare resistance modes (derived: N >= 1/(q*eps^2) for q=0.05, eps=0.20)

Score: heuristic 0-1 aggregate; lower-is-better metrics inverted via 1-clip(value,0,1).

### `tier_c/baseline_adversary.py`
**Class `BaselineAdversaryAgent(EvalAgent)`**: Pre-registered head-to-head comparisons.

**Pre-registered design choices** (5 entries):
1. Multi-omics fusion vs transcriptome_only
2. ODE trajectory vs last_observation
3. GNN-on-PPI vs transcriptome_only
4. VAE latent vs clinical_only
5. Survival head vs simple_survival

**Dataclass `BaselineComparisonSpec`**: Pre-registered comparison row with bootstrap CI and BH adjustment.
**Dataclass `IncrementalValueRow`**: One row of the incremental-value table.

Key principles:
1. **Pre-registration**: Full schema declared before any number computed
2. **No training**: Agent only adjudicates existing predictions
3. **Nested CV + bootstrap CIs + Benjamini-Hochberg**: FDR=0.10 across family
4. **Incremental value table**: Each architectural choice paired with deletion baseline and empirical delta

Verdict: FAIL if any baseline matches/beats main model with BH significance; CONDITIONAL if any choice unjustified; PASS otherwise.

Helpers:
- `_bootstrap_delta()`: Bootstrap metric delta (main - baseline) with two-sided p-value
- `_benjamini_hochberg()`: BH multiple testing correction
- `_auc_score()`: Mann-Whitney U based AUC (no sklearn dependency)
- `_resolve_metric()`: Maps metric name to callable

### `tier_c/clinical_safety.py`
**Class `ClinicalTranslationSafetyAgent(EvalAgent)`**: Actionability and clinical-safety review.

**Drug safety notes** (6 drugs): Bortezomib (peripheral neuropathy), Lenalidomide (thromboembolism), Dexamethasone (hyperglycemia), Carfilzomib (cardiopulmonary - potentially fatal), Pomalidomide (hematologic toxicity), Daratumumab (infusion reactions, blood-bank interference).

**Pre-registered therapy failure modes** (7 modes): False negative for aggressive disease, overconfident calls on rare phenotypes, calibration drift, subgroup-blind metrics, time-axis miscalibration, memorized biomarker, no abstention pathway.

**CLIA-adjacent evidence ladder**:
- Analytical validity: accurate on benchmark data
- Clinical validity: correlates with outcome in independent cohort
- Clinical utility: improves patient outcomes vs standard of care

Checks: drug_safety_map, CLIA ladder, subgroup audit (flags subgroups >0.10 below cohort median), OOD detector presence, abstention rate.

Score: fraction of CLIA rungs cleared (max 1.0 only with utility). Never PASS without utility data.

---

## 6. Tier D Agent (Governance Chair)

### `tier_d/__init__.py`
Exports: `PrincipalIntegratorAgent`, `ChairReport`

### `tier_d/chair.py`
**Class `PrincipalIntegratorAgent(EvalAgent)`**: Integrates all lower-tier findings into final assessment.

**Dataclass `ChairReport`**: `overall_grade` (TRL 1-9), `blocking_issues`, `required_changes_prioritized` (sorted by severity), `re_run_conditions`, `rationale`, `per_tier_summary`, `debate_log`.

**Dataclass `PrioritizedChange`**: `description`, `severity`, `severity_score`, `source_agent`, `source_tier`.

**Severity ordering**: block_clinical(100) > block_publication(90) > block_merge(80) > high(70) > medium(50) > low(30) > info(10).

**TRL grading** (walk-down from 9):
- No clinical utility evidence -> TRL <= 6
- No clinical validity evidence -> TRL <= 4
- Tier C failure -> TRL <= 3
- Tier B failure -> TRL <= 3
- **Tier A failure -> TRL bounded above at 2** (hard rule documented: data foundations broken invalidates technology concept)

**Debate resolution**: Only peer-reviewed-style evidence (citations, DOIs, `peer_reviewed: True`) and pre-registered experiments (`pre_registered: True`) carry weight. Anonymous opinions/post-hoc rationalizations discarded. Admissible evidence weight: pre_registration=5, peer_reviewed=3, citation/doi=2.

**Severity classification**: "fatal/cardiac/potentially fatal" -> block_clinical; "calibrate ode/leak/identifiability" -> block_merge; Tier A FAIL -> block_merge; Tier C FAIL -> block_publication; "clinical validity/prospective" -> high; "subgroup/ood" -> high; "report/document" -> medium; else -> low.

---

## 7. Baselines

### `resistancemap/evaluation/baselines.py` (top-level, older)
Required baseline panel with registration dicts.

Functions:
- `predict_global_mean()`, `predict_per_drug_mean()`, `ridge_on_esm2_meanpool()`: Regression baselines
- `predict_base_rate()`, `logistic_on_esm2()`: Classification baselines
- `deepcdr_reimpl()`: Stub (raises NotImplementedError)
- `require_baseline_row(results_table)`: Fails if `per_drug_mean` or `ridge_esm2` missing from results table

Dicts: `BASELINES_REGRESSION`, `BASELINES_CLASSIFICATION` (4 and 3 entries respectively).

### `resistancemap/evaluation/baselines/` (subpackage, newer)

#### `baselines/base.py`
**Class `Baseline(ABC)`**: Abstract base class. Subclasses must implement `fit(X_train, y_train, meta)` and `predict(X)`. Optional: `predict_survival(X, times)`.
- `_require_nonempty(X, name)`: Raises ValueError for None/empty inputs -- hard rule, no fabrication.
- `_require_fitted()`: Raises RuntimeError if predict called before fit.

#### `baselines/clinical_only.py`
**Class `ClinicalOnlyBaseline(Baseline)`**: L2-penalized logistic regression on clinical features (age, ISS stage, cytogenetic risk). Uses scipy L-BFGS-B when available, otherwise numpy gradient descent. Feature normalization (zero-mean, unit-variance).

#### `baselines/transcriptome_only.py`
**Class `TranscriptomeOnlyBaseline(Baseline)`**: PCA via numpy SVD + linear head. Truncated SVD -> ridge regression (continuous) or IRLS logistic (binary). No sklearn PCA dependency.

#### `baselines/last_observation.py`
**Class `LastObservationBaseline(Baseline)`**: Per-patient last-observation carry forward. Two input paths: 2D y_train (n_patients, n_visits) or meta['longitudinal_records'] dict. Refuses cross-sectional data. Falls back to cohort mean for unseen patients (with warning).

#### `baselines/simple_survival.py`
**Class `SimpleSurvivalBaseline(Baseline)`**: Stratified Kaplan-Meier + optional one-covariate Cox PH (when scipy available). Column 0 of X = stratum indicator. y_train shape (n, 2): [time, event]. Also implements `predict_survival(X, times)` returning per-patient survival probabilities at given times.

Helper functions: `_kaplan_meier()` (pure-numpy KM), `_fit_cox()` (Newton-Raphson on neg partial log-likelihood with Breslow tie correction).

---

## 8. Ablation Harness

### `resistancemap/evaluation/ablation_harness.py`
Honest ablation harness with frozen upstream + paired bootstrap.

**ABLATIONS list** (8 entries): vae_only, vae_plus_trajectory, vae_plus_gnn, vae_plus_trajectory_no_gnn, vae_plus_gnn_no_trajectory, full_minus_fusion, full_minus_landscape, full.

**Dataclass `AblationResult`**: name, per_seed scores, mean/std properties.

- `paired_bootstrap_pvalue(scores_a, scores_b, n_boot=10000, seed=0) -> float`: Two-sided paired bootstrap p-value. Requires identical sample indices.
- `run_ablation(name, train_and_eval_fn, seeds)`: Runs train_and_eval_fn(name, seed) for each seed. Contract: identical CV split per seed, frozen upstream weights.
- `ablation_table(results) -> DataFrame`: Builds table with ablation, mean, std, n_seeds, p_vs_prev, significant, verdict ("baseline"/"significant"/"n.s.").

---

## 9. Calibration

### `resistancemap/evaluation/calibration.py`
Calibration metrics and conformal intervals.

- `empirical_coverage(y_true, lo, hi, nominal=0.9) -> dict`: Empirical coverage vs nominal, with miscoverage.
- `interval_score(y_true, lo, hi, alpha=0.1) -> float`: Gneiting-Raftery proper interval score (lower is better).
- `conformal_quantile(residuals_cal, alpha=0.1) -> float`: Split-conformal quantile.
- `conformal_trajectory_intervals(pred, residuals_cal, alpha) -> (lo, hi)`: Conformal prediction intervals.
- `reliability_table(probs, y_true, n_bins=10) -> list[dict]`: Reliability diagram rows for binary calibration audits.

### `resistancemap/evaluation/post_hoc_calibration.py`
Post-hoc calibration methods.

**Class `TemperatureScaling`**: Single-parameter T applied to logits before softmax. Fit via NLL minimization on validation set. T>1 = under-confident, T<1 = over-confident.
- `fit(logits, labels)`: Minimizes NLL via scipy.optimize.minimize_scalar
- `calibrate(logits) -> probs`: Apply fitted temperature

**Class `IsotonicCalibrator`**: Non-parametric isotonic regression (requires sklearn). Per-class isotonic regressors for multiclass; single for binary.
- `fit(probs, labels)`, `calibrate(probs)`

**Class `CalibrationDiagnostics`**: Computes ECE, MCE, reliability curves before and after calibration.
- `compute(probs_before, probs_after, labels, n_bins=15) -> dict`: Returns before/after/improvement dict with ECE/MCE change.
- Fallback implementations if metrics.calibration unavailable.

**Function `calibrate_model_outputs(logits, labels, val_logits, val_labels, method) -> (probs, diagnostics)`**: Convenience function for temperature or isotonic calibration.

---

## 10. Leakage Detection

### `resistancemap/evaluation/leakage.py`
Two narrowly-scoped leakage audits.

**Dataclass `LeakageReport`**: `duplicate_patient_ids` (dict), `temporal_leaks` (list), `clean` (bool), `details` (dict). Properties: `overlaps`, `violations`, `leaking_patients`.

- `audit_patient_split(train_ids, val_ids, test_ids) -> LeakageReport`: Detects duplicate patient IDs across splits. Accepts both positional and keyword args. `clean=True` only when no duplicates AND non-empty train+test.

- `detect_temporal_leakage(samples, split_assignments, ...) -> LeakageReport`: Flags temporal leakage where train-time >= test-time for any patient. For each patient appearing in multiple splits, compares max(train_time) vs min(test_time). Supports polymorphic entry: either (samples, split_assignments) or separate train=/val=/test= sample lists.

- `_coerce_timestamp(value) -> float|None`: Best-effort conversion of datetime, date, ISO strings, or numerics to float timestamp.

---

## 11. Benchmarking & Comparison

### `resistancemap/evaluation/benchmarking.py`
SOTA comparison framework (Gaps 3.6 and 3.7).

**Metric classes:**
- `HarrellsCIndex`: Censoring-adjusted concordance. O(N^2) pairwise.
- `TimeDependentAUC`: Uno's estimator with KM censoring weights at fixed horizons.
- `IntegratedBrierScore`: IPCW integrated Brier score using trapezoidal integration.
- `CalibrationAssessor`: Slope/intercept via weighted logistic regression; Hosmer-Lemeshow chi-squared test.
- `NetReclassificationImprovement`: NRI with SE and z-test; IDI.
- `DecisionCurveAnalysis`: Net benefit computation at decision thresholds; multi-model DCA curves.

**Dataclass `BenchmarkResult`**: method_name, regression_metrics, classification_metrics, shapley_importance, flops, inference_time_sec, memory_mb, cross_dataset_auc, reproducibility_score.

**Class `ComputationalProfiler`**: `profile_inference()` (time + tracemalloc), `estimate_flops()` (simplified FC estimate).

**Class `SOTABenchmarkRunner`**: 11-method comparison (DrugCell, DeepCDR, GraphDRP, PathDSP, PaccMann, MOLI, scDrug, TCRP, PRECISE, Velodrome, ResistanceMap).
- 6-phase protocol: baseline -> integration -> statistical comparison (DeLong) -> computational profiling -> cross-dataset generalization -> reproducibility checklist.
- `_delong_test()`: DeLong's test for comparing two AUCs via Mann-Whitney U matrices.
- Regression metrics: Pearson r, Spearman rho, RMSE, MAE, R^2.
- Classification metrics: AUC-ROC, balanced accuracy, F1, MCC.

**Class `BenchmarkOrchestrator`**: Coordinates Gap 3.6 (clinical risk scores) + Gap 3.7 (SOTA) into unified report.
- `benchmark_clinical_scores()`: C-index (pass >= 0.55), td-AUC at 2/3/5yr, IBS, calibration slope (pass 0.85-1.15), intercept (pass |intercept| <= 0.1), Hosmer-Lemeshow (pass p>0.05), NRI (pass >0.05), IDI (pass >0.01), DCA net benefits.

### `resistancemap/evaluation/model_comparison.py`
Unified writer/reader for baseline-vs-MORTFM comparison table.

**Schema** (frozen): model_name, split, metric_name, metric_value, ci_low, ci_high, n, seed, hash_data (sha256[:16] of sorted patient IDs), hash_code (git SHA).

**Pydantic `ComparisonRow`**: Validates schema with field descriptions.

- `hash_patient_ids(patient_ids) -> str`: Canonical split-hash (sort + newline-join + sha256[:16]).
- `write_comparison_rows(rows, path, append=True) -> Path`: Append to parquet file.
- `read_comparison(path) -> DataFrame`: Read + validate schema.

**Honest-behaviour invariant**: No metric computed inside this module. Callers compute metrics; module only does I/O + schema validation. No np.random / synthetic value ever written.

---

## 12. External Validation

### `resistancemap/evaluation/external_validation.py`
External validation with real cohort loaders (parallels `ablations.py`).

**Dataclass `TRIPODAIChecklist`**: 27-item compliance checklist (title, abstract, ..., availability).

**Class `TRIPODAIComplianceChecker`**: Scores checklist items (non-empty = complete). Generates compliance report with `passes_tier1` threshold at >= 85%.

**Dataclass `ExternalCohort`**: name, n_samples, features, risk_scores, times, events, metadata.

**Class `ExternalCohortLoader`**:
- AVAILABLE_COHORTS: MMRF_CoMMpass (n=996, GDC IA22)
- UNAVAILABLE_COHORTS: GMMG_MM5 (604), IFM_DFCI_2009 (323), PETHEMA_GEM (1265), HOVON_65 (290) -- all raise NotImplementedError with actionable next steps
- `load_mmrf_compass(data_ready_path)`: Loads from data_ready.pt
- No synthetic data fabrication; previous synthetic-fallback removed in v7

**Class `ValidationMetricsCalculator`**: harrell_c_index, time_dependent_auc, integrated_brier_score, calibration_slope_intercept, calculate_all_metrics.

**Class `BootstrapConfidenceEstimator`**: bootstrap_ci (percentile method), delong_test.

**Dataclass `PostMarketAudit`** / **Class `PostMarketMonitor`**: Quarterly post-market monitoring. DRIFT_THRESHOLD = 0.05 C-index decline. quarterly_audit(), get_audit_summary().

**Class `ExternalValidationOrchestrator`**: Four-tier validation pipeline.
- Tier 1: TRIPOD+AI >= 85% compliance
- Tier 2: Single external cohort, C-index decline <= 0.05
- Tier 3: Multi-center >= 2 sites with fairness audit (C-index std < 0.05). NOTE: only 1 cohort has real connector; Tier 3 cannot currently run.
- Tier 4: Prospective 200-300 patients with post-market monitoring

### `resistancemap/evaluation/prospective_validation.py`
FDA PCCP framework for prospective validation study design.

**Dataclass `LandmarkAnalyzer`**: Time-dependent AUC via landmark Cox models at 3/6/12 months. `fit_landmark_cox()` fits Cox PH with accumulated biomarkers at each landmark (Newton-Raphson Hessian for SEs). `compute_time_dependent_auc()` via Uno's estimator.

**Dataclass `SampleSizeCalculator`**: Sample size for landmark survival studies. `sample_size_from_c_index()` via Hanley-McNeil variance. `calculate_cohort_sizes()`: 55% training (200-250), 20% test (60-75), 25% external (100-150). `validate_adequacy()`: checks total >= 350, events >= 15*n_predictors.

**Dataclass `InterimAnalysisManager`**: O'Brien-Fleming spending function for alpha management. Futility gate at 33% info (AUC >= 0.65), efficacy gate at 67% (AUC >= 0.75). `obrien_fleming_spending(t)`, `compute_critical_values()`, `evaluate_interim_gate()`.

**Dataclass `PCCPFramework`**: FDA Predetermined Change Control Plan. Allowed updates: recalibration, coefficient adjustment, harmonization. `check_retraining_trigger()`: C-index drop below threshold OR accumulated events. `generate_specification_document()`.

**Dataclass `BiomarkerScheduler`**: Collection schedule at 0/1/3/6/12/18/24 months. Assays: WGS/WES, scRNA-seq (10X), MRD by NGS (10^-5/10^-6), M-protein (SPEP/IFE).

**Dataclass `ProspectiveStudyDesigner`**: Orchestrates all components into complete study protocol. `generate_study_protocol()`, `summarize_protocol()`.

---

## 13. Other Evaluation Modules

### `resistancemap/evaluation/ablations.py`
TRIPOD+AI compliance, external cohort loading, validation metrics, post-market monitoring, external validation orchestrator. (Near-duplicate of external_validation.py with minor differences; both files contain `TRIPODAIChecklist`, `ExternalCohortLoader`, `ValidationMetricsCalculator`, `BootstrapConfidenceEstimator`, `PostMarketMonitor`, `ExternalValidationOrchestrator`.)

### `resistancemap/evaluation/competing_risks.py`
Competing risks models for intrinsic vs acquired resistance.

**Class `CompetingRisksModel`**: Cause-specific hazard model for two failure modes. `fit()` via MLE with Nelder-Mead. `compute_cif()` via numerical integration (scipy.integrate.quad).

**Class `MixtureCureModel`**: Population split into cured (permanently sensitive, fraction pi) and uncured. Survival plateaus at S(inf) = pi. `fit()` via MLE (4 parameters: cure_fraction, hazard_intrinsic, hazard_acquired, mixture_prob). `survival_function()`.

Convenience functions: `fit_competing_risks()`, `compute_cif()`.

### `resistancemap/evaluation/cytogenetic_stratification.py`
Bayesian hierarchical risk modeling by cytogenetic group (Gaps 3.3 & 3.4).

**Class `FocalLoss(nn.Module)`**: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t). Default alpha=0.25, gamma=2.0.

**Class `CytogeneticFeatureEncoder(nn.Module)`**: 8 binary markers -> embedding (del(17p), t(4;14), t(14;16), t(14;20), gain(1q), del(1p), non-hyperdiploidy, treatment).

**Class `LatentSynthesizer(nn.Module)`**: VAE for synthetic sample generation for n=3-5 rare subtypes. Encoder -> (mu, sigma), decoder, reparameterization trick.

**Class `BayesianHierarchicalRisk(nn.Module)`**: Partial pooling across cytogenetic groups. alpha[i] ~ Normal(mu_alpha_global, sigma_alpha) with elastic net. Outputs: 5 risk classes + PFS/OS predictions.

**Class `PowerPriorEstimator`**: Bayesian power prior for n=1-3 rare subtypes. Discounts historical data with alpha in [0.3, 0.5]. `posterior_estimate()` via Normal-Normal conjugacy.

**Class `MAMLMetaLearner(nn.Module)`**: Model-Agnostic Meta-Learning for n=5-10 subtypes. Inner loop (3-5 steps), outer loop (1000 steps).

**Class `DecisionCurveAnalyzer`**: Net benefit analysis per cytogenetic group. `optimal_threshold()`, `analyze_all_groups()`.

**Class `CytogeneticRiskStratifier(nn.Module)`**: Top-level orchestrator routing by sample size.
- n=1-3: Power prior
- n=3-5: VAE synthesis + Platt calibration
- n=5-10: MAML + focal loss
- n >= 10: Direct training with SMOTE + focal loss

### `resistancemap/evaluation/epistemic_boundaries.py`
Epistemic boundaries and formal claim structure.

**Enum `EvidenceLevel`**: OBSERVED, INTERPOLATED, EXTRAPOLATED, HYPOTHESIZED, UNSUPPORTED.

**Dataclass `EpistemicClaim`**: statement, evidence_level, conditions, disclaimers, validation_requirements, references.

**Dict `CLAIM_REGISTRY`** (6 claims):
1. `state_classification`: OBSERVED level. Cell-line trained; patient validation required.
2. `temporal_trajectory`: INTERPOLATED level. Requires >= 3 time points. ODE model is simplification.
3. `pathway_attribution`: INTERPOLATED level. Correlational, not causal without CRISPR/RNAi.
4. `drug_specific_predictions`: EXTRAPOLATED level. Limited to GDSC/CTRPv2 screened compounds.
5. `uncertainty_quantification`: HYPOTHESIZED level. Model-based; unknown unknowns not captured.
6. `competing_risks`: HYPOTHESIZED level. Two-pathway decomposition; hybrid mechanisms not modeled.

**Class `ClaimValidator`**: `validate_claim()` checks context-specific warnings; `generate_disclaimers()` collects all mandatory disclaimers.

**Function `publication_checklist()`**: 15 mandatory pre-publication checks (sample size, subgroups, calibration, ablation, external validation, competing risks, uncertainty, reproducibility, claim registration, disclaimers, pathway causation, drug applicability, temporal extrapolation, data leakage, no fabrication).

### `resistancemap/evaluation/fairness_audit.py`
Comprehensive fairness audit framework.

**Classes:**
- `AncestryEstimator`: PCA-based ancestry estimation + ADMIXTURE-style proportions
- `RepresentationChecker`: Minimum thresholds for Black (>= 30%) and Hispanic (>= 20%) representation
- `ProxyBiasDetector`: Flags outcome rate disparity > 40% as suspicious cost-based proxy
- `FairnessMetricsCalculator`: C-index parity, calibration parity, equalized odds. ACCEPTABLE_DISPARITY=0.05, ACCEPTABLE_CALIBRATION_BIAS=0.10
- `FairnessAwareConstraint`: Lagrangian penalty for RL policy fairness
- `QuarterlyFairnessAuditor`: DRIFT_THRESHOLD=0.05, CRITICAL_THRESHOLD=0.10 (FDA notification required)
- `FairnessAuditOrchestrator`: Three-phase pipeline
  - Tier 0: Pre-development (representation + proxy bias)
  - Tier 1: Validation (C-index disparity <= 0.05, calibration disparity <= 0.10, equalized odds)
  - Tier 3: Post-market (quarterly drift + FDA escalation if disparity > 10%)

### `resistancemap/evaluation/power_analysis.py`
Sample size adequacy and statistical power analysis.

**Class `SampleSizePowerAnalyzer`**: Power analysis for deep learning validation.
- `_estimate_effective_dof()`: sqrt(n_params), capped by n_samples/2. For 130M params: ~11,400 effective DoF.
- `_compute_power_normal_approx()`: Normal approximation using noncentrality parameter.
- `analyze()`: Computes power, flags CRITICAL/SEVERE/WARNING for samples-per-parameter ratio.
- `compute_minimum_sample_size()`: Binary search for minimum n at desired power.
- `recommend_validation_strategy()`: k-fold CV, bootstrap CIs, effect size, scope narrowing, hematologic stratification.

**Function `flag_sample_size_concerns(n_samples=150, n_params=130M)`**: Quick check for ResistanceMap context.

---

## 14. Governance Layer (`resistancemap/governance/`)

### `governance/__init__.py`
Empty (1 line).

### `governance/claim_gates.py`
Top-level claim-gate evaluator.

**Dataclass `RunEvidence`**: 22 fields capturing what the run produced (endpoint_name, n_patients, n_events, n_longitudinal_pairs, has_c_index, has_patient_disjoint_split, has_real_time_units, coverage rates, etc.).

**Dataclass `ClaimGateReport`**: endpoint_name, claim_levels_allowed, claim_levels_blocked, per_gate details.

**6 gates** (each loaded from `configs/mortfm_gates.yaml`):
1. `gate_static_drug_response`: min_models, min_drugs, min_observed_response_rows, min_model_mapping_rate
2. `gate_pathway_mechanism`: min_gene_to_uniprot_coverage, min_drug_to_target_coverage, min_pathway_mapping_coverage, require_reactome, require_drug_target_edges
3. `gate_survival_claim`: min_patients, min_events, require_c_index, require_patient_disjoint_split
4. `gate_trajectory_claim`: min_longitudinal_pairs, min_unique_patients_with_2_timepoints, require_real_time_units, forbid_pseudotime_as_calendar_time
5. `gate_resistance_claim`: allowed_endpoints list, forbid_endpoint_only_overall_survival
6. `gate_sequence_aware_claim`: require_uniprot_fasta, min_string_protein_coverage, min_feature_gene_coverage, require_embedding_cache

**Function `run_gates(evidence, config_path, out_json) -> ClaimGateReport`**: Evaluates all gates, overlays endpoint_validator, writes JSON report to `logs/mortfm/claim_gate_report.json`.

### `governance/endpoint_validator.py`
Endpoint-semantics validator. Prevents the most common honesty failure: calling an OS-based model a "resistance predictor."

**VALID_ENDPOINTS** (10): overall_survival, progression_free_survival, relapse_time, refractory_status, mrd_conversion, drug_resistance_label, ex_vivo_resistance_shift, ic50, auc, viability, drug_response_ranking.

**Claim levels per endpoint:**
- overall_survival -> survival_head_technical_validation only (NOT resistance)
- progression_free_survival -> survival + resistance proxy (if explicitly framed)
- relapse_time / refractory_status / mrd_conversion / drug_resistance_label / ex_vivo_resistance_shift -> resistance_claim_allowed
- ic50 / auc / viability / drug_response_ranking -> static_drug_response only

**Dataclass `EndpointSemanticsReport`**: endpoint_name, endpoint_type, claim_level, resistance/trajectory/survival/drug_response_claim_allowed, blocking_reasons.

**Function `validate_endpoint_semantics()`**: Returns typed report. Trajectory claims require >= 100 longitudinal pairs.

**Guard functions**: `require_resistance_endpoint()`, `require_trajectory_endpoint()` -- raise `EndpointSemanticsError` if claim not permitted.

---

## Summary of Key Design Patterns

1. **No data fabrication**: Every component enforces this. Missing data -> SKIPPED/error, never synthesized.
2. **Tiered hard-stops**: Tier A FAIL blocks everything downstream (BLOCKED verdict).
3. **Pre-registration**: Baselines, comparisons, and design choices declared before computation.
4. **Honest uncertainty**: Conformal intervals, proper scoring rules, perturbation budget checks.
5. **Audit trails**: SHA-256 verification chain, JSON reports, JSONL findings per run.
6. **Epistemic humility**: Charter encodes what claims are permitted; endpoint validator prevents conflation; evidence levels constrain publication language.
7. **TRL grading**: Conservative 1-9 scale; max TRL 4 reachable from evaluation alone; higher requires in-vivo/prospective/regulated evidence.

---

## Appendix E: Data, Training, Infrastructure, Configs

# Part 5: Data, Training, Infrastructure, and Configs

## 1. `resistancemap/data/` (34 files)

### 1.1 `__init__.py`
Package init re-exports from `loaders`, `preprocessors`, `splits`:
- `load_ccle_proteomics`, `load_ccle_epigenomics`, `load_string_ppi`, `load_drug_sensitivity`
- `harmonize_omics`, `build_train_val_test_splits`
- `make_split`, `assert_no_overlap`

### 1.2 `loaders.py` (~1087 lines)
Core data loading and the primary `MultiOmicsDataset`.

**Class: `MultiOmicsDataset(Dataset)`**
- PyTorch Dataset holding matched proteomics + epigenomics for cell lines/patients.
- Fields: `proteomics` (N,P), `epigenomics` (N,E), `sample_ids`, `lineage`, `drug_sensitivity` (N,D), `protein_names`, `epigenome_feature_names`, `ppi_edges`, `ppi_scores`, `source`, `drug_names`, `drug_target_mean`, `drug_target_std`, `patient_ids`, `scrna_pseudobulk`, `scrna_stages`, `scrna_samples`, `scrna_sources`, `scrna_gene_names`, `crispr_effect`, `crispr_gene_names`, `protein_sequences`, `mmrf`.
- `__getitem__` returns dict with `proteomics`, `epigenomics`, `drug_sensitivity` tensors.
- `subset_by_lineage(lineages)` -> filtered MultiOmicsDataset.

**Functions:**
- `load_ccle_proteomics(config)` -> dict with `data` (DataFrame), `sample_ids`, `protein_names`. Applies coverage filter, UniProt-to-gene mapping, strips `(entrez_id)` decoration.
- `load_ccle_epigenomics(config)` -> dict. Loads chromatin_profiling, atac_seq, h3k4me3, h3k27me3 from directory.
- `load_string_ppi(config)` -> dict with `edges`, `scores`, `proteins`, `num_edges`, `num_proteins`. Normalizes STRING scores 0-1000 to 0-1, filters by `config.ppi_confidence`, translates ENSP to gene symbols via `_load_string_protein_info`.
- `_load_string_protein_info(ppi_path)` -> dict mapping ENSP -> gene_symbol from STRING protein.info file.
- `load_scrna_h5ad(path, config)` -> dict with `adata`, `sample_ids`, `gene_names`.
- `load_drug_sensitivity(config)` -> dict. Merges GDSC + CTRPv2 + PRISM, filters to target drugs, canonicalizes names, canonicalizes sample IDs to DepMap via `_map_sample_ids_to_depmap`, pivots to matrix.
- `_standardize_drug_columns(df, source_name)` -> DataFrame with `sample_id`, `drug_name`, `ic50`.
- `_map_sample_ids_to_depmap(df, sample_info_path)` -> DataFrame with sample_id canonicalized to ACH-NNNNNN.
- `load_scrna_data(config)` -> optional dict keyed by GEO accession.
- `load_depmap_crispr(config)` -> optional dict with CRISPR gene-effect matrix.
- `load_mmrf_data(config)` -> dict with clinical, patient_ids, os_time, os_event, pfs_time, pfs_event, iss_stage, cytogenetics, response_label, expression, mutations. Uses GDC TSV layout.
- `_compute_pfs_from_treatments(treatments_path, patient_ids, os_time, os_event)` -> PFS proxy from treatment lines.
- `_load_gdc_cytogenetics(cyto_path)` -> optional DataFrame.

### 1.3 `preprocessors.py` (~685 lines)
Transforms raw DataFrames into matched PyTorch tensors.

**Functions:**
- `preprocess_scrna(adata, config, ...)` -> preprocessed AnnData. Steps: QC metrics, filter cells (min/max genes, MT%), filter genes, log-normalize, HVG selection.
- `align_protein_names(proteomics_names, ppi_names, esm2_names)` -> (aligned_names, name_mapping). Exact + case-insensitive matching.
- `_load_lineage_labels(sample_ids, config)` -> list of lineage strings from CCLE sample_info.csv.
- `_impute(df, method)` -> DataFrame. Methods: knn, median, zero.
- `_normalize(df, method)` -> (DataFrame, scaler). Methods: quantile, zscore, log2.
- `_build_id_mapping(config)` -> dict CCLEName -> ModelID.
- `harmonize_omics(proteomics, epigenomics, ppi_graph, scrna_data, mmrf_data, crispr_data, config)` -> MultiOmicsDataset. Core harmonization: intersects sample IDs, imputes, normalizes, loads drug sensitivity, z-scores drug targets, attaches scRNA pseudobulk, CRISPR, UniProt sequences, MMRF block.
- `_zscore_normalize_on_train(data, train_idx, fit_on_train_only)` -> (normalized, mean, std). Train-only normalization to prevent leakage.
- `build_train_val_test_splits(dataset, config, fit_on_train_only)` -> dict with train/val/test index arrays. Group-stratified by patient_ids when available, re-normalizes drug targets on train stats.

### 1.4 `preprocessing.py` (~822 lines)
Legacy multi-omics preprocessing pipeline (v6-era).

**Classes:**
- `RNASeqPreprocessor` -- `fit_transform(counts, gene_names, batch_labels)`, `transform(counts)`. Steps: gene filtering, CPM/median-of-ratios/scran normalization, log1p, HVG selection (Seurat v3), batch correction (ComBat/Harmony), scaling.
- `ProteomicsPreprocessor` -- `fit_transform(data, protein_names, protein_gene_map)`, `transform(data)`. Steps: missing filter, log2, median normalization, KNN imputation, protein-gene aggregation, variance selection, RobustScaler.
- `EpigenomicsPreprocessor` -- `fit_transform(data, feature_names)`, `transform(data)`. TF-IDF for ATAC, M-value for methylation, variance selection, optional LSI (TruncatedSVD).
- `PPINetworkLoader` -- `load(string_file, gene_names, alias_file)`, `load_from_precomputed(path)`. Builds sparse adjacency from STRING with confidence threshold.
- `DrugResponseProcessor` -- `fit_transform(drug_response_df)`. Binarizes IC50/AUC using median/waterfall/bimodal (GMM) thresholds.
- `FeatureSelector` -- `select(X, gene_names, y)`. Modes: pathway (curated KEGG + myeloma genes), variance, mutual_info, combined.
- `MultiOmicsQC` -- `filter_samples(modalities, sample_ids)`. Filters by missing modality fraction + MAD-based outlier detection.
- `MultiOmicsPreprocessor` -- orchestrator composing all above.
- `DataSplitter` -- `get_nested_cv(patient_ids, labels, diagnosis_dates)`. Nested stratified K-fold with optional temporal split.

### 1.5 `splits.py` (~107 lines)
Hierarchical group splits for leakage prevention.

- `SPLIT_KINDS = ("patient", "lineage", "drug_holdout")`
- `class LeakageError(RuntimeError)`
- `@dataclass SplitManifest` -- kind, seed, n_folds, group_hash, fold_sizes.
- `make_split(meta, kind, n_folds, seed)` -> (folds, manifest). Uses `GroupKFold`.
- `assert_no_overlap(a_ids, b_ids, key, label_a, label_b)` -- raises `LeakageError` on overlap.
- `assert_folds_disjoint(folds, meta, key)` -- post-hoc fold validation.

### 1.6 `targets.py` (~61 lines)
Target registry: every metric must cite a target key + units.

- `@dataclass Target` -- key, kind (binary/regression), units, primary_metric, range, naive_baseline_required.
- `TARGETS` dict with: `resistance_binary`, `log_ic50`, `normalised_auc`, `trajectory_score`.
- `require_target(key)` -> Target or raises KeyError.

### 1.7 `mmrf_loader.py` (~1022 lines)
Full MMRF CoMMpass data loader.

**Classes:**
- `HGNCMapper` -- gene alias/Ensembl-to-HGNC mapping. `harmonize(gene_id)`, `harmonize_index(genes)`.
- `MMRFCoMMpassDataset(Dataset)` -- PyTorch Dataset for MMRF multi-omics. Supports GDC and MMRF-RG sources. Loads expression (HTSeq/salmon TPM), mutations (MAF), clinical. Builds stratified train/val/test cohort splits. `__getitem__` returns dict with expression, mutations, clinical tensors, labels, metadata.
- `collate_variable_length(batch)` -- pads variable-length sequences with masks.

**Functions:**
- `parse_gdc_clinical(clinical_dir)` -> DataFrame. Parses GDC-format TSV.
- `parse_mmrf_clinical(data_dir)` -> DataFrame. Combines per-patient, treatment response, cytogenetics.
- `extract_response_label(clinical)` -> Series. Binary: sCR/CR/VGPR/PR=1, SD/PD/MR=0.
- `extract_cytogenetic_risk(clinical)` -> DataFrame. t(4;14), t(14;16), del(17p), gain(1q) plus risk_score (0/1/2) and risk_category.
- `extract_iss_stage(clinical)` -> Series.
- `extract_treatment_regimen(clinical)` -> DataFrame with contains_PI, contains_IMiD, contains_dex, line_number.
- `load_gdc_htseq_counts(count_dir, gene_mapper)` -> DataFrame genes x samples.
- `load_mmrf_rnaseq(data_dir, gene_mapper)` -> DataFrame.
- `load_mutation_data(data_dir, source)` -> binary mutation matrix.
- `build_sample_patient_map(data_dir)` -> Dict[sample_id, patient_id].
- `map_samples_to_patients(expression, sample_patient_map)` -> patient-level expression (mean across samples).
- `extract_longitudinal_visits(data_dir, patient_ids)` -> Dict[patient_id, Dict[month, row]].
- `stratified_cohort_split(patient_ids, response_labels, iss_stages, risk_scores, ...)` -> (train, val, test) patient arrays.

### 1.8 `mortfm_dataset.py` (~297 lines)
PyTorch Dataset + collate for MORT-FM training.

**Class: `MORTFMDataset(Dataset[TemporalTrainingPair])`**
- Wraps list of `TemporalTrainingPair` objects.

**Functions:**
- `_stack_modality(snapshots, modality)` -> (stacked_values, presence_mask) or None. Zero-fills missing rows.
- `_stack_drug(snapshots)` -> (values, presence) for drug context (dose, start_time, end_time).
- `_stack_optional_tensor(tensors, fill_shape)` -> stacked tensor with NaN/False/-1 fill.
- `_collate_modalities(snapshots, patient_ids, cell_ids)` -> partial MORTBatch with per-modality tensors + masks.
- `mort_collate(batch)` -> MORTBatch. Stacks baseline modalities, supervision targets (event_time, event_observed, resistance_label, drug_response, future_state), and v19 Phase 1 follow-up snapshot batch (`future_snapshot_batch`, `delta_t_days`).

### 1.9 `trajectory_pair_builder.py` (~286 lines)
Constructs `TemporalTrainingPair` objects from snapshot + outcome streams.

**Functions:**
- `_group_by_patient(snapshots)` -> Dict[patient_id, sorted list of snapshots].
- `_index_outcomes_by_patient(outcomes)` -> Dict[patient_id, sorted outcomes].
- `_match_outcome(patient_id, baseline_time, outcomes_by_patient, tol)` -> Optional[ResistanceOutcome].
- `_make_unlabelled_outcome(snap)` -> stub ResistanceOutcome.
- `build_temporal_pairs(snapshots, outcomes, *, allowed_time_gaps, gap_tolerance, outcome_match_tolerance, include_unlabelled, include_survival_only, include_drug_response_only)` -> List[TemporalTrainingPair]. Groups by patient, constructs longitudinal pairs (x_t, x_t_delta), survival-only rows, drug-response-only rows, unlabelled pretraining rows. Never fabricates.
- `summarise_pairs(pairs)` -> Dict[str, float] diagnostic counts.

### 1.10 `clinical_outcome_loader.py` (~324 lines)
Converts MMRF parsers into MORT-FM schema objects.

**Functions:**
- `load_mortfm_outcomes(data_dir, *, endpoint, response_col_hint, strict)` -> List[ResistanceOutcome]. Endpoints: pfs, os, relapse. Handles GDC vital_status fallback. Maps IMWG response to resistance_label (0=sensitive, 1=tolerant, 3=relapsed-resistant).
- `summarise_outcomes(outcomes)` -> Dict with n_patients, n_with_event_time, n_events_observed, median_event_time_months, event_rate.

### 1.11 `drug_ontology.py` (~216 lines)
Curated MM-relevant drug ontology.

**Dataclass: `DrugOntologyEntry`** -- canonical_name, drug_class, target_genes (HGNC), target_proteins (UniProt), synonyms.

**Registry:** 21 drugs covering proteasome inhibitors (Bortezomib, Carfilzomib, Ixazomib), IMiDs (Lenalidomide, Pomalidomide, Thalidomide), HDAC inhibitors (Panobinostat, Vorinostat, Romidepsin), BCL2 inhibitor (Venetoclax), CDK inhibitors (Dinaciclib, Palbociclib), cytotoxics (Doxorubicin, Etoposide, Cyclophosphamide, Melphalan), mAbs (Daratumumab, Elotuzumab, Isatuximab), Dexamethasone, Selinexor.

**Functions:**
- `lookup_entry(name)`, `lookup_entry_loose(name)` -- case-insensitive, synonym-aware lookup.
- `normalize_drug_name(name)` -- strips parentheticals, punctuation, lowercase.
- `known_classes()`, `known_drugs()`, `class_to_index()`, `drug_to_index()`.
- `materialise_drug_context(name, *, dose, start_time, end_time, combination_id)` -> DrugContext.

### 1.12 `drug_target_loader.py` (~61 lines)
Load drug-target edges from DrugBank/DGIdb/ChEMBL CSV.

- `load_drug_target_edges(path, *, drug_col, target_col, weight_col, default_weight)` -> List[Tuple[str, str, float]].

### 1.13 `chembl_drug_target_loader.py` (~192 lines)
ChEMBL drug->target loader with SQLite and CSV modes.

- `load_from_sqlite(sqlite_path, *, limit)` -> DataFrame with canonical schema.
- `load_from_csv(csv_path)` -> DataFrame.
- `write_chembl_coverage_report(*, drug_targets, gdsc_drugs, prism_drugs, out_json)` -> Dict.

### 1.14 `treatment_exposure.py` (~132 lines)
Convert MMRF treatment regimen tables into DrugContext records.

- `_split_regimen_name(regimen)` -- expands abbreviations (VRD, CyBorD, KRd, DaraVRd, etc.) into component drug names.
- `parse_treatment_exposures(treatment_df, ...)` -> Dict[patient_id, List[DrugContext]].
- `primary_drug_for_patient(exposures, patient_id, *, at_time)` -> Optional[DrugContext].

### 1.15 `feature_alignment.py` (~103 lines)
Projects expression matrix onto reference feature space.

- `@dataclass AlignmentReport` -- n_reference_features, n_source_features, n_features_present, n_features_missing, coverage_rate.
- `align_to_reference_features(source, reference_features, *, case_normalise)` -> (aligned DataFrame, mask DataFrame, AlignmentReport). Zero-fills missing genes with mask tracking.

### 1.16 `identifier_mapping.py` (~197 lines)
Stable cross-namespace identifier mapping (HGNC <-> Ensembl <-> UniProt <-> STRING <-> Reactome <-> ChEMBL).

- `@dataclass IdentifierMaps` -- bidirectional lookup tables with `harmonise_gene_symbol`, `gene_to_uniprot`, `uniprot_to_string_id`, `pathways_for_gene`, `drug_targets` methods.
- `load_identifier_maps(*, hgnc_ensembl_path, uniprot_xref_path, string_alias_path, reactome_membership_path, chembl_target_path)` -> IdentifierMaps.
- `unify_identifiers(identifiers, maps)` -> Dict[str, Optional[str]]. Returns None for unmappable (no fallback).

### 1.17 `manifest_schemas.py` (~304 lines)
Canonical metadata schemas for MORT-FM public-data assembly.

**Dataclasses:**
- `SamplesTable` -- wraps samples.csv. Methods: `patients()`, `patients_in_split(split)`, `validate_patient_disjoint_splits()`, `modality_coverage()`.
- `FeatureManifest` -- wraps feature_manifest.csv. Methods: `features_for_modality(modality)`, `map_to_hgnc(feature_ids)`.
- `DatasetManifest` -- wraps dataset_manifest.csv. Methods: `datasets_by_modality(modality)`, `licenses()`.
- `PublicDataManifest` -- combines all three. `summary()` method.

**Loaders:** `load_samples_table`, `load_feature_manifest`, `load_dataset_manifest`, `load_public_data_manifest`. All validate required columns, uniqueness, split validity.

### 1.18 `model_id_resolver.py` (~288 lines)
Cell-line model-ID resolver normalizing DepMap/Sanger/COSMIC/CCLE/GDSC/PRISM IDs to canonical model_id.

- `load_model_metadata(model_csv_path)` -> DataFrame from DepMap Model.csv.
- `bootstrap_partial_resolver(*, depmap_dir, out_csv)` -> minimal metadata from CRISPRGeneEffect headers.
- `class ModelIDResolver` -- `to_model_id(raw)` maps any-namespace ID. `coverage(*, gdsc_models, prism_models)` -> CoverageReport.
- `write_coverage_report(resolver, *, gdsc_models, prism_models, out_json)`.

### 1.19 `graph_builder.py` (~275 lines)
Build PyTorch Geometric graphs from STRING PPI + ESM-2 embeddings.

- `build_protein_index(protein_names)` -> dict.
- `build_edge_index(ppi_edges, ppi_scores, protein_index)` -> (edge_index [2,E], edge_attr [E,1]). Undirected. Self-loop fallback if no edges match.
- `build_sample_graph(proteomics, esm2_embeddings, memory_state, stability_score, edge_index, edge_attr, drug_sensitivity)` -> PyG Data. Concatenates ESM-2 (1280d) + abundance + VAE latent + stability.
- `build_batch_graphs(...)` -> PyG Batch.
- `load_esm2_embeddings(protein_names, model_name, device)` -> Tensor (placeholder).

### 1.20 `pathway_harmonization.py` (~560 lines)
Pathway scoring, cross-modality harmonization, dimensionality reduction.

**Classes:**
- `GSVAScorer(nn.Module)` -- rank-based GSVA. `forward(expression_matrix)` -> (num_samples, num_pathways) scores.
- `PathwaySetManager` -- loads GMT files, aligns with expression data. `load_gmt(gmt_path, collection_name)`, `create_from_dict(gene_sets, collection_name)`, `align_with_expression_data(gene_names)`.
- `DANNHarmonizer(nn.Module)` -- domain adversarial neural network with gradient reversal for cross-modality pathway harmonization. feature_extractor + domain_classifier.
- `GradientReversalFunction` -- custom autograd: forward=identity, backward=-lambda*grad.
- `PathwayVAECompressor(nn.Module)` -- VAE (250 pathways -> 64d latent). `encode`, `reparameterize`, `decode`, `vae_loss` (MSE + beta-KL), `get_latent`.

### 1.21 `string_debiasing.py` (~640 lines)
STRING PPI debiasing, CRISPR validation, phosphoproteomics augmentation.

**Classes:**
- `STRINGDebiaser` -- `filter_edges(edges_df)` removes text-mining circularity. `gene_level_bias_report(edges_df)`. `debiased_adjacency(edges_df, protein_list)` -> sparse matrix.
- `CRISPRValidator` -- validates model-attributed genes against DepMap CRISPR dependency. `validate_attributions(attributed_genes, cell_line_ids, threshold)`. `compute_precision_at_k(attributed_genes, crispr_hits)`. `enrichment_test(attributed_genes, crispr_hits, background_size)` -- Fisher's exact test.
- `PhosphoproteomicsAugmenter` -- augments protein node features with phosphosite data. `site_to_protein_mapping()`. `augment_features(protein_features, protein_names, phospho_df, aggregation)`.

### 1.22 Modality-specific loaders (MORT-FM schema)

All emit `ModalityTensor` objects for the MORT-FM pipeline.

**`scrna_loader.py` (~208 lines):**
- `load_h5ad_as_modality(path, *, feature_layer, feature_names_col, max_cells)` -> ModalityTensor (cells x genes).
- `pseudobulk_per_patient(path, *, patient_obs_col, aggregation)` -> (patient_ids, ModalityTensor).
- `build_snapshots_from_h5ad(path, ...)` -> List[PatientCellSnapshot]. One snapshot per (patient, timepoint), pseudobulked.

**`scatac_loader.py` (~74 lines):**
- `load_scatac_h5ad(path, *, binarise, max_cells)` -> ModalityTensor (cells x peaks).

**`cite_seq_loader.py` (~73 lines):**
- `load_citeseq_h5mu(path, *, rna_modality_key, protein_modality_key)` -> (rna ModalityTensor, protein ModalityTensor). Uses muon for .h5mu.

**`proteomics_loader.py` (~73 lines):**
- `load_proteomics_matrix(path, *, sample_id_axis, log_transform, min_coverage)` -> ModalityTensor.

**`phosphoproteomics_loader.py` (~59 lines):**
- `load_phosphoproteomics(path, *, sample_id_axis, log_transform)` -> ModalityTensor. log2(|x|+1)*sign(x).

**`methylation_loader.py` (~79 lines):**
- `load_methylation_matrix(path, *, sample_id_axis, as_m_values, epsilon)` -> ModalityTensor. Optional beta-to-M-value conversion.

**`histone_ptm_loader.py` (~88 lines):**
- Channels: H3K27ac, H3K27me3, H3K4me3, H3K9me3, global_acetylation, global_methylation.
- `load_histone_ptm_matrix(path, *, channels)` -> ModalityTensor.

**`perturbation_loader.py` (~111 lines):**
- `load_crispr_gene_effect(path, *, sample_id_axis)` -> (sample_ids, gene_symbols, values Tensor).
- `load_perturbseq_deltas(path, *, perturbation_obs_col, control_label)` -> Dict[perturbation_name, mean delta Tensor].

### 1.23 `reactome_loader.py` (~120 lines)
- `load_reactome_membership(reactome_dir)` -> DataFrame (pathway_id, pathway_name, gene_symbol, uniprot_id, species, evidence_source). Tries HGNC2Reactome, UniProt2Reactome, NCBI2Reactome.
- `load_reactome_hierarchy(reactome_dir)` -> DataFrame (parent_pathway_id, child_pathway_id). HSA-only.
- `write_pathway_coverage_report(*, membership, feature_genes, out_json)` -> Dict.

### 1.24 `single_cell_preprocessing.py` (~289 lines)
Canonical scanpy pipeline for MORT-FM single-cell inputs.

- `DEFAULT_DISEASE_STAGE_MAP`: HD/healthy=0, MGUS=1, SMM=2, MM=3, NBM=0, AML=3.
- `@dataclass QCReport` -- n_cells_in/out, filtered counts, pct_mt stats.
- `_looks_like_counts(X)` -> bool heuristic (integer-valued check).
- `preprocess_scrna(adata, *, min_genes, max_genes, max_pct_mt, min_cells_per_gene, n_top_genes, disease_stage_map, ...)` -> AnnData. Steps: validate raw counts, MT annotation, cell filter, gene filter, save counts layer, TP10K + log1p, HVG (Seurat v3), enrich obs with patient_id and `pseudotime_disease_stage` (integer ordinal, NOT calendar time).

### 1.25 `protein_sequence_cache.py` (~301 lines)
Disk cache for UniProt sequences + ESM-2 embeddings.

- `load_sequences(uniprot_dir)` -> DataFrame [uniprot_id, gene_symbol, sequence, length]. Parses FASTA.
- `write_uniprot_coverage_report(...)` -> UniProtCoverageReport.
- `class ESMEmbeddingCache` -- `.pt` file cache. `load(cache_dir)`, `save(cache_dir)`, `compute(uniprot_df, *, batch_size)` (runs ESM-2 via transformers), `get(uniprot_id)`. Cache key includes model name + sequence MD5. Refuses to return zeros on failure.

### 1.26 `uniprot_loader.py` (~234 lines)
UniProt FASTA loader for ESM-2 protein-sequence embeddings.

- Strategy: parquet cache -> bulk FASTA -> UniProt REST (rate-limited, gated by `allow_network`).
- `load_protein_sequences(protein_names, cache_dir, bulk_fasta, allow_network, write_cache)` -> list[Optional[str]]. Misses are explicit None, never synthetic.
- `_parse_fasta_blob(text)`, `_load_bulk_fasta(fasta_path)`, `_fetch_one_symbol(symbol)`.

---

## 2. `resistancemap/training/` (6 files)

### 2.1 `__init__.py`
Exports: `ResistanceMapTrainer`, `ResistanceMapLoss`, `AutomaticLossWeighting`, `CausalConsistencyLoss`, `DisentanglementLoss`, `IdentifiableODELoss`, `LyapunovStabilityLoss`, `PathwayAnchorLoss`.

### 2.2 `losses.py` (~971 lines)
Loss functions for ResistanceMap v6.

**Classes:**
- `IdentifiableODELoss(nn.Module)` -- `trajectory_mse(pred, true, mask)`, `sparsity_l1(A)`, `diagonal_dominance_penalty(A)`, `orthogonality_penalty(P)`. Forward returns dict of all components + total.
- `LyapunovStabilityLoss(nn.Module)` -- enforces V(x*)=0, V(x)>0, dV/dt < -epsilon near attractors. Samples points in ball, computes grad(V).f(x).
- `DisentanglementLoss(nn.Module)` -- beta-TCVAE total correlation. Decomposes KL into MI + TC + DW-KL. Only TC upweighted by beta. Annealing schedule.
- `PathwayAnchorLoss(nn.Module)` -- ties latent dims to pathway mean expression via Pearson correlation. Specificity penalty against cross-pathway correlation.
- `CausalConsistencyLoss(nn.Module)` -- NOTEARS acyclicity penalty h(W) = tr(e^{WoW})-d. Intervention consistency checks attention weights respect causal ordering.
- `AutomaticLossWeighting(nn.Module)` -- Kendall et al. 2018 homoscedastic uncertainty. Learnable log(sigma^2) per task. `get_weights_summary()`.
- `ResistanceMapLoss(nn.Module)` -- composite of all 8 task losses (drug_response, trajectory_mse, reconstruction, lyapunov_stability, tc_penalty, sparsity_l1, pathway_anchor, causal_consistency) with auto or fixed weighting.

### 2.3 `loss_router.py` (~132 lines)
Disciplined loss composition with guardrails.

- `@dataclass LossAuditEntry` -- name, fired, weight, raw_value, skipped_reason.
- `class LossRouter` -- `compute(outputs, batch)` -> (total, components, audit). Validates required supervision per loss; refuses to add term if supervision missing. Records per-step audit trail.

### 2.4 `mortfm_losses.py` (~372 lines)
Composable loss functions for MORT-FM (pure functions, no module state).

**Functions:**
- `reconstruction_loss(targets, params, *, distribution)` -- dispatches to gaussian_nll / nb_nll / bernoulli_nll.
- `masked_modality_loss(predicted, target, *, presence_mask)` -- L2 on masked modality predictions.
- `contrastive_alignment_loss(embeddings_a, embeddings_b, *, temperature)` -- symmetric InfoNCE.
- `drug_response_loss(predicted, target)` -- MSE, NaN-target rows ignored.
- `trajectory_distribution_loss(predicted, target, *, metric)` -- MSE, MMD (RBF kernel), or Sinkhorn (via geomloss).
- `optimal_transport_trajectory_loss(predicted, target)` -- Sinkhorn alias.
- `survival_loss(head, head_output, event_time, event_observed)` -- dispatches to Cox or discrete-time NLL.
- `resistance_state_loss(logits, targets)` -- cross-entropy, invalid targets ignored.
- `pathway_attribution_loss(protein_scores, positive_targets)` -- BCE + L1 sparsity.
- `perturbation_consistency_loss(predicted_delta, observed_delta)` -- 1 - cosine similarity.
- `landscape_regularization_loss(potential, z_batch, ...)` -- smoothness + basin separation + drug barrier penalties.
- `counterfactual_consistency_loss(predicted_rank_scores, oracle_rank_scores)` -- pairwise margin/hinge loss.
- `calibration_loss(predicted_probability, observed)` -- ECE surrogate (differentiable).
- `assemble_total_loss(components, weights)` -> (total, breakdown dict).

### 2.5 `canonical_mortfm_losses.py` (~746 lines)
Loss functions consuming the canonical MORTFM.forward() LENS dict (v18+).

**Stage spec:**
- `STAGE_LOSS_SPEC`: E=(trajectory, sde_path, basin_transition), F=(survival, hitting, hitting_nll, surv_hit_consistency), G=(basin_transition), H=(survival).
- `REQUIRED_SUPERVISION_PER_STAGE`: E=trajectory, F=survival, G=basin_transition, H=survival.

**Functions:**
- `lens_survival_loss(out, batch)` -- discrete-time NLL + Brier score on LENS survival surface. Reads hazard, survival_curve, cif_per_event.
- `hitting_time_loss(out, batch)` -- discrete NLL of observed event time against hitting_cdf + MSE on mean_tau.
- `basin_transition_loss(out, batch)` -- cross-entropy on terminal basin probability vs resistance_label.
- `sde_path_loss(out, batch)` -- path length + smoothness regularizer on z_traj. Always finite.
- `canonical_trajectory_loss(predicted_z_traj, encoded_z_future, delta_t)` -- per-row latent L2 weighted by 1/delta_t.
- `latent_future_state_loss(predicted_z_traj, encoded_z_future, delta_t_days)` -- Phase 1 canonical trajectory loss with optional MMD bonus (gated by batch size >= 16).
- `hitting_time_nll(hitting_cdf, t_grid, event_time, event_observed)` -- Phase 7 discrete NLL on canonical grid.
- `survival_hitting_consistency_loss(survival_curve, hitting_cdf, cif_other_events)` -- Phase 7 hinge penalty S(t) <= 1-F_hit(t) + sum(c_other).
- `assemble_canonical_loss(stage, out, batch, weights)` -- dispatches all terms for a stage, returns bundle with loss_total, components, per_term_n_supervised, required_term.

### 2.6 `trainer.py` (~889 lines)
ResistanceMap v6 multi-task trainer.

**Classes:**
- `EMAModel` -- exponential moving average of model parameters. `update(model)`, `apply_shadow(model)`, `restore(model)`.
- `EarlyStopping` -- patience-based with min_delta, max/min mode.
- `CheckpointManager` -- `save(epoch, model, optimizer, scheduler, scaler, metrics)`, `load_latest()`, `load_best()`. Keeps last N, saves best by monitor metric.
- `ExperimentTracker` -- unified wandb/TensorBoard interface. `log(metrics, step)`, `finish()`.
- `ResistanceMapTrainer` -- full training loop.
  - Optimizer: AdamW with configurable lr, weight_decay, betas.
  - Scheduler: CosineAnnealingWarmRestarts.
  - Mixed precision: torch.cuda.amp autocast + GradScaler.
  - Gradient clipping: max_norm=1.0.
  - `_train_one_epoch(train_loader)` -- forward, composite loss, backward, clip, EMA update, TC annealing.
  - `_validate(val_loader)` -- EMA shadow, compute AUROC/AUPRC.
  - `fit(train_loader, val_loader, epochs, resume)` -- full loop with early stopping, checkpointing, tracking.
  - `evaluate(test_loader)` -- AUROC, AUPRC, F1, MCC, Brier, trajectory MSE.

---

## 3. `resistancemap/infrastructure/` (8 files)

### 3.1 `__init__.py`
Lazy imports with fallback handling for: missing_modality, continual_learning, uncertainty_quantification, multi_site_harmonization, scrna_integration, imaging_radiomics.

### 3.2 `guardrails.py` (~89 lines)
8 executable guardrail predicates.

- `_rule_probability_range` -- resistance_score in [0,1].
- `_rule_ic50_range_log` -- log_ic50 in [-4, 4].
- `_rule_monotone_dose` -- dose-response curves non-increasing.
- `_rule_stochastic_ci` -- CI width > 1e-6 (non-degenerate UQ).
- `_rule_mechanism_in_vocab` -- top mechanisms in vocabulary.
- `_rule_drug_in_training` -- drug_holdout tag when drug unseen.
- `_rule_latent_nan_free` -- no NaN/Inf in latent_z.
- `_rule_cohort_access_tag` -- refuse public output from controlled-access data.
- `adjudicate(outputs)` -> list of failing guardrail names.

### 3.3 `missing_modality.py` (~668 lines)
Robust handling of missing modalities.

**Classes:**
- `ModalitySpecificEncoder(nn.Module)` -- per-modality encoder producing Gaussian posterior (mu, logvar).
- `ProductOfExpertsVAE(nn.Module)` -- multiplicative combination of available posteriors: mu_combined, var_combined via precision-weighted sum.
- `FlexMoERouter(nn.Module)` -- dynamic expert routing based on modality availability. Learned importance weights + gating network.
- `ModalityMaskingStrategy(nn.Module)` -- training-time random masking (p_mask=0.3).
- `SelectivePredictionGate(nn.Module)` -- uncertainty-based abstention. Learned modality importance, degradation coefficients. Empirical: 1 minor=~2% AUC drop, 1 major=~5-8%, 2+ major=~15-20% -> abstain.
- `MissingModalityHandler(nn.Module)` -- top-level orchestrator. Forward: mask -> encode each modality -> PoE-VAE -> reparameterize -> MoE routing -> selective prediction gate. Returns z, logvar_z, uncertainty, abstention_mask, availability.

### 3.4 `uncertainty_quantification.py` (~757 lines)
6-layer UQ architecture.

**Dataclass: `UncertaintyOutput`** -- point_estimate, aleatoric_variance, epistemic_variance, prediction intervals, risk_category, confidence_band, abstain_flag.

**Classes:**
- Layer 1: `DeepEnsembleWrapper(nn.Module)` -- K independent models, returns ensemble mean + variance.
- Layer 2: `EvidentialRegressionHead(nn.Module)` -- Normal Inverse-Gamma (NIG) distribution. Outputs (gamma, nu, alpha, beta). Aleatoric = beta/(alpha-1), epistemic = beta/(nu*(alpha-1)). NIG NLL + KL regularization.
- Layer 3: `BayesianODEWrapper(nn.Module)` -- MC sampling of ODE parameters. Returns (mean_trajectory, lower_band, upper_band).
- Layer 4: `ConformalCalibrator(nn.Module)` -- Conformalized Quantile Regression. `calibrate(x_calib, y_calib)` computes q_hat. `predict_with_conformal(x)` adjusts intervals.
- Layer 5: `SelectivePredictionGate(nn.Module)` -- abstention on high epistemic uncertainty. `get_coverage_accuracy_curve`.
- Layer 6: `ClinicalUncertaintyCommunicator(nn.Module)` -- maps to risk categories (Low/Moderate/High/Very High) with 95% CI bands.
- `UncertaintyQuantifier(nn.Module)` -- top-level orchestrator. `forward(x)` -> UncertaintyOutput. `compute_loss(x, y)` combines ensemble MSE + evidential NLL+KL + CQR quantile loss.

### 3.5 `continual_learning.py` (~994 lines)
Dynamic monitoring and continual learning for clinical deployment.

**Dataclasses:** `PatientObservation`, `ClinicalEvent`.

**Classes:**
- `JointLongitudinalSurvival(nn.Module)` -- joint longitudinal (M-protein trajectory with random intercept+slope) + survival (Weibull baseline hazard + Cox linkage via gamma * m_i(t)). `cox_partial_likelihood`.
- `EWCRegularizer(nn.Module)` -- Elastic Weight Consolidation. `compute_fisher_diagonal`, `consolidate`, `penalty` = lambda_EWC * sum(F_i * (w_i - w*_i)^2).
- `NeuralODEKalmanFilter(nn.Module)` -- Extended Kalman Filter for Neural ODE dynamics. `predict` (Euler step + covariance propagation), `update` (measurement update with Kalman gain). Finite-difference Jacobians.
- `StreamingCoxUpdate(nn.Module)` -- incremental partial likelihood updates for Cox model.
- `ClinicalTriggerMonitor` -- rule-based detection: >50% M-protein reduction (response), MRD conversion, +/-25% M-protein change, >2xULN LDH, new lesions, eGFR drop >25%, CUSUM/EWMA on log(M-protein).
- `ContinualLearningManager` -- orchestrator. `add_observation(observation)` -> (events, model_updated). Coordinates trigger detection, Kalman state tracking, streaming Cox updates, EWC regularization.

### 3.6 `scrna_integration.py` (~621 lines)
scRNA-seq integration pipeline for resistance prediction.

**Classes:**
- `FoundationModelEncoder(nn.Module)` -- placeholder for scGPT/scFoundation. Projects gene expression to 512d embeddings.
- `ScArchesAdapter(nn.Module)` -- bottleneck adapter (512 -> 128 -> 512) with residual + LayerNorm for transfer learning.
- `HarmonyBatchCorrector(nn.Module)` -- iterative per-batch centroid correction toward global centroid.
- `GatedAttentionMIL(nn.Module)` -- gated attention Multiple Instance Learning. alpha_i = softmax(tanh(W*z) * sigmoid(V*z)). Aggregates cell embeddings to patient embedding.
- `CellNeighborhoodAnalyzer(nn.Module)` -- MILO-style k-NN DA analysis. Computes per-neighborhood log2 odds ratio of resistant vs sensitive cells.
- `CompositionalAnalyzer(nn.Module)` -- Dirichlet-multinomial for cell-type proportion shifts.
- `ScRNAIntegrationPipeline(nn.Module)` -- orchestrator: encode_cells -> adapt_embeddings -> correct_batch -> aggregate_to_patient. Also: analyze_neighborhoods, analyze_composition.

### 3.7 `imaging_radiomics.py` (~694 lines)
Imaging (pathomics) and radiomics integration.

**Classes:**
- `TileEncoder(nn.Module)` -- placeholder for UNI/GigaPath. 256x256 tiles -> 2048d embeddings.
- `TransMILPooling(nn.Module)` -- Transformer MIL with Nystrom attention. Positional encoding, Q/K/V projections, landmark sampling, global average pooling -> patient embedding.
- `MorphologyFeatureExtractor(nn.Module)` -- encodes Bartl infiltration pattern (focal/diffuse/mixed), microvessel density, fibrosis grade (0-4).
- `RadiomicsFeatureEncoder(nn.Module)` -- PET/CT (MTV, TLG, SUVmax/mean, GLCM, GLRLM) + MRI (pattern, DCE wash-in/out/AUC).
- `HierarchicalCrossAttentionFusion(nn.Module)` -- 4-layer hierarchical fusion: L1 Genomics<->Proteomics, L2 +Metabolomics, L3 +Pathomics, L4 +Radiomics (asymmetric, learned weight ~0.7).
- `ImagingRadiomicsIntegrator(nn.Module)` -- top-level: tile encoding -> TransMIL -> pathomics + morphology fusion -> radiomics encoding -> hierarchical cross-attention with genomic/proteomic/metabolomic inputs.

### 3.8 `multi_site_harmonization.py` (~679 lines)
Multi-site data harmonization.

**Dataclasses:** `HarmonizationConfig`, `CorrectionsApplied`.

**Classes:**
- `ComBatSeqCorrector` -- empirical Bayes location-scale batch correction for count data. `fit(data, batch_labels)`, `transform(data, batch_labels)`.
- `HarmonyIntegrator` -- iterative soft-clustering + ridge regression. k-means++ init, E-step (soft assignment), M-step (ridge regression correction). `fit`, `transform`.
- `FederatedBatchNorm(nn.Module)` -- site-specific running mean/var with shared global weight/bias.
- `LoRAAdapter(nn.Module)` -- low-rank adaptation (A: in->r, B: r->out). scaling=1/rank.
- `CorrectionRouter` -- decision tree: counts->ComBat-seq; low severity->GSVA; medium->Harmony or LoRA; high->Harmony or LoRA by dataset size.
- `CrossSiteMonitor` -- tracks AUC variance across sites. Status: ok (<5%), warning (5-10%), alert (>10%).
- `MultiSiteHarmonizer` -- top-level orchestrator. Routes to appropriate corrector, applies, monitors.

---

## 4. `data/` (top-level, 3 files)

### 4.1 `data/clinical_labels.py` (~535 lines)
Clinical resistance label construction and augmentation.

**Classes:**
- `ClinicalResistanceLabelConstructor` -- 4 states: Sensitive (TTP>24mo), Primary_Refractory (no response by 3 cycles), Acquired_Resistant (initial response then relapse <12mo), Multi_Drug_Resistant (resistant to >=2 drug classes). `construct_labels(ttp, ttr, mrd_status, n_prior_lines, response_depth)`. `distinguish_intrinsic_acquired(labels, treatment_history)`.
- `SemiSupervisedAugmenter` -- `mixup_latent(z1, z2, y1, y2, alpha)` for latent-space MixUp. `pseudo_label(model, unlabeled_data, threshold)` for high-confidence pseudo-labeling.
- `AncestryAwareFeatureEngine` -- `stratified_normalization(features, ancestry_labels)` (z-score per ancestry group). `ancestry_covariate_correction(features, ancestry_pcs)` (regress out PCs).
- `DrugCoverageValidator` -- `validate_coverage(drug_ids, drug_response_matrix)` -> stats per drug. `flag_low_coverage_drugs(min_samples)`.

### 4.2 `data/graph_preprocessing.py` (~516 lines)
PPI graph construction with STRING integration.

**Class: `STRINGGraphBuilder`**
- `filter_edges(edge_index, edge_weights, threshold)` -> filtered tensors.
- `separate_edge_types(edge_sources)` -> categorized types (0=text-mined, 1=experimental, 2=database, 3=coexpression) from bit-packed sources.
- `build_graph(protein_list, string_scores, string_types)` -> dict with edge_index, edge_weights, edge_types, node_names.
- `graph_statistics(edge_index, n_nodes)` -> dict with n_edges, avg_degree, density, n_connected_components, largest_cc_size, avg_clustering_coeff. Uses Union-Find for components.

**Functions:**
- `load_string_edges(filepath, confidence_threshold, max_edges)` -> (scores_dict, types_dict).
- `create_synthetic_string_graph(n_proteins, edge_density, confidence_threshold)` -> (edge_index, edge_weights, edge_types).

### 4.3 `data/velocity_integration.py` (~478 lines)
RNA velocity, CytoTRACE, discrete resistance state definitions.

**Enum: `ResistanceState(IntEnum)`** -- SENSITIVE=0, PRIMARY_REFRACTORY=1, ACQUIRED_RESISTANT=2, MULTI_DRUG_RESISTANT=3.

**Dataclass: `StateCriteria`** -- has_t4_14, has_gain_1q, has_del_17p, tp53_mutated, cr_vgpr_response, relapsed, n_resistant_drug_classes.

**Classes:**
- `DiscreteResistanceStates` -- `classify_from_molecular(cytogenetics, tp53_status, treatment_response, n_resistant_drugs)` -> (state_labels, confidence_scores). `state_transition_matrix(labels_t1, labels_t2)` -> empirical transition probability matrix. `validate_state_definitions(labels, expression_data)` -> silhouette score + UMAP separability.
- `RNAVelocityIntegrator(nn.Module)` -- `compute_velocity_pseudotime(spliced, unspliced)` estimates beta/gamma, computes ds/dt = beta*u - gamma*s, projects onto first SVD component, integrates. `velocity_consistency_score(sde_trajectories, velocity_field)` -> cosine similarity [0,1]. `cytotrace_score(expression_matrix)` -> [0,1] where 1=stem-like/drug-resistant.

**Function:** `integrate_velocity_with_sde(spliced, unspliced, sde_trajectories)` -> dict with pseudotime, cytotrace, velocity_consistency.

---

## 5. `configs/ablation_configs.py` (~399 lines)
Programmatic ablation config generation.

**Registry: `ABLATION_REGISTRY`** -- 12 ablation conditions in 3 tags:
- **Component ablations** (6): no_identifiable_ode, no_causal_fusion, no_beta_tcvae, no_lyapunov, no_sparsity, no_pathway_anchoring.
- **Modality ablations** (4): no_transcriptomics, no_proteomics, no_epigenomics, no_ppi.
- **Scaling ablations** (3): data_10pct, data_25pct, data_50pct.

**Functions:**
- `_set_nested(d, dotted_key, value)`, `_get_nested(d, dotted_key, default)` -- dot-path dict access.
- `generate_ablation_config(base_config, ablation_name, seed)` -> modified config dict. Applies modifications, sets seed, adds metadata, updates output directory.
- `generate_all_ablation_configs(base_config, output_dir, seeds, conditions)` -> list of YAML file paths.
- `get_ablation_summary()` -> dict grouped by tag.
- `validate_ablation_config(config)` -> list of warnings.

CLI: `python configs/ablation_configs.py --base configs/full.yaml --output configs/ablations/ [--conditions ...] [--seeds ...] [--list]`.

---

## Appendix F: Clinical, Interpretability, Theory

## Part 6: Clinical, Interpretability, and Theory Modules

---

### 6.1 Clinical Module (`resistancemap/clinical/`)

#### `resistancemap/clinical/__init__.py`
Package init with lazy imports for all 8 clinical sub-modules plus utils. Uses `__getattr__` for deferred loading with error handling.

**`__all__`**: `treatment_conditioned`, `dynamic_treatment_regimes`, `relapse_prediction`, `early_relapse_detection`, `smm_progression`, `mrd_prediction`, `emd_prediction`, `immunotherapy_response`, `utils`.

---

#### `resistancemap/clinical/utils.py`
Shared utilities: treatment registries, feature normalization, genetic encoding, and survival analysis.

**Dataclass:**
- `MMRegimen` -- Metadata for an NCCN-approved MM treatment regimen. Fields: `name`, `mechanism` (list of mechanism strings), `line_of_therapy` (1=induction, 2+=relapsed), `is_maintenance`, `contraindications`, `typical_duration_months`, `iss_optimal_range`, `requires_monitoring`.

**Enum:**
- `MMTreatmentRegistry(Enum)` -- NCCN 2025 multiple myeloma regimens. Members:
  - Induction: `VRd`, `KRd`, `DRd`, `DKRd`
  - Maintenance: `VRd_maintenance`, `Rd_maintenance`
  - Relapsed/Refractory: `IRd`, `CYAD`, `PomDex`
  - CAR-T: `CART_preparation`
  - Novel (2025): `BsDex` (bispecific antibody)

**Class `ClinicalFeatureNormalizer`:**
- Constants: `ISS_STAGES`, `ECOG_SCORES`, `EGFR_NORMAL=90`, `EGFR_SEVERE=15`, `LDH_NORMAL=245`, `LDH_HIGH=800`, `M_PROTEIN_THRESHOLD=3.0`, `FLC_RATIO_NORMAL=(0.26, 1.65)`
- `normalize_iss_stage(iss_value)` -> float [0,1]: Maps ISS I/II/III to 0/0.5/1.
- `normalize_ecog(ecog_value)` -> float [0,1]: ECOG/4.
- `normalize_egfr(egfr_value)` -> float [0,1]: Clamps to [15,90], linear.
- `normalize_ldh(ldh_value)` -> float [0,1]: 0 if below normal, linear to 1 at 800.
- `normalize_m_protein(m_protein_value)` -> float [0,1]: Divided by 3.0 threshold.
- `normalize_flc_ratio(flc_ratio)` -> float [0,1]: Distance from normal range.
- `normalize_features(features: Dict)` -> Dict: Batch normalize all available features.

**Class `CytogeneticEncoder`:**
- `MARKERS`: `del_17p`, `t_4_14`, `t_14_16`, `t_14_20`, `gain_1q`, `del_1p` (6 adverse markers).
- `encode(cytogenetics: Dict[str,bool])` -> ndarray (6,): Binary vector encoding.
- `decode(encoding)` -> Dict: Reverse mapping.
- `get_high_risk_status(cytogenetics)` -> bool: True if any of del_17p, t_4_14, t_14_16, t_14_20.

**Class `TreatmentHistoryEncoder`:**
- `COMMON_AGENTS`: 12 agents (PIs, IMiDs, monoclonal antibodies, corticosteroids, alkylating agents).
- `encode_agents(agents_used, durations_months)` -> Dict: Returns one-hot vector, log-normalized duration vector, num_unique_agents, total_treatment_months.

**Class `SurvivalUtils`:**
- `kaplan_meier(time, event)` -> (unique_times, survival_prob, stderr): Kaplan-Meier product-limit estimator with Greenwood variance formula.
- `log_rank_test(time1, event1, time2, event2)` -> (chi2_stat, p_value): Two-sample log-rank test using hypergeometric variance, chi-squared p-value.

---

#### `resistancemap/clinical/treatment_conditioned.py`
Treatment-conditioned prediction and causal inference. Loss: `L = L_factual + lambda_prop * L_propensity + lambda_iptw * L_IPTW_weighted + lambda_causal * L_ATE_regularization`.

**Class `DragonNetHead(nn.Module)`:**
- Shared representation layer -> 3 heads: propensity P(T=1|X), outcome under treatment E[Y|T=1,X], outcome under control E[Y|T=0,X].
- `__init__(input_dim, hidden_dim=256, outcome_dim=1, dropout_rate=0.1)`
- `forward(x)` -> (representation, propensity, outcome_treated, outcome_control)

**Class `TreatmentConditionedODEFunc(nn.Module)`:**
- Neural ODE dynamics with FiLM conditioning: `dz/dt = gamma(u) * f(z) + beta(u)`.
- Uses `gamma_net` (Softplus activation for positivity) and `beta_net` for FiLM parameters.
- `forward(t, z, u)` -> dz/dt tensor.

**Class `IPTWEstimator(nn.Module)`:**
- Inverse Probability of Treatment Weighting with trimming.
- `__init__(trim_quantile=0.05, epsilon=1e-6)`
- `forward(propensity, treatment, treatment_marginal)` -> (weights_trimmed, propensity_score). Supports stabilized IPTW.
- `compute_stability_diagnostics(weights)` -> Dict with mean, std, min, max, effective sample size ratio.

**Class `CounterfactualCrossAttention(nn.Module)`:**
- Multi-head cross-attention for retrieving similar treatment-response pairs from historical data.
- Query: current patient state; Key/Value: historical patient database.
- `forward(query, memory_key, memory_value, memory_mask)` -> (attended, attention_weights)

**Class `TreatmentActorCritic(nn.Module)`:**
- Actor-Critic RL head. Action space: 0=stay, 1=escalate, 2=switch, 3=add CD38 antibody.
- Actor: softmax policy over 4 actions.
- Critic: expected PFS in months (0-60), sigmoid * 60.
- `forward(state)` -> (policy, value)

**Class `TreatmentConditionedPredictor(nn.Module)`:**
- Top-level orchestrator integrating DragonNet + Neural ODE + IPTW + CounterfactualCrossAttention + ActorCritic.
- `__init__(input_dim, n_ode_steps=10, repr_dim=256, treatment_dim=8, use_ode=True, lambda_propensity=0.1, lambda_iptw=0.05, lambda_causal=0.1)`
- `forward(x, treatment, treatment_sequence, recent_labs)` -> Dict with representation, propensity, outcomes (treated/control), IPTW weight, ODE trajectory, counterfactual outcome, policy, value.
- `_integrate_ode(repr_x, treatment_sequence)` -> trajectory using `odeint_adjoint` with linear interpolation of treatment sequences.
- `compute_loss(output, y_true, treatment)` -> (total_loss, loss_dict) with L_factual, L_propensity, L_IPTW_penalty, L_ATE_reg.

---

#### `resistancemap/clinical/dynamic_treatment_regimes.py`
Dynamic treatment regimes, Q-learning, A-learning (AIPW), and Counterfactual Recurrent Networks.

**Class `GradientReversalLayer(torch.autograd.Function)`:**
- Forward: identity. Backward: negate gradients by alpha.

**Function `gradient_reversal(x, alpha=1.0)`** -- Apply gradient reversal.

**Class `TreatmentFeasibilityMask`:**
- NCCN 2025 clinical constraint enforcement. Action space: 0=watch, 1=lenalidomide, 2=bortezomib, 3=carfilzomib, 4=daratumumab, 5=venetoclax, 6=CAR-T, 7=transplant.
- `get_feasible_actions(egfr, ecog, age, cart_used)` -> bool array (8,). Rules: CAR-T one-time; carfilzomib excluded if eGFR<30; ECOG>=3 excludes transplant+CAR-T; age>=75 excludes transplant.
- `mask_logits(logits, feasible_mask)` -> masked logits (infeasible set to -inf).

**Class `FittedQIteration(nn.Module)`:**
- Neural Q-function for DTR via backward recursion. Q_k(H_k, a_k) approximates expected outcome under optimal future actions.
- `__init__(history_dim, n_actions, hidden_dims=[256,128], dropout=0.1)`
- `forward(history, actions)` -> Q-values (batch, 1). Concatenates history with one-hot encoded actions.
- `compute_optimal_actions(history)` -> (optimal_actions, optimal_q_values): Enumerates all actions, picks argmax Q.
- `backward_recursion(histories, rewards, gamma=0.95, n_epochs=10)` -> metrics dict. K-stage backward recursion from final stage.

**Class `CounterfactualRecurrentNetwork(nn.Module)`:**
- Encoder-decoder GRU with gradient reversal for domain-adversarial balanced representations.
- Encoder: 2-layer GRU (history_dim + n_actions -> hidden_dim).
- Discriminator: predicts treatment from representation (reversed gradients).
- Decoder: per-action outcome prediction heads.
- `encode(treatment_history, actions)` -> balanced representation.
- `predict_outcomes(rep)` -> Dict[action_idx, outcome_tensor].
- `forward(treatment_history, actions)` -> (counterfactual_outcomes, treatment_logits).
- `train_step(treatment_history, actions, targets, alpha_adversarial=0.1)` -> metrics dict. Combined outcome + KL-divergence adversarial loss.

**Class `ALearnerDoublyRobust(ABC)`:**
- AIPW-based blip function estimation for optimal DTR.
- `compute_aipw_weights(actions, propensity)` -> weights (1/propensity, clipped to [1e-4, 1-1e-4]).
- `estimate_blip(histories, actions, outcomes, propensity_scores, mu_predictions)` -> Dict with blip_coefficients, blip_predictions, residual_variance, mean/max weight. Uses weighted least squares (or pseudoinverse fallback).

**Class `DTRPolicyOptimizer(nn.Module)`:**
- Top-level coordinator: FittedQIteration + CRN + ALearnerDoublyRobust + TreatmentFeasibilityMask + optional ResistanceMap sensitivity integration.
- `__init__(history_dim, n_actions, n_stages, resistance_map_fn=None, dropout=0.1, seed=42)`
- `augment_history(history, biomarker)` -> history augmented with P_sensitivity from ResistanceMap.
- `select_action(history, biomarker, egfr, ecog, age, cart_used)` -> (recommended_action, diagnostics_dict). Picks Q-optimal action if feasible, else best feasible fallback.
- `train_joint(histories, biomarkers, actions, outcomes, propensity_scores, n_epochs=10, alpha_crn=0.1)` -> metrics. Alternates Q-learning backward recursion and CRN adversarial training.

**Function `build_dtr_optimizer(config)`** -- Factory function returning `DTRPolicyOptimizer`.

---

#### `resistancemap/clinical/early_relapse_detection.py`
M-protein kinetics, early warning signals, CUSUM change detection, EWMA monitoring, cytokine fingerprinting, and comprehensive ER18 risk assessment.

**Dataclasses:**
- `MProteinKineticsParams` -- Fields: model_type ("mono_exponential" or "bi_exponential"), alpha_1, alpha_2, amplitude_a, amplitude_b, m_0, r_squared, fit_error.
- `EarlyWarningSignal` -- Fields: signal_name, signal_value, threshold, is_triggered, clinical_urgency ("low"/"moderate"/"high").
- `EarlyRelapseRiskAssessment` -- Fields: er18_risk_probability, risk_category, contributing_factors, kinetic_trajectory, recommended_action.

**Functions:**
- `_mono_exponential(t, m0, alpha)` -> M(t) = M_0 * exp(-alpha*t).
- `_bi_exponential(t, m0, alpha1, a, alpha2, b)` -> biphasic decay.

**Class `MProteinKineticsModel(nn.Module)`:**
- Mono/bi-exponential fitting of M-protein decay kinetics.
- `__init__(regimen_type="standard")`: Baseline decay constants per regimen (standard=0.08, intensive=0.12, maintenance=0.04 week^-1).
- `fit_kinetics(timepoints, m_protein_values, m_0, try_biexponential=True)` -> MProteinKineticsParams. Uses scipy `curve_fit`. Selects bi-exponential if R2 improves by >0.05.
- `forward(kinetics_params, timepoints)` -> predicted M-protein tensor.

**Class `EarlyWarningSignalDetector`:**
- Rule-based detection: M-protein reduction at week 4 (<25%), plateau detection weeks 8-12, rising LDH (>10%), mHR-IMS criteria.
- `detect_warnings(timepoints, m_protein, ldh_baseline, ldh_current, ecog, iss, calcium)` -> list of EarlyWarningSignal.
- `_compute_mhr_ims_score(ecog, iss, calcium)` -> int (0-3): ECOG==2 +1, ISS==3 +1, calcium>=11 +1.

**Class `KineticTrajectoryClassifier`:**
- Classifies trajectory using velocity/acceleration of M-protein curves.
- `classify_trajectory(timepoints, m_protein, flc_involved)` -> (trajectory_category, er18_risk_estimate). Categories: "unfavorable" (0.60 risk), "favorable" (0.15), "stable" (0.35).

**Class `CUSUMChangeDetector(nn.Module)`:**
- Two-sided CUSUM for sustained M-protein changes.
- `__init__(threshold=5.0, slack_param=0.5)`: Registers `cusum_pos`, `cusum_neg` buffers.
- `forward(log_m_protein, expected_log_m)` -> (cusum_pos, cusum_neg) tensors.
- `detect_change(cusum_values)` -> bool.

**Class `EWMAMonitor(nn.Module)`:**
- Exponentially weighted moving average for short-term trend tracking.
- `__init__(alpha=0.3)`. `forward(new_observation)` -> updated EWMA. `reset()`.

**Class `CytokineActivityFingerprint`:**
- Models microenvironment-driven resistance via IL-6, TNF-alpha, VEGF, HGF.
- Thresholds: IL6=5.0, TNFa=15.0, VEGF=300.0, HGF=500.0 pg/mL.
- `compute_resistance_phenotype(il6, tnfa, vegf, hgf)` -> (resistance_score, phenotype_label). Score = fraction of elevated cytokines.

**Class `EarlyRelapsePredictor(nn.Module)`:**
- Top-level integrator: kinetics + warnings + trajectory + CUSUM + EWMA + cytokine fingerprint.
- `predict_early_relapse(timepoints, m_protein, flc_involved, ldh_baseline, ldh_current, ecog, iss, calcium, il6, tnfa, vegf, hgf)` -> EarlyRelapseRiskAssessment. Aggregates base risk (0.25) + trajectory + warnings + cytokines, clipped to [0, 0.95]. Risk categories: <0.30="low", 0.30-0.55="intermediate", >0.55="high". Recommends clinical action per tier.

**Function `setup_logging(level)`** -- Configures logging for the module.

---

#### `resistancemap/clinical/emd_prediction.py`
5-layer extramedullary disease (EMD) risk prediction: transcriptomics, radiomics, microenvironment, protein pathways, clinical features.

**Class `TranscriptomicEMDEncoder(nn.Module)`:**
- Encodes adhesion loss (CD56/NCAM1, VLA-4/ITGA4), migration (CXCR4/CXCL12), immune escape (PD-L1, VISTA).
- Input: 6 gene expression features -> 64-dim embedding via 3 sub-networks + fusion.

**Class `RadiomicsEncoder(nn.Module)`:**
- PET radiomics (entropy, uniformity), MRI diffusion (ADC, MD), DCE perfusion (Ktrans, vp), texture (GLCM, GLRLM).
- Input: 8 features -> 64-dim embedding.

**Class `MicroenvironmentEncoder(nn.Module)`:**
- Cytokine panel: pro-migration (THBS1, PPBP) vs anti-migration (EGF, BDNF).
- Input: 4 cytokines -> 64-dim embedding.

**Class `ProteinPathwayEncoder(nn.Module)`:**
- Focal adhesion (FAK, paxillin), PI3K/AKT, MAPK/ERK, Wnt/beta-catenin.
- Input: 8 pathway activity scores -> 64-dim embedding.

**Class `ClinicalRiskEncoder(nn.Module)`:**
- Lab values (LDH, calcium), cell burden (CCC, BM infiltration), cytogenetics (del17p, t414, amp1q), treatment history.
- Input: 9 clinical features -> 64-dim embedding.

**Class `EMDSurvivalHead(nn.Module)`:**
- 3 prediction heads from fused 320-dim embedding:
  - `emd_risk_head` -> P(EMD development) (sigmoid)
  - `survival_head` -> Weibull [log_k, log_lambda] parameters
  - `cns_head` -> P(CNS-MM) (sigmoid, OS <6 months flag)

**Class `EMDRiskPredictor(nn.Module)`:**
- Top-level: 5 encoders -> MultiheadAttention fusion (4 heads) -> projection -> EMDSurvivalHead.
- `forward(transcriptomics, radiomics, cytokines, pathways, clinical)` -> Dict with emd_probability, time_to_emd_k, time_to_emd_lambda, cns_probability, layer_embeddings, fused_embedding.
- `compute_layer_attribution(layer_embeddings)` -> Dict of L2-norm attribution scores per layer.
- `get_risk_summary(outputs)` -> Dict with risk_category (High/Intermediate/Low), median_time_to_emd (from Weibull: lambda * ln(2)^(1/k)), cns_high_risk flag.

**Function `create_emd_predictor(embedding_dim=64)`** -- Factory function; logs parameter count.

---

#### `resistancemap/clinical/immunotherapy_response.py`
Novel immunotherapy response prediction with MAML few-shot learning, antigen loss forecasting, and sequential therapy planning.

**Dataclasses:**
- `PatientImmuneProfile` -- Fields: antigen_bcma, antigen_gprc5d, antigen_fcrl5, exhaustion_score, cd4_cd8_ratio, tcr_clonality, batch.
- `AgentMechanism` -- Fields: primary_target, mechanism_type, mechanism_idx.

**Class `PatientImmuneProfiler(nn.Module)`:**
- Encodes antigen expression (BCMA, GPRC5D, FCRL5) and T-cell fitness (exhaustion, CD4/CD8, TCR clonality).
- Antigen encoder + T-cell encoder -> fusion -> hidden_dim embedding.

**Class `AgentMechanismEncoder(nn.Module)`:**
- MECHANISM_TYPES: "bispecific", "car_t", "celmod", "adc".
- Encodes target antigen expression + mechanism type embedding + 1-hop STRING pathway neighbor expression.
- Target encoder + mechanism embedding + neighbor encoder -> fusion -> hidden_dim.

**Class `MAMLFewShotAdapter(nn.Module)`:**
- MAML for few-shot learning on novel agents (n=5-20 responder samples).
- `__init__(input_dim=128, hidden_dim=64, inner_lr=0.01, n_inner_steps=5)`
- `inner_loop(support_embeddings, support_responses)` -> adapted_params dict. SGD inner loop with BCE loss.
- `outer_loop(query_embeddings, query_responses, adapted_params)` -> outer loss.
- `forward(support_embeddings, support_responses, query_embeddings)` -> predictions on query set.

**Class `AntigenLossForecaster(nn.Module)`:**
- LSTM-based forecasting of BCMA/GPRC5D/FCRL5 expression trajectory.
- 2-layer LSTM encoder -> decoder (n_forecast_steps * 3) + loss_risk_head.
- `forward(antigen_history: (batch, seq_len, 3))` -> (forecast: (batch, n_steps, 3), loss_risk: (batch,)).

**Class `SequentialImmunotherapyPlanner(nn.Module)`:**
- Plans BCMA -> GPRC5D -> CELMoD transitions.
- `forward(patient_embedding, agent_embedding, n_steps=3)` -> (sequence: List[int], efficacy_scores: Tensor).

**Class `ImmunotherapyResponsePredictor(nn.Module)`:**
- Top-level orchestrator: PatientImmuneProfiler + AgentMechanismEncoder + MAMLFewShotAdapter + AntigenLossForecaster + SequentialImmunotherapyPlanner + CRS/ICANS risk head.
- `predict_response(patient_profile, mechanism, target_expression, pathway_neighbors)` -> Dict with response_probability.
- `predict_with_few_shot(patient_profile, mechanism, target_expression, support_profiles, support_responses, pathway_neighbors)` -> Dict with adapted_response_probability.
- `predict_resistance_trajectory(antigen_history)` -> Dict with antigen_forecast, loss_risk.
- `plan_sequential_therapy(patient_profile, initial_mechanism, initial_target, n_steps=3)` -> Dict with planned_sequence, efficacy_predictions.
- `assess_crs_icans_risk(patient_profile, mechanism, target_expression)` -> Dict with crs_risk, icans_risk.

---

#### `resistancemap/clinical/mrd_prediction.py`
MRD negativity prediction: multi-modal encoders, trajectory prediction, multi-timepoint/multi-depth probability regression.

**NamedTuple `MRDPrediction`:** mrd_prob_10e5 (batch,3), mrd_prob_10e6 (batch,3), latent_state, stability_score, treatment_modulation.

**Class `GEPEncoder(nn.Module)`:**
- Variational bottleneck for gene expression profiles (2000 genes -> 128 latent).
- `encode(x)` -> (mu, logvar). `reparameterize(mu, logvar)` -> z. `decode(z)` -> reconstruction.
- `forward(x)` -> (latent_z, reconstruction, mu, logvar).

**Class `ClinicalEncoder(nn.Module)`:**
- 64 clinical features -> 128-dim embedding via 3-layer MLP with BatchNorm.

**Class `ImmuneEncoder(nn.Module)`:**
- 48 immune markers (T-cell subsets, NK) -> 128-dim embedding via 3-layer MLP with BatchNorm.

**Class `LatentStabilityEstimator(nn.Module)`:**
- Estimates epigenetic stability from VAE latent + reconstruction variance.
- Input: latent_z (128) + mean_var (1) -> stability score in [0,1] via sigmoid.
- High reconstruction variance -> labile/plastic state -> high P(MRD-).

**Class `TrajectoryTransformer(nn.Module)`:**
- Transformer-based trajectory prediction for latent state evolution (8 timesteps).
- Sinusoidal positional encoding. 2-layer TransformerEncoder with 4 heads.
- `forward(initial_latent)` -> trajectory (batch, num_timesteps, latent_dim).

**Class `MRDProbabilityRegressor(nn.Module)`:**
- Flattened trajectory -> shared feature extractor -> two heads:
  - `head_10e5`: P(MRD-) at 10^-5 sensitivity at months 2, 6, 12.
  - `head_10e6`: P(MRD-) at 10^-6 sensitivity at months 2, 6, 12.

**Class `CrossAttentionFusion(nn.Module)`:**
- GEP queries clinical and immune via MultiheadAttention -> fused embedding.
- `forward(gep_embed, clinical_embed, immune_embed)` -> fused (batch, embed_dim).

**Class `MRDNegPredictor(nn.Module)`:**
- Full pipeline: GEPEncoder + ClinicalEncoder + ImmuneEncoder -> CrossAttentionFusion -> LatentStabilityEstimator -> TrajectoryTransformer -> MRDProbabilityRegressor.
- `forward(gene_expr, clinical_feats, immune_markers)` -> MRDPrediction. Treatment modulation = 1 - stability (higher stability -> lower intensity needed).
- `get_reconstruction_loss(gene_expr, gep_recon, gep_mu, gep_logvar)` -> (recon_loss, kl_divergence).
- `get_latent_disentanglement_loss(gep_logvar, target_variance=0.1)` -> disentanglement loss (variance + balance penalties).

---

#### `resistancemap/clinical/relapse_prediction.py`
Neural ODE-based competing risks model with cause-specific hazard decoders.

Multi-state model: Post-ASCT -> Biochemical relapse / Clinical relapse / EMD/CNS / Death.

**Class `CauseSpecificHazardDecoder(nn.Module)`:**
- Pathway-focused MLP for a single hazard rate. Pathways: early_relapse_TP53, late_relapse_DNA_repair, emd_migration, cns_bbb_penetration, non_relapse_mortality.
- Xavier initialization. Output: exp(log_hazard) for positivity.

**Class `CompetingRisksODE(nn.Module)`:**
- Shared latent ODE -> 5 cause-specific hazard decoders.
- `__init__(input_dim, latent_dim, num_causes=5, hidden_dims)`. Encoder: biomarkers -> latent state. ODE dynamics: shared latent trajectory.
- `forward(biomarkers, times, method="dopri")` -> (hazards: (batch, 5, T), latent_traj, latent_z0). Uses `torchdiffeq.odeint`.

**Class `CumulativeIncidenceFunction(nn.Module)`:**
- CIF_k(t) = integral_0^t S(u) * lambda_k(u) du. Trapezoid rule integration.
- Enforces CIF constraint: sum_k CIF_k <= 1 via proportional scaling.
- `forward(hazards, times)` -> (cif, surv_prob, cum_hazards).

**Class `LandmarkRecalibrator(nn.Module)`:**
- Recalibrates latent state at landmark times (100d, 6mo, 12mo, 24mo) with fresh biomarker measurements (M-protein, FLC).
- `forward(latent_state, new_biomarkers, current_time)` -> updated latent state.

**Class `CompetingRisksLoss(nn.Module)`:**
- DeepHit-style: NLL + ranking loss (concordance) + CIF constraint regularization.
- `__init__(alpha=0.5, beta=1.0)`.
- `forward(cif, observed_times, observed_causes, uncensored)` -> (loss, metrics). NLL: event=-log(CIF), censored=-log(1-CIF_sum). Ranking loss: pairwise concordance.

**Class `RelapseSpecificPredictor(nn.Module)`:**
- Top-level: CompetingRisksODE + CIF + LandmarkRecalibrator + CompetingRisksLoss.
- Default times buffer: [0, 50, 100, 180, 365, 730, 1095] days post-ASCT.
- `forward(biomarkers, times)` -> (cif, surv_prob, state_dict).
- `compute_loss(biomarkers, observed_times, observed_causes, uncensored, times)` -> (loss, metrics).
- `predict_at_landmark(biomarkers, landmark_updates, landmark_time=100.0)` -> recalibrated CIF. Re-integrates ODE from recalibrated latent state.

---

#### `resistancemap/clinical/smm_progression.py`
SMM-to-MM progression prediction: PANGEA-style scoring, clonal expansion tracking, immune decay estimation, Weibull mixture cure survival.

**Class `PANGEARiskScorer(nn.Module)`:**
- Static risk model: M-protein, BMPC, FLC ratio, genomic drivers (t(4;14), del(17p), gain(1q), MYC rearrangement).
- Learnable feature scaling parameters.
- `forward(m_protein, bmpc, flc_ratio, t414, del17p, gain1q, myc_rearr)` -> 2-year progression probability.

**Class `ClonalExpansionTracker(nn.Module)`:**
- Exponential VAF growth model: VAF(t) = VAF_0 * exp(growth_rate * t).
- `estimate_growth_rate(timepoints, vafs)` -> (growth_rate, R2). Log-linear regression.
- `forward(timepoints, vafs)` -> (growth_rates, confidences) per batch sample.

**Class `ImmuneDecayEstimator(nn.Module)`:**
- Tracks TCR diversity decline, NK CD38+ decline, Treg increase, CD8+ exhaustion.
- `estimate_immune_score(tcr_diversity, nk_cd38_freq, treg_freq, cd8_exhaustion)` -> float [0,1]. Weighted combination (0.3 TCR + 0.3 NK + 0.2 inverse-Treg + 0.2 inverse-exhaustion).
- `forward(tcr_diversity, nk_cd38_freq, treg_freq, cd8_exhaustion)` -> (immune_scores, decay_rates).

**Class `MixtureCureSurvival(nn.Module)`:**
- S(t) = pi + (1 - pi) * exp(-(lambda*t)^k). Learnable log_lambda, log_k.
- `forward(times, cure_fraction)` -> survival probability.
- `predict_time_to_progression(risk_score, cure_fraction, quantile=0.5)` -> time-to-progression in years.

**Class `SMMProgressionPredictor(nn.Module)`:**
- Top-level: PANGEARiskScorer + ClonalExpansionTracker + ImmuneDecayEstimator + MixtureCureSurvival + fusion network.
- `forward(...)` -> Dict with progression_prob, risk_tier (1/2/3), time_to_progression, cure_fraction, static_risk, fusion_risk, clonal_growth_rate, immune_score.
- Risk tiers: <6%=1, 6-18%=2 (conservative), >44%=3 (PANGEA).
- Ensemble: 0.6 * static + 0.4 * fusion.

**Function `create_example_batch(batch_size=4)`** -- Creates example inputs for testing.

---

### 6.2 Interpretability Module (`resistancemap/interpretability/`)

#### `resistancemap/interpretability/__init__.py`
Comprehensive public API re-exporting all classes from 4 sub-modules + visualization. Defines `explain_patient()` convenience function.

**Function `explain_patient(model, patient_data, config)`** -> Dict:
Full interpretability pipeline for a single patient:
1. Pathway attribution (Temporal Integrated Gradients)
2. Counterfactual trajectory generation (if drug_history provided)
3. Critical window detection
4. Trajectory decomposition into biological programs
5. Phase transition analysis
6. Uncertainty decomposition
7. Publication-quality figure generation (temporal heatmap, patient journey)
8. Human-readable summary string

Required model interface: `model.ode_func(t, x)`, `model.solve(x0, t_eval)`, `model.predict(x)`.

Required patient_data keys: `x0`, `t_span`, `gene_names`, optional `program_names`, `pathway_db`, `drug_history`, `data_mask`.

---

#### `resistancemap/interpretability/pathway_attribution.py`
Temporal Integrated Gradients adapted for ODE models + pathway aggregation + gene interaction recovery.

**Dataclasses:**
- `PathwayAttributionConfig` -- n_ig_steps=100, baseline_mode="zero"/"mean"/"random", n_random_baselines=10, n_permutations=1000, significance_threshold=0.05, min_pathway_size=10, max_pathway_size=500, interaction_threshold=0.01, temporal_resolution=50, device.
- `PathwayAttributionResult` -- gene_attributions (T,G), pathway_scores Dict, pathway_pvalues Dict, significant_pathways List, temporal_profile (T,), metadata.
- `InteractionResult` -- learned_interactions List[(gene_i, gene_j, weight)], precision_vs_known Dict, recall_vs_known Dict, novel_interactions List, interaction_matrix (G,G).

**Class `PathwayAttributor`:**
- Temporal Integrated Gradients (TIG): `TIG_i = integral_{t0}^{t1} (df/dx_i(t)) * (dx_i/dt) dt`.
- `_get_baseline(x)` -> zero/mean/random baseline tensor.
- `compute_temporal_integrated_gradients(x0, t_span, target_fn)` -> (T_interp, n_genes) tensor. Computes gradient * velocity at each intermediate timepoint, trapezoidal integration.
- `compute_standard_integrated_gradients(x, t_span, target_fn)` -> (n_genes,) tensor. Sundararajan et al. (2017) straight-line interpolation for comparison.
- `aggregate_to_pathways(gene_attributions)` -> Dict[pathway_name, score array]. Mean absolute attribution * sqrt(pathway_size).
- `compute_pathway_significance(gene_attributions, pathway_scores)` -> Dict[pathway_name, FDR-corrected p-value]. Permutation test + Benjamini-Hochberg.
- `attribute(x0, t_span, target_fn)` -> PathwayAttributionResult. Full pipeline.
- `_benjamini_hochberg(pvalues)` -- Static method for FDR correction.

**Class `GeneInteractionExtractor`:**
- Extracts interaction matrix A from model (searches model.A, model.ode_func.A, model.ode_func.linear.weight).
- `extract_interaction_matrix()` -> (G,G) numpy array.
- `extract_interactions(threshold)` -> InteractionResult. Compares to TRRUST, RegNetwork, STRING databases; computes precision/recall; identifies novel interactions.
- `compute_interaction_significance(n_permutations=1000)` -> Dict[(gene_i, gene_j), p-value]. Permutation of gene labels.
- `get_top_regulators(k=20)` -> List[(gene_name, outgoing_weight)].
- `get_top_targets(k=20)` -> List[(gene_name, incoming_weight)].

**Function `load_pathway_database(source="hallmark", gmt_path)`** -> Dict[pathway_name, List[gene_symbol]]:
Supports "hallmark" (6 built-in sets: APOPTOSIS, PI3K_AKT_MTOR, NF_KB, DNA_REPAIR, DRUG_METABOLISM, STEMNESS), "kegg" (ABC_TRANSPORTERS, PROTEASOME), or custom GMT files.

**Function `_parse_gmt(path)`** -- Parses GMT tab-separated format.

---

#### `resistancemap/interpretability/counterfactual_engine.py`
Counterfactual trajectory generation and causal effect estimation.

**DEPRECATION NOTICE**: `WhatIfAnalyzer` and `CausalEffectEstimator.do_intervention` use latent-clamping which is associational (not interventional per Pearl's do-calculus). Phase 8 will introduce true do-calculus via `CounterfactualSimulator`.

**Dataclasses:**
- `CounterfactualConfig` -- n_trajectory_samples=100, ode_solver_steps=200, min/max_intervention_strength, dose_search_resolution=50, time_search_resolution=50, convergence_threshold=0.1, combination_max_drugs=3, device.
- `CounterfactualResult` -- factual_trajectory, counterfactual_trajectories Dict, factual_outcome, counterfactual_outcomes Dict, intervention_details, minimal_intervention, critical_window.
- `CausalEffectResult` -- ate, cate Dict, ite ndarray, confidence_intervals Dict, subgroup_labels.

**Class `CounterfactualTrajectoryGenerator`:**
- Modifies ODE dynamics for drug interventions: `dx/dt = f(x,t) + g(x,drug) * I(t in [t_start, t_end])`.
- `simulate_trajectory(x0, t_span, drug_id, drug_dose, drug_start_time, drug_end_time)` -> trajectory tensor.
- `generate_counterfactuals(x0, t_span, factual_drug, factual_dose, alternative_drugs, drug_names)` -> CounterfactualResult.
- `find_minimal_intervention(x0, t_span, drug_id, target_outcome=0.0)` -> Dict. Binary search over dose levels for minimum effective dose ("resistance reversal dose").
- `find_optimal_intervention_time(x0, t_span, drug_id, drug_dose, target_outcome=0.0)` -> Dict. Finds latest effective time answering "how long can we wait?"
- `_build_drug_effect(drug_id, dose)` -> Callable. Uses drug_encoder if available, else model.drug_effect, else simple -dose*0.1*x fallback.
- `_integrate_with_intervention(x0, t_eval, drug_effect_fn, t_start, t_end)` -> Euler integration with drug effect.

**Class `WhatIfAnalyzer`:**
- **ASSOCIATIONAL ONLY** (emits DeprecationWarning on init and method calls).
- `inhibit_pathway(x0, t_span, pathway_name, inhibition_strength=1.0)` -> Dict. Clamps pathway dimension at each integration step. Returns inhibited/baseline trajectories and outcome change.
- `shift_treatment_timing(x0, t_span, drug_id, drug_dose, original_start, time_shifts)` -> Dict of scenarios.
- `screen_drug_combinations(x0, t_span, drug_ids, drug_doses, drug_names)` -> Dict. Bliss independence synergy scoring: synergy = E_AB - (1 - (1-E_A)(1-E_B)). Returns individual_outcomes, combination_outcomes, synergy_scores, best_combination.
- `_simulate_combination(x0, t_span, drug_ids, drug_doses)` -> tensor. Sums drug effects at each step.

**Class `CausalEffectEstimator`:**
- `estimate_ate(patient_states, t_span, treatment_drug, treatment_dose, control_drug, control_dose)` -> Dict. ATE = E[Y(treatment)] - E[Y(control)] with 1000-bootstrap 95% CI.
- `estimate_cate(patient_states, t_span, treatment_drug, treatment_dose, subgroup_labels, ...)` -> CausalEffectResult. Per-subgroup CATE with bootstrap CIs.
- `do_intervention(x0, t_span, layer_name, intervention_values)` -> trajectory. **DEPRECATED** -- uses forward hook to clamp layer output (associational, not do-calculus).

---

#### `resistancemap/interpretability/temporal_explanations.py`
Geometric characterization of learned latent ODE: critical windows, trajectory decomposition, phase transitions.

**IMPORTANT CAUSAL SCOPE**: All quantities are properties of the learned model U_theta (associative/Pearl L1), NOT causal estimates. Bifurcations are mathematical properties of the trained function, not identified biological switches. Critical windows are Jacobian sensitivities, not validated intervention timing guidance.

**Dataclasses:**
- `TemporalConfig` -- n_timepoints=100, perturbation_eps=1e-4, eigenvalue_threshold=0.01, sensitivity_n_perturbations=50, critical_slowing_window=10, device.
- `CriticalWindowResult` -- time_sensitivity (T,), critical_windows List[(t_start, t_end)], tipping_points List[float], sensitivity_by_program (T,P), clinical_annotations.
- `TrajectoryDecomposition` -- timepoints, trajectory, velocity, program_contributions (T,P), driver_program (T,), handoff_times, program_names.
- `PhaseTransitionResult` -- eigenvalues (T,D) complex, bifurcation_times, order_parameter (idx, name), critical_slowing_down (T,), jacobian_sequence, basin_classification.

**Class `CriticalWindowDetector`:**
- Sensitivity S(t) = d(model_output)/d(z(t)) via perturbation analysis.
- `detect_critical_windows(x0, t_span, perturbation_direction, sensitivity_threshold=0.5)` -> CriticalWindowResult. For each timepoint: perturb state, re-integrate to final, measure outcome change. Per-program sensitivity via individual dimension perturbation. Identifies contiguous high-sensitivity windows and tipping points (local maxima).
- `_extract_contiguous_windows(times, mask)` -> List of (start, end) tuples.

**Class `TrajectoryDecomposer`:**
- Decomposes velocity dx/dt into per-program contributions via L2 norm of program subspace velocity.
- `__init__(model, program_names, program_dims, config)`: If program_dims not given, assumes equal division of state space.
- `decompose(x0, t_span)` -> TrajectoryDecomposition. Identifies driver program (argmax contribution) at each time and handoff points.
- `compute_program_acceleration(decomposition)` -> (T, P) array of d(contribution)/dt.

**Class `PhaseTransitionAnalyzer`:**
- Detects bifurcations via Jacobian eigenvalue analysis along trajectory.
- `analyze(x0, t_span)` -> PhaseTransitionResult:
  1. Solve trajectory.
  2. Compute Jacobian J(t) via finite differences at each timepoint.
  3. Eigenvalue decomposition; sort by real part descending.
  4. Detect eigenvalue sign crossings (linear interpolation for precise timing).
  5. Identify order parameter (program with largest crossing magnitude).
  6. Critical slowing down: lag-1 autocorrelation of trajectory increments in rolling window (Scheffer et al., 2009).
  7. Basin classification: negative leading eigenvalue="sensitive", near-zero="transitional", positive="resistant".
- `find_fixed_points(x_range, n_samples=100, max_iter=1000, tol=1e-6)` -> List[Dict]. Gradient descent toward f(x)=0; classifies stability via Jacobian eigenvalues.

---

#### `resistancemap/interpretability/uncertainty_explanations.py`
Uncertainty decomposition, conformal calibration, and uncertainty visualization.

**Dataclasses:**
- `UncertaintyConfig` -- n_mc_samples=50, n_ensemble_members=5, sde_n_samples=100, ode_tolerances=[1e-3..1e-6], conformal_alpha=0.1, abstention_threshold=0.5, calibration_bins=15, device.
- `UncertaintyDecomposition` -- total_uncertainty, aleatoric, epistemic, ode_solver, data_missing, per_timepoint (T,), per_program (P,), metadata.
- `CalibrationResult` -- prediction_intervals (N,2), coverage, interval_width, ece, reliability_diagram (bins,2), abstention_mask (N,).

**Class `UncertaintyDecomposer`:**
- Decomposes: Var[Y] = aleatoric + epistemic + ode_solver + data_missing.
- `decompose(x0, t_span, data_mask, program_names)` -> UncertaintyDecomposition.
- `_estimate_epistemic(x0, t_span)` -> (variance, predictions). MC Dropout or Deep Ensemble variance.
- `_estimate_aleatoric(x0, t_span)` -> float. From SDE diffusion, heteroscedastic noise output, or 0.01 fallback.
- `_estimate_sde_aleatoric(x0, t_span)` -> float. Variance of SDE trajectory samples.
- `_estimate_ode_solver_uncertainty(x0, t_span)` -> float. Variance across different tolerance levels (Richardson extrapolation principle).
- `_estimate_data_uncertainty(x0, data_mask)` -> float. Missing fraction * 0.1 heuristic.
- `_compute_per_timepoint_uncertainty(x0, t_span)` -> (T,) array via MC dropout.
- `_compute_per_program_uncertainty(x0, t_span, n_programs)` -> (P,) array via MC dropout on program subspaces.
- `_enable_dropout(model)` -- Sets Dropout layers to train mode for MC Dropout.
- `_solve_sde(x0, t_span)` -> Euler-Maruyama SDE integration.

**Class `ConfidenceCalibrator`:**
- Split conformal prediction (Vovk et al., 2005) + Romano et al. (2019) adaptive intervals.
- `calibrate(cal_inputs, cal_targets, t_span)`: Computes nonconformity scores, stores (1-alpha)-quantile.
- `predict_with_intervals(x0, t_span)` -> (prediction, (lower, upper), should_abstain). Abstains if interval width > threshold.
- `batch_predict_with_intervals(inputs, t_span)` -> CalibrationResult.
- `selective_predict(x0, t_span, risk_level)` -> Dict with prediction, interval, confidence ("high"/"medium"/"low"), explanation level ("concise"/"standard"/"detailed"), clinical recommendation text.
- `_compute_calibration_metrics(predictions)` -> (ECE, reliability_diagram).

**Class `UncertaintyVisualizer`:**
- Publication-quality plots (Nature style: Arial 7pt, 300 DPI).
- `plot_trajectory_fan(timepoints, trajectory_samples, outcome_dim, ...)`: Median + 50%/95% CI fan plot.
- `plot_uncertainty_decomposition(decomposition, ...)`: Horizontal bar chart of 4 uncertainty sources.
- `plot_per_timepoint_uncertainty(timepoints, uncertainty, ...)`: Line plot with fill.

---

#### `resistancemap/interpretability/visualization.py`
Publication-quality figures for ICML/Nature/Science.

**Constants:**
- `NATURE_STYLE`: Dict of matplotlib rcParams (Arial 7pt, 0.5pt axes, 300 DPI).
- `PALETTE`: Colorblind-friendly colors (Wong 2011): blue, orange, green, red, purple, yellow, cyan, black, gray.
- `PROGRAM_COLORS`: List of 8 colors from palette.

**Function `apply_nature_style()`** -- Applies NATURE_STYLE to matplotlib.

**Class `WaddingtonLandscapePlot`:**
- 3D surface plot of learned potential landscape with patient trajectory overlays.
- `plot(potential_grid, x_range, y_range, trajectories, trajectory_labels, attractors, saddle_points, ...)` -> Figure. Surface colored by potential, trajectories interpolated onto surface via RegularGridInterpolator.

**Class `PathwayCircosPlot`:**
- Circos-style pathway interaction diagram.
- `plot(pathway_names, interaction_matrix, known_mask, threshold, ...)` -> Figure. Pathways as colored nodes on a circle; chords show interactions (red=activating, blue=inhibiting; solid=known, dashed=novel). Bezier curve paths.

**Class `CounterfactualTrajectoryPlot`:**
- Side-by-side observed vs counterfactual trajectories.
- `plot(timepoints, factual, counterfactuals, intervention_window, outcome_threshold, ...)` -> Figure. Left panel: observed + intervention window shading. Right panel: counterfactuals with divergence region shading.

**Class `TemporalAttributionHeatmap`:**
- Time x Pathway heatmap with clinical event annotations.
- `plot(timepoints, pathway_names, attributions, clinical_events, cmap="RdBu_r", ...)` -> Figure. Auto-transposes to (P,T). Diverging colormap centered at 0.

**Class `PatientJourneyPlot`:**
- Comprehensive 3-row + optional right-column layout:
  - Top: Latent space trajectory (scatter colored by time).
  - Middle: Stacked area of biological program contributions.
  - Bottom: Drug intervention timeline (horizontal bars).
  - Right (optional): Counterfactual outcomes bar chart.
- `plot(timepoints, trajectory, program_contributions, program_names, drug_history, counterfactual_outcomes, patient_id, ...)` -> Figure.

---

### 6.3 Theory Module (`resistancemap/theory/`)

#### `resistancemap/theory/__init__.py`
Re-exports from two sub-modules:
- From `identifiability_proof`: `IdentifiabilityConfig`, `FisherInformationAnalyzer`, `StructuralIdentifiabilityTest`, `EmpiricalConvergenceTest`, `SensitivityAnalyzer`, `run_full_identifiability_analysis`.
- From `stability_analysis`: `StabilityConfig`, `BifurcationAnalyzer`, `BasinOfAttraction`, `LyapunovExponentComputer`, `PhasePortraitGenerator`, `FixedPointFinder`.

---

#### `resistancemap/theory/identifiability_proof.py`
Numerical verification of identifiability for structured biological ODE: `dx/dt = A @ sigma(x) + b(x)`.

**Dataclass `IdentifiabilityConfig`:**
- n_programs=8, n_time_points=20, n_trajectories=50, fim_noise_std=0.1, n_random_inits=10, convergence_tol=0.05, sensitivity_epsilon=1e-4, max_train_steps=500, lr=0.01.

**Class `FisherInformationAnalyzer`:**
- FIM: F_ij = (1/sigma^2) sum_t (dy/dtheta_i) (dy/dtheta_j). Full rank => locally identifiable (Rothenberg 1971).
- `compute_trajectory_sensitivity(ode_func, x0, t_span, params)` -> (T*D, P) Jacobian via autograd.
- `compute_fim(ode_func, initial_conditions, t_span, params)` -> (P,P) FIM averaged over N trajectories.
- `analyze_fim(F)` -> Dict with rank, n_params, is_identifiable, eigenvalues, condition_number, sloppy_directions (eigenvectors for small eigenvalues), n_sloppy (eigenvalues < 1% of max).

**Class `StructuralIdentifiabilityTest`:**
- Observability rank condition via Lie derivatives (Ljung & Glad, 1994).
- `compute_observability_matrix(A, x, n_derivatives=3)` -> (n_deriv*K, K). O_0 = I, O_1 = A * diag(sigma'(x)), O_k = iterated Jacobian composition.
- `check_identifiability(A, n_test_points=20, n_derivatives=4)` -> Dict. Checks SVD rank at random evaluation points. Identifiable if majority achieve full rank K.
- `verify_diagonal_dominance_sufficiency(n_trials=100)` -> Dict with DD vs non-DD identifiability rates.

**Class `EmpiricalConvergenceTest`:**
- Tests whether multiple random initializations converge to same A (up to permutation).
- `generate_synthetic_data(A_true, n_trajectories, t_span, noise_std=0.05)` -> (x0s, trajectories). dx/dt = A @ tanh(x), with observation noise.
- `permutation_aligned_distance(A1, A2)` -> (distance, permutation). Greedy row matching + Frobenius norm.
- `run_convergence_test(A_true, n_inits)` -> Dict with converged bool, pairwise_distances, mean/max distance, learned_matrices, training_losses. Trains via Euler integration + MSE loss.

**Class `SensitivityAnalyzer`:**
- S_ij = ||dy/dA_ij||_2 / ||y||_2 via finite differences.
- `compute_parameter_sensitivity(A, x0, t_span, epsilon)` -> (K,K) sensitivity matrix.
- `identify_critical_interactions(S, top_k=10)` -> List[(i, j, sensitivity)] sorted descending.
- `pruning_recommendation(S, threshold_quantile=0.25)` -> (K,K) binary mask. Always keeps diagonal (self-regulation).

**Function `run_full_identifiability_analysis(A_true, config, verbose=True)`** -> Dict:
Runs all 3 analyses in sequence: structural identifiability, sensitivity analysis, empirical convergence. Returns combined results dict.

---

#### `resistancemap/theory/stability_analysis.py`
Formal dynamical systems analysis: bifurcations, basins of attraction, Lyapunov exponents, phase portraits.

**Dataclass `StabilityConfig`:**
- state_dim=8, grid_resolution=50, integration_time=20.0, dt=0.01, lyapunov_transient=10.0, lyapunov_compute_time=50.0, n_lyapunov_steps=100, bifurcation_n_points=100, fixed_point_tol=1e-5, fixed_point_max_iter=1000, state_range=3.0.

**Class `FixedPointFinder`:**
- Newton's method (L-BFGS on ||f(x)||^2) from random initializations.
- `find_fixed_points(dynamics_fn, n_inits=100, merge_threshold=0.1)` -> (fixed_points (M,D), stability (M,)). Merges nearby FPs. Stability: 1=stable, -1=saddle, 0=unstable.
- `_classify_stability(dynamics_fn, fixed_points)` -> stability tensor. Uses autograd Jacobian eigenvalues: all Re(lambda)<0 => stable, all>0 => unstable, mixed => saddle.

**Class `BifurcationAnalyzer`:**
- Computes bifurcation diagrams as control parameter (e.g., drug concentration) varies.
- `compute_bifurcation_diagram(parameterized_dynamics, param_range, projection_dim=0, n_points)` -> Dict with param_values, fixed_point_values, stability, n_stable, bifurcation_points. Detects where n_stable changes.

**Class `BasinOfAttraction`:**
- 2D grid of initial conditions -> batched Euler integration -> assign to nearest attractor.
- `compute_2d_basins(dynamics_fn, attractors, dim1=0, dim2=1, other_dims_value)` -> Dict with grid_x, grid_y, basin_ids (R,R), final_states (R,R,D), basin_volumes (M,).
- `compute_basin_boundary(basin_ids)` -> (R,R) binary boundary mask (4-neighbor check).
- `basin_stability_index(basin_volumes)` -> float [0,1]. BSI = V_max/V_total (Menck et al. 2013).

**Class `LyapunovExponentComputer`:**
- Maximal Lyapunov Exponent (MLE) via Benettin et al. (1980) algorithm.
- `compute_mle(dynamics_fn, initial_state, jacobian_fn)` -> (mle, mle_history). Transient warmup -> perturbation propagation -> periodic renormalization -> MLE = (1/t) * sum ln(||delta||). Negative = stable attractor, large |MLE| = hard to escape.
- `compute_lyapunov_spectrum_linear(A)` -> (D,) sorted real parts of eigenvalues. Quick estimate for linear systems near fixed points.

**Class `PhasePortraitGenerator`:**
- 2D phase portraits projected onto chosen dimensions.
- `compute_flow_field(dynamics_fn, dim1=0, dim2=1, other_dims_value, grid_resolution)` -> Dict with grid_x, grid_y, u, v, speed. Batched evaluation of dynamics on grid.
- `compute_trajectories(dynamics_fn, initial_conditions, dim1, dim2, n_steps)` -> Dict with trajectories_x (N,T), trajectories_y (N,T), time (T,). Euler integration with clamping.
- `compute_nullclines(dynamics_fn, dim1, dim2, ...)` -> Dict with nullcline1, nullcline2 (|dx_i/dt| on grid).
- `compute_potential_landscape(dynamics_fn, dim1, dim2, ...)` -> Dict with potential (R,R), normalized to [0,1]. Quasi-potential approximation: V(x) = -log(speed + eps). Low speed = valleys (attractors), high speed = ridges.

---

### 6.4 Cross-Module Architecture: How Clinical Predictions Connect to Model Outputs

**Treatment-conditioned predictions** use DragonNet for propensity scoring and causal outcome estimation. The Neural ODE provides longitudinal dynamics via FiLM conditioning on treatment sequences. IPTW removes confounding bias. The Actor-Critic RL head recommends treatment actions (stay/escalate/switch/add-CD38).

**Dynamic treatment regimes** augment patient history with ResistanceMap sensitivity scores (`resistance_map_fn`), then use Q-learning backward recursion + CRN adversarial training + AIPW blip estimation. Treatment feasibility constraints enforce NCCN clinical rules.

**Relapse prediction** uses a shared Neural ODE latent trajectory feeding 5 cause-specific hazard decoders, with CIF computation and landmark recalibration at clinical milestones (100d, 6mo, 12mo, 24mo post-ASCT).

**MRD prediction** fuses gene expression (VAE), clinical features, and immune markers via cross-attention, then predicts latent state trajectory (Transformer) and regresses MRD probabilities at two sensitivity thresholds. Epigenetic stability from VAE reconstruction variance modulates treatment intensity recommendations.

**EMD prediction** fuses 5 data layers (transcriptomics, radiomics, microenvironment, protein pathways, clinical) via attention, producing Weibull time-to-EMD and CNS-MM flagging.

**Immunotherapy response** uses MAML few-shot learning to rapidly adapt to novel agents with n=5-20 samples, forecasts antigen loss trajectories via LSTM, and plans sequential therapy (BCMA -> GPRC5D -> CELMoD).

**SMM progression** combines static PANGEA risk scoring with longitudinal clonal expansion (VAF) and immune decay tracking, feeding a Weibull mixture cure survival model.

**Interpretability** connects to model outputs through: (1) Temporal Integrated Gradients computing attributions along the ODE trajectory, (2) Counterfactual trajectory generation by modifying ODE dynamics, (3) Critical window detection via perturbation sensitivity analysis, (4) Phase transition detection via Jacobian eigenvalue analysis, (5) Uncertainty decomposition into aleatoric/epistemic/solver/data components with conformal calibration.

**Theory** verifies model identifiability (FIM rank, Lie derivative observability, empirical convergence) and analyzes stability (fixed points, bifurcations, Lyapunov exponents, basin boundaries). These ensure the learned ODE parameters are uniquely determined from data and the resistance landscape is properly characterized.

---

## Appendix G: Support Modules (Baselines, Verification, Landscape, etc.)

## Support / Utility Directories

### 1. `resistancemap/baselines/` (9 files, ~3600 lines)

Baseline model implementations for benchmarking ResistanceMap against established approaches.

#### `resistancemap/baselines/__init__.py`
- Exports: `PERCEPTIONBaseline`, `CPABaseline`, `SimpleBaselinesSuite`, `MOFAPlusBaseline`, `CellRankBaseline`, `ScVIBaseline`, `IntegrationBenchmark`

#### `resistancemap/baselines/registry.py`
Central baseline registry for MORT-FM Phase 10. Defines the uniform contract for all baselines.

- **Protocol `BaselineModel`**: Runtime-checkable protocol requiring `name: str`, `fit(X_train: Mapping[str, np.ndarray], y_train, **kwargs)`, `predict(X_test) -> np.ndarray`, `predict_survival(X_test) -> Optional[Dict]`, `save(path)`, `load(path)`
- **Type `Factory`**: `Callable[[Mapping[str, Any]], BaselineModel]`
- **Class `BaselineRegistry`**: Name-to-factory map singleton
  - `register(name, factory, *, overwrite=False)`: Register a factory; raises `KeyError` on duplicate unless `overwrite=True`
  - `list_available() -> List[str]`: Sorted list of registered names
  - `instantiate(name, config) -> BaselineModel`: Build baseline from config; validates contract compliance
  - `clear()`: Test-only helper to drop all entries
- **`REGISTRY`**: Module-level singleton instance

#### `resistancemap/baselines/classical_baselines.py`
Comprehensive classical ML baseline panel (v6 era).

- **Function `compute_metrics(y_true, y_prob, threshold=0.5)`**: Returns dict with auroc, auprc, f1, mcc, balanced_accuracy, brier_score
- **Class `BaseBaseline`**: ABC with `_preprocess(X, fit)` (StandardScaler + SimpleImputer), `fit()`, `predict_proba()`, `evaluate()`, `get_feature_importance()`
- **Class `RidgeBaseline`** (name="Ridge"): L2-regularized LogisticRegression wrapper
- **Class `RandomForestBaseline`** (name="RandomForest"): 500 trees, class-balanced, feature importance via `feature_importances_`
- **Class `ElasticNetBaseline`** (name="ElasticNet"): SGDClassifier with elasticnet penalty, sigmoid conversion for probabilities
- **Class `LogisticL1Baseline`** (name="LogisticL1"): L1 (Lasso) logistic with saga solver
- **Class `LogisticL2Baseline`** (name="LogisticL2"): L2 logistic with lbfgs solver
- **Class `XGBoostBaseline`** (name="XGBoost"): XGBClassifier with SHAP explanations via `get_shap_values()`; optional dep
- **Class `SVMBaseline`** (name="SVM_RBF"): RBF-kernel SVC with probability calibration
- **Class `KNNBaseline`** (name="KNN"): Distance-weighted KNN
- **Class `TrivialMeanBaseline`** (name="GlobalMean"): Predicts training-set prevalence
- **Class `PerDrugMeanBaseline`** (name="PerDrugMean"): Per-drug prevalence; `fit_multi_drug()`, `predict_drug()`
- **Class `SimpleBaselinesSuite`**: Orchestrator that runs all baselines with same splits
  - `run_all(X_train, y_train, X_test, y_test)`: Fit+evaluate all
  - `run_cv(X, y, n_folds=5)`: Stratified CV with per-fold fresh instances
  - `get_feature_importances(feature_names)`: Extract from all capable baselines
  - `results_table(results)`: Convert to DataFrame
- **Tests**: `test_individual_baselines`, `test_suite_runner`, `test_feature_importances`, `run_tests`

#### `resistancemap/baselines/cpa_baseline.py`
Compositional Perturbation Autoencoder (CPA/chemCPA) baseline (Lotfollahi et al., MSB 2023).

- **Class `SimpleCPAEncoder`** (nn.Module): Encoder to latent mu/logvar (LayerNorm + ReLU + Dropout)
- **Class `SimpleCPADecoder`** (nn.Module): Decoder from latent to expression
- **Class `DrugEncoder`** (nn.Module): Drug embedding + dose encoding with multiplicative interaction
- **Class `SimpleCPA`** (nn.Module): Full CPA with `reparameterize()`, `forward(x, drug_ids, doses)` returning (x_recon, mu, logvar, z_basal, adv_pred), `predict_perturbation(x_control, drug_ids, doses)`
- **Class `CPABaseline`**: Wrapper supporting scvi-tools CPA or SimpleCPA fallback
  - `fit(expression, drug_labels, doses, control_mask)`: Train CPA
  - `_fit_simple()` / `_fit_scvi()`: Backend dispatch
  - `predict_expression(control_expression, drug_name, dose)`: Predict post-perturbation expression
  - `fit_response_classifier()`: Bridge expression prediction to binary response via LogisticRegression
  - `predict_response()`, `evaluate_expression()`, `evaluate_response()`

#### `resistancemap/baselines/static_ml.py`
Static-ML baseline panel implementing the `BaselineModel` contract for Phase 10.

- **Function `_concat_modalities(X)`**: Stable sorted-key concatenation of modality dict
- **Class `_SklearnBaseline`**: Shared scaffolding with joblib save/load
- **Class `ElasticNetBaseline`** (name="elasticnet"): sklearn ElasticNet regressor
- **Class `RidgeBaseline`** (name="ridge"): sklearn Ridge regressor
- **Class `RandomForestBaseline`** (name="random_forest"): sklearn RandomForestRegressor
- **Class `XGBoostBaseline`** (name="xgboost"): xgboost.XGBRegressor; lazy import, raises ImportError if missing
- **Class `LightGBMBaseline`** (name="lightgbm"): lightgbm.LGBMRegressor; lazy import
- Auto-registers all into `REGISTRY` at import

#### `resistancemap/baselines/deep_static.py`
Deep-tabular baselines on static feature concatenation for Phase 10.

- **Function `_concat_modalities(X)`**: Same stable concatenation as static_ml
- **Class `_DeepBaseline`**: Shared scaffolding with deterministic init, `fit/predict/save/load`
- **Class `MLPBaseline`** (name="mlp"): 3-layer FC regressor (`_MLPModule`)
- **Class `CNNFeatureOrderBaseline`** (name="cnn_feature_order"): 1-D CNN over feature axis (`_CNNFeatureOrderModule`)
- **Class `FTTransformerBaseline`** (name="ft_transformer"): Minimal FT-Transformer-lite with per-feature token embeddings + single self-attention block + CLS aggregator (`_FTTransformerLiteModule`); cites Gorishniy et al. NeurIPS 2021
- Auto-registers all into `REGISTRY`

#### `resistancemap/baselines/sequence.py`
Longitudinal-aware baselines: LSTM and GRU over per-patient visit sequences.

- **Function `_extract_sequence(X)`**: Extract `(visits, mask)` from modality dict; supports 3-D `visits` key or falls back to concatenation with length-1 time axis
- **Class `_RecurrentBaseline`**: Shared scaffolding for RNNs with `fit/predict/save/load`
- **Class `LSTMVisitBaseline`** (name="lstm_visits"): `_LSTMVisitModule` with mask-aware last-valid-timestep extraction
- **Class `GRUVisitBaseline`** (name="gru_visits"): `_GRUVisitModule` with same mask logic
- Auto-registers into `REGISTRY`

#### `resistancemap/baselines/perception_baseline.py`
PERCEPTION framework reimplementation (Sinha et al., Nature Cancer 2024).

- **Constant `HALLMARK_MODULES`**: Dict of 10 MSigDB Hallmark pathway gene lists (PI3K_AKT_MTOR, MTORC1, MYC_TARGETS, P53_PATHWAY, NFKB_SIGNALING, WNT_BETA_CATENIN, APOPTOSIS, DNA_REPAIR, PROTEASOME, UNFOLDED_PROTEIN_RESPONSE)
- **Function `compute_pathway_scores(expression, gene_names, modules)`**: Mean z-scored expression per pathway module
- **Function `compute_gene_program_scores(expression, n_programs, seed)`**: PCA-based gene program scores
- **Class `DomainAdapter`**: Domain adaptation with methods `mean_shift`, `quantile`, `coral` (CORrelation ALignment)
- **Class `PERCEPTIONBaseline`**: Transfer learning from cell lines to patients
  - `_build_features()`: Concatenate expression + pathway scores + gene programs
  - `fit(cell_line_expression, cell_line_response, gene_names, patient_expression)`: Per-drug ElasticNetCV with domain adaptation
  - `predict(patient_expression, drug_names)`: Drug response prediction
  - `evaluate(patient_expression, patient_response)`: Per-drug + aggregate metrics
  - `get_feature_importance(drug_name)`: Coefficient-based feature ranking
- **Function `run_perception_benchmark()`**: Multi-seed benchmark runner

#### `resistancemap/baselines/integration_baselines.py`
Multi-omics integration baselines: MOFA+, CellRank, scVI.

- **Function `silhouette_by_label(latent, labels)`**: Silhouette score for integration quality
- **Function `compute_kbet(latent, batch_labels, n_neighbors)`**: Simplified kBET batch effect test
- **Function `compute_lisi(latent, labels, perplexity)`**: Simplified LISI (Local Inverse Simpson Index)
- **Function `evaluate_integration(latent, cell_type_labels, batch_labels)`**: Full integration quality evaluation
- **Class `MOFAPlusBaseline`**: MOFA+ factor analysis; falls back to joint PCA
  - `fit(modalities, sample_ids)`, `get_factors()`, `get_weights(modality)`, `predict_downstream()`, `evaluate_integration()`
- **Class `CellRankBaseline`**: Cell fate prediction via Markov chain; falls back to diffusion pseudotime + spectral clustering
  - `fit(expression, gene_names, velocity)`, `get_fate_probabilities()`, `get_pseudotime()`, `predict_downstream()`
- **Class `ScVIBaseline`**: Deep generative model; falls back to simple VAE or PCA
  - `fit(expression, batch_labels, gene_names)`, `get_latent()`, `predict_downstream()`, `evaluate_integration()`
- **Class `IntegrationBenchmark`**: Runs all three baselines with same data
  - `run()`, `summary_table()`

---

### 2. `resistancemap/verification/` (4 files, ~1550 lines)

Zero-trust verification system for multi-agent resistance prediction.

#### `resistancemap/verification/__init__.py`
Exports: `VerificationResult`, `ZeroTrustVerifier`, `InformationTheoreticBounds`, `DynamicalSystemsTheory`, `NetworkTopologyTheory`, `CognitiveDecisionTheory`, `Guardrail`, `GuardrailEngine`

#### `resistancemap/verification/zero_trust.py`
Cryptographic verification and statistical validation at every agent boundary.

- **Dataclass `VerificationResult`**: Fields: `passed`, `checks` (list of dicts), `hash_verified`, `statistical_valid`, `timestamp`, `agent_name`, `data_hash`; method `to_dict()`
- **Class `ZeroTrustVerifier`**: Maintains append-only hash chain and verification log
  - `compute_tensor_hash(tensor)` (static): SHA256 of tensor bytes + shape + dtype
  - `compute_dict_hash(data)` (static): SHA256 of JSON-serialized dict
  - `verify_data_integrity(data, expected_hash)`: Zero-trust hash verification
  - `verify_tensor_statistics(tensor, expected_stats, ...)`: Shape, NaN/Inf, range, mean/std, distribution checks
  - `verify_model_output(output, model_name, ...)`: NaN/Inf, range, model collapse detection, gradient health
  - `verify_biological_constraints(predictions)`: Resistance prob in [0,1], stability in [0,1], stochastic matrix rows sum to 1, IC50 in [1e-12, 1e-3] M, target protein format
  - `create_verification_checkpoint(agent_name, data)`: Append to hash chain; SHA256(previous || current || agent)
  - `verify_checkpoint(expected_hash, position)`, `get_audit_trail()`, `get_hash_chain()`, `log_verification()`

#### `resistancemap/verification/math_reasoning.py`
First-principles mathematical bounds and stability analysis.

- **Class `InformationTheoreticBounds`**:
  - `compute_fano_bound(features, labels, n_classes)`: Fano's inequality bound on achievable accuracy
  - `_knn_entropy_estimate()`: k-NN conditional entropy estimation
  - `mutual_information(features, labels)`: H(Y) - H(Y|X)
  - `information_bottleneck_bound(input_dim, latent_dim, output_classes)`: Check if latent capacity is feasible
- **Class `DynamicalSystemsTheory`**:
  - `compute_lyapunov_exponents(jacobian)`: Eigenvalue-based stability analysis
  - `basin_depth(jacobian)`: Negative of max real eigenvalue
  - `model_escape_rate_from_potential_well(basin_depth, noise_level, ...)`: Kramers-analog escape rate for learned U_theta; IMPORTANT docstring clarifies this is a model property, not a biological kinetic estimate; bug fixes #3 and #4 documented inline
  - `waddington_potential(states, velocities, bandwidth)`: Reconstructs quasi-potential via KDE Boltzmann inversion V(x) = -log(P_ss(x)); bug fix #4 replaces incorrect kinetic-energy computation
- **Function `_kde_fallback(states, bandwidth)`**: Histogram-based fallback
- **Function `_high_dim_kde_potential(states, bandwidth)`**: PCA-projected KDE for high-dim
- **Class `NetworkTopologyTheory`**:
  - `compute_degree_distribution(edge_index, n_nodes)`: Degree distribution + power-law exponent + scale-free test
  - `compute_pagerank(edge_index, n_nodes, damping, n_iterations)`: PageRank
  - `identify_network_motifs(edge_index, n_nodes, motif_type)`: Feed-forward loops, feedback loops, bistable switches
  - `target_druggability_score(pagerank, degree, betweenness)`: Weighted combination score
- **Class `CognitiveDecisionTheory`**:
  - `tversky_similarity(features_a, features_b, alpha, beta)`: Tversky's asymmetric similarity
  - `elimination_by_aspects(candidates, feature_names, feature_importance, ...)`: Feature-importance-ranked elimination

#### `resistancemap/verification/guardrails.py`
Clinical and scientific constraint enforcement.

- **Dataclass `Guardrail`**: Fields: `name`, `check_fn` (callable returning (bool, str)), `severity` ("error"/"warning"/"info"), `description`, `always_required`
- **Class `GuardrailEngine`**: Registry of domain constraints
  - `_register_default_guardrails()`: Registers 8 guardrails:
    1. `resistance_probability_range`: [0, 1] (error)
    2. `ic50_physiological_range`: [1e-12, 1e-3] M (error)
    3. `stability_score_range`: [0, 1] (error)
    4. `no_nan_in_predictions`: NaN/Inf check (error, always_required)
    5. `transition_probability_stochastic`: Rows sum to 1 (error)
    6. `target_protein_format`: Valid identifiers (warning)
    7. `stability_monotonicity`: Deeper basin -> longer transition time correlation (warning)
    8. `fano_bound_respected`: Model error >= Fano lower bound (info)
  - `register_guardrail(guardrail)`, `check_all(predictions)`, `check_single(guardrail_name, predictions)`, `get_failure_count(results, severity)`

---

### 3. `resistancemap/observability/` (3 files, ~411 lines)

Logging, tracing, and stage decorators for pipeline observability.

#### `resistancemap/observability/__init__.py`
Exports: `TraceContext`, `Recorder`, `set_recorder`, `get_recorder`, `record_handoff`, `record_tool_call`, `record_guardrail_violation`

#### `resistancemap/observability/instrumentation.py`
Append-only JSONL recorder sidecar for multi-agent layer.

- **Dataclass `TraceContext`**: Fields: `trace_id`, `agent_name`, `parent_span_id`, `started_at`, `ended_at`, `tool`, `cost_usd`, `tokens_in`, `tokens_out`, `span_id`
- **Class `Recorder`**: Bound to a single run_id, writes JSONL to `logs/<run_id>/observability.jsonl`
  - Thread-safe writes via `threading.Lock`; process-safe via per-write open + fsync
  - `record_trace_context(ctx)`, `record_handoff(from_agent, to_agent, ...)`, `record_tool_call(tool_name, ...)`, `record_guardrail_violation(rule, evidence, ...)`
- Module-level functions: `set_recorder(recorder)`, `get_recorder()`, `record_handoff(...)`, `record_tool_call(...)`, `record_guardrail_violation(...)` -- all no-ops if no recorder installed

#### `resistancemap/observability/v11_stage_decorator.py`
Context manager wiring AgentOps Tracer + Evaluator around v11 sprint scripts.

- **Class `_Tee`**: Mirror stdout writes to buffer + original stream for length measurement
- **Function `v11_stage(name)`** (context manager): Emits AgentOps span + TaskResult for a stage; captures stdout length as cost proxy (no LLM in v11 stages); writes per-stage JSON to `logs/agentops/v11/`

---

### 4. `resistancemap/data_quality/` (2 files, ~741 lines)

Data quality profiling and validation.

#### `resistancemap/data_quality/__init__.py`
Exports: `DataQualityReport`, `DataProfiler`, `MissingnessReport`, `BatchEffectReport`, `ClassBalanceReport`

#### `resistancemap/data_quality/profiler.py`
Dataset profiling against published SOTA benchmarks, specialized for MM genomics.

- **Dataclass `MissingnessReport`**: `missing_rate_per_feature`, `global_missing_rate`, `mcar_test_p_value`, `missingness_pattern` (MCAR/MAR/MNAR), `features_with_high_missingness`, `max_missing_feature`
- **Dataclass `BatchEffectReport`**: `batch_key`, `n_batches`, `kbet_score`, `lisi_score`, `silhouette_score`, `batch_distances`, `correctable`, `recommended_method`
- **Dataclass `ClassBalanceReport`**: `class_frequencies`, `class_proportions`, `shannon_entropy`, `effective_number_of_classes`, `min_class_size`, `imbalanced`, `recommended_reweighting`
- **Dataclass `SOTAComparison`**: `dataset_name`, `metric_name`, `our_value`, `sota_value`, `delta`, `percent_of_sota`, `better_than_sota`, `reference`
- **Dataclass `DataQualityReport`**: Comprehensive report with all sub-reports + `summary_string()`
- **Class `DataProfiler`**:
  - **Constant `BENCHMARKS`**: Known SOTA for MMRF_CoMMpass, CCLE_proteomics, GDSC, STRING_PPI_v12 with sample counts, published metrics, references
  - **Constant `MM_DRIVER_PROTEINS`**: 16 known MM drivers (BCMA/TNFRSF17, CD38, GPRC5D, FGFR3, KRAS, NRAS, TP53, DIS3, FAM46C, BRAF, TRAF3, MAX, IRF4, PTEN, RB1, CDKN2A)
  - `profile_dataset(data, dataset_name, labels, batch_key, feature_names, sample_names)`: Full profiling
  - `_check_missingness()`: MCAR test via missingness indicator correlations
  - `_compute_outlier_rate()`: IQR method
  - `_check_batch_effects()`: PCA + pairwise distances + kBET/LISI approximation
  - `_check_class_balance()`: Shannon entropy, effective classes, inverse-frequency reweighting
  - `_compute_feature_entropy()`: L2-norm-based feature importance entropy
  - `_check_protein_coverage()`: MM driver protein presence check
  - `_compare_to_sota()`: Compare against `BENCHMARKS`
  - `_compute_quality_score()`: 0-1 composite with missingness/outlier/batch/imbalance penalties
  - `_generate_recommendations()`: Actionable issue + recommendation lists

---

### 5. `resistancemap/mcp_integration/` (2 files, ~836 lines)

Model Context Protocol (MCP) connectors for external data sources.

#### `resistancemap/mcp_integration/__init__.py`
Exports: `MCPConnector`, `PubMedConnector`, `ChEMBLConnector`, `ClinicalTrialsConnector`, `HuggingFaceConnector`, `MCPOrchestrator`

#### `resistancemap/mcp_integration/connectors.py`
- **Dataclass `CacheEntry`**: `value`, `timestamp`, `ttl_seconds`, `is_expired()`
- **Class `LRUCache`**: LRU cache with TTL support; `get()`, `set()`, `clear()`, `size()`
- **Class `RateLimiter`**: Token bucket rate limiter; async `acquire(tokens)`
- **ABC `MCPConnector`**: Base class with `query()`, `validate_response()`, `query_with_cache()`
- **Class `PubMedConnector`**: v7 stub disarmed -- `query()` raises `NotImplementedError` directing to evidence-curator subagent; `validate_response()`, `search_resistance_literature()`, `verify_protein_drug_association()`, `get_latest_mm_studies()`
- **Class `ChEMBLConnector`**: v7 stub disarmed; `get_drug_targets()`, `get_bioactivity()`, `validate_predicted_target()`
- **Class `ClinicalTrialsConnector`**: v7 stub disarmed; `search_mm_trials()`, `get_trial_outcomes()`
- **Class `HuggingFaceConnector`**: v7 stub disarmed; `download_esm2_model()`, `check_model_version()`
- **Class `MCPOrchestrator`**: Manages all MCP connections with global LRU cache, method dispatch, `clear_cache()`, `cache_stats()`

---

### 6. `resistancemap/legacy/` (19 files, ~1440 lines)

Archived modules from v8/v15/v16 development phases. Preserved for historical checkpoint loading and `forward_legacy()` path.

#### `resistancemap/legacy/__init__.py`
Documents that new training MUST go through `resistancemap.mortfm` canonical surface.

#### `resistancemap/legacy/v15_planned_mortfm/` (17 files)
The v15 "planned MORT-FM" tree predating v17 LENS unification.

**Dynamics modules** (`models/dynamics/`, 8 files):
- **`graph_conditioned_drift.py`** - `GraphConditionedDrift(nn.Module)`: Latent drift conditioned on (z, t, drug, graph); pure-PyTorch message passing over PPI graph; `_message_pass()` with scatter-add
- **`neural_ode.py`** - `GraphConditionedNeuralODE(nn.Module)`: Wraps drift as Neural ODE; `_rk4_step()`, `_manual_rk4_integrate()` fallback; torchdiffeq if available
- **`neural_sde.py`** - `GraphConditionedNeuralSDE(nn.Module)`: SDE on learned Waddington landscape `dz = [-grad U + g_theta] dt + sigma dW`; `_DiffusionNet` for state-dependent diffusion; `_euler_maruyama_integrate()` fallback; torchsde if available; uses `torch.enable_grad()` for potential gradient in no_grad blocks
- **`jump_sde.py`** - `NeuralJumpSDE(nn.Module)`: SDE + compound Poisson jumps for discontinuous resistance events; `_JumpHead` learns intensity + jump vector; thinning-based jump sampling
- **`treatment_conditioned_dynamics.py`** - `TreatmentConditionedDynamics(nn.Module)`: Drug-specific drift contribution
- **`trajectory_sampler.py`** - `TrajectorySampler(nn.Module)`: Dispatches to ODE/SDE/Jump-SDE; returns `TrajectorySamples` dataclass (mean_path, samples, std_path, time_grid)
- **`temporal_decoder.py`** - `TemporalDecoder(nn.Module)`: Decodes z(t) back to per-modality tokens via `nn.ModuleDict` of projections

**Head modules** (`models/heads/`, 8 files):
- **`resistance_state_head.py`** - `ResistanceStateHead(nn.Module)`: MLP classifier for resistance phenotype (sensitive/persister/MRD-like/relapsed-resistant); cross-entropy loss
- **`time_to_resistance_head.py`** - Three survival heads:
  - `CoxSurvivalHead`: Log-hazard output, Cox partial likelihood loss (Breslow approximation)
  - `DiscreteTimeSurvivalHead`: Per-bin hazard logits, `survival_curve()` via cumprod, Gensheimer & Narasimhan 2019 NLL
  - `TimeToResistanceHead`: Facade dispatching cox/discrete
- **`trajectory_head.py`** - `TrajectoryDistributionHead(nn.Module)`: Per-time-bin diagonal Gaussian (mean, log_var); `sample()` method
- **`pathway_route_head.py`** - `PathwayRouteHead(nn.Module)`: Protein/edge/pathway attribution scores; `attribution_loss()` with BCE + L1 sparsity
- **`drug_risk_head.py`** - `DrugSpecificTrajectoryRiskHead(nn.Module)`: Per-drug trajectory risk via drug embedding + MLP
- **`uncertainty_head.py`** - `EvidentialUncertaintyHead(nn.Module)`: Dirichlet (classification) or NIG (regression) uncertainty; `evidential_classification_loss()` (Sensoy 2018)
- **`counterfactual_head.py`** - `CounterfactualInterventionHead(nn.Module)`: Counterfactual simulation via trajectory re-rollout; `simulate_node_blockade()`, `simulate_drug_combination()`, `rank_interventions()` returning `CounterfactualResult` dataclass

---

### 7. `resistancemap/experiments/` (3 files, ~1777 lines)

Ablation studies and statistical testing for ICML/ICLR submission.

#### `resistancemap/experiments/__init__.py`
Exports: `AblationStudy`, `AblationConfig`, `ModelFactory`, `run_full_ablation`, `StatisticalComparison`, `delong_test`, `bootstrap_ci`

#### `resistancemap/experiments/ablation_study.py`
Systematic component ablation across 4 dimensions.

- **Dataclass `AblationConfig`**: `n_seeds=5`, `n_cv_folds=5`, `test_frac=0.2`, `metric_names=[auroc, auprc, f1, mcc]`, `modalities`, `scale_fractions=[0.10..1.00]`, `horizons=[3, 6, 12, 24]`, `alpha=0.05`
- **Function `compute_metrics(y_true, y_prob)`**: auroc, auprc, f1, mcc
- **Class `ModelFactory`**: Creates model variants for ablation
  - `full_model()`, `without_modality(mod)`, `with_vanilla_ode()`, `with_naive_concatenation()`, `with_standard_vae()`, `without_lyapunov()`, `without_sparsity()`, `without_pathway_anchoring()`
- **Function `train_and_evaluate(model_fn, X_train, y_train, X_test, y_test, seed)`**: Generic train+eval
- **Class `AblationStudy`**:
  - `run_modality_ablation()`: Drop one modality at a time + single-modality tests
  - `run_architecture_ablation()`: Replace/remove architectural components (6 variants)
  - `run_scale_ablation()`: Vary training set size (10%-100%)
  - `run_horizon_ablation()`: Evaluate at different prediction horizons
  - `compute_significance()`: Paired Wilcoxon + Cohen's d + Bonferroni correction
  - `generate_ablation_table()`: Paper-ready DataFrame
  - `generate_scale_curve_data()`: Learning curve data
  - `save_results()`: JSON + CSV output
  - `plot_ablation_bar_chart()`, `plot_scale_curve()`: matplotlib visualization
- **Function `run_full_ablation()`**: Complete pipeline (all 4 ablation types + save + plot)

#### `resistancemap/experiments/statistical_testing.py`
Rigorous statistical testing for publication.

- **Functions for DeLong test**: `_compute_midrank()`, `_auc_variance_components()`, `delong_test(y_true, y_score_1, y_score_2)` -- returns z_statistic, p_value, auc_diff
- **Function `bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=1000, ci_level=0.95)`**: Bootstrap CIs for auroc/auprc
- **Function `bootstrap_paired_test(y_true, y_score_1, y_score_2, ...)`**: Paired bootstrap p-value
- **Function `mcnemar_test(y_true, y_pred_1, y_pred_2)`**: McNemar's test with continuity correction
- **Function `bonferroni_correction(p_values, alpha)`**: Bonferroni MHT correction
- **Function `benjamini_hochberg(p_values, alpha)`**: BH FDR control
- **Function `cohens_d(x1, x2)`**: Pooled-SD Cohen's d
- **Function `interpret_cohens_d(d)`**: "negligible"/"small"/"medium"/"large"
- **Function `power_analysis(n, effect_size, alpha, alternative)`**: Two-sample t-test power
- **Function `required_sample_size(effect_size, power, alpha)`**: Sample size for target power
- **Function `detectable_effect_size(n, power, alpha)`**: MDES at given n
- **Function `friedman_test(metric_matrix)`**: Friedman chi-squared for >2 models x >2 datasets
- **Function `nemenyi_post_hoc(metric_matrix, model_names, alpha)`**: Nemenyi critical difference test
- **Class `StatisticalComparison`**: Orchestrates all tests
  - `compare_two_models()`: Full pairwise (DeLong + bootstrap + McNemar)
  - `compare_multiple_models()`: vs reference with Bonferroni/BH correction
  - `compare_across_datasets()`: Friedman + Nemenyi
  - `power_analysis_report()`: Power table
  - `generate_results_table()`: Publication-ready table

---

### 8. `resistancemap/inference/` (3 files, ~1037 lines)

Prediction serving and batch inference.

#### `resistancemap/inference/__init__.py`
Docstring only: "End-to-end orchestration of all pipeline stages from raw data preparation through resistance landscape prediction."

#### `resistancemap/inference/api.py`
FastAPI server for ResistanceMap inference.

- **Pydantic models**: `PredictRequest`, `DrugResistancePrediction`, `InterventionTarget`, `PredictResponse`, `TrajectoryPredictRequest`, `TemporalLandscapeResponse`, `BatchPredictRequest`, `HealthResponse`
- **Function `create_app(pipeline, config)`**: FastAPI app factory with endpoints:
  - `GET /health`: Health check
  - `POST /predict`: Single sample prediction
  - `POST /predict/batch`: Batch prediction (max_batch_size configurable)
  - `POST /predict/trajectory`: Temporal trajectory prediction
  - `GET /targets/{sample_id}`: Intervention targets (stub)
  - `GET /landscape/{sample_id}`: Landscape data (stub)
- **Function `_interpret_landscape()`**: Human-readable interpretation generation

#### `resistancemap/inference/pipeline.py`
End-to-end L0-L5 inference pipeline.

- **Dataclass `TemporalLandscape`**: `timepoints`, `landscapes`, `trajectory_state_sequence`, `predicted_state_sequence`, `trajectory_accuracy`
- **Class `ResistanceMapPipeline`**: Full inference wiring L0-L5
  - `from_checkpoints(checkpoint_dir, config)` (classmethod): Loads all stage models from `.pt` files; graceful degradation if any stage checkpoint missing
    - L1: VAE (`ProteomeToEpigenomeVAE`)
    - L2: Stability scorer (`ChromatinODE` or `NeuralJumpSDE`)
    - L3: GNN (`PPIGraphNetwork`)
    - L4: Fusion (`ResistanceMapFusion`)
    - L5: Landscape (`ResistanceLandscape` + `ResistanceLandscapePredictor` + `TargetRanker`)
  - `predict_single(proteomics, sample_id)`: Full L0-L5 inference with graceful degradation per layer
  - `predict_batch(dataset, batch_size)`: Batched inference
  - `predict_trajectory(proteomics_timeseries, timepoint_labels, actual_states)`: Multi-timepoint trajectory

---

### 9. `resistancemap/landscape/` (12 files, ~1879 lines)

Resistance landscape modeling: Waddington potential, attractor basins, graph propagation, prediction heads.

#### `resistancemap/landscape/__init__.py`
Docstring only: "L5 layer: Takes fused multi-omics representation and produces resistance landscape predictions."

#### `resistancemap/landscape/potential.py`
Learned Waddington potential U_theta(z, drug).

- **Class `WaddingtonPotential(nn.Module)`**: Sum of Gaussian wells + MLP residual
  - Learnable: `attractor_centres` (K, d), `log_precision` (K, d), `depth` (K), MLP residual
  - `_mixture_energy(z)`: -log sum_k depth_k * N(z | mu_k, diag(prec_k^-1))
  - `forward(z, *, drug)`: U_mix + residual_scale * U_resid
  - `attractor_positions()`: Detached centres
- **Function `potential_gradient(potential, z, drug)`**: Autograd-based grad_z U

#### `resistancemap/landscape/scalar_potential.py`
Scalar potential trained by denoising score matching + PPI-Laplacian Tikhonov.

- **Dataclass `ScalarPotentialConfig`**: `latent_dim=64`, `hidden_dims=(256,256,128)`, `gene_proj_dim=0`, `activation="silu"`
- **Class `ScalarPotential(nn.Module)`**: Smooth MLP U_theta: R^d -> R
  - `forward(z)`: Scalar potential value
  - `grad_z(z)`: Autograd gradient
  - `gene_gradient(z)`: Project gradient to gene space via learnable `gene_proj`
- **Function `dsm_loss(model, z, sigma)`**: Vincent 2011 denoising score matching
- **Function `ppi_tikhonov_loss(model, z, L_ppi_indices, L_ppi_values, n_genes)`**: Quadratic form g^T L_PPI g

#### `resistancemap/landscape/attractor_basin.py`
Assign cells to attractor basins.

- **Function `assign_basin_nearest(z, potential)`**: Argmin distance to attractor centres
- **Function `assign_basin_descent(z, potential, *, n_steps, lr, drug)`**: Steepest-descent rollout assignment via iterated z = z - lr * grad U

#### `resistancemap/landscape/barrier_estimation.py`
Transition barrier estimation between attractor basins.

- **Function `estimate_pairwise_barriers(potential, *, n_path_points, drug)`**: Linear-path barrier height for each (k1, k2) pair
- **Function `estimate_transition_probability(barriers, temperature)`**: Arrhenius-style P(k1->k2) ~ exp(-barrier/T)

#### `resistancemap/landscape/landscape_regularizers.py`
Auxiliary losses shaping the learned Waddington potential.

- **Function `smoothness_penalty(potential, z, ...)`**: Finite-difference Hessian-trace approximation
- **Function `basin_separation_penalty(potential, *, min_distance)`**: Hinge penalty for overlapping attractors
- **Function `drug_barrier_penalty(potential, z_sensitive, z_resistant, drug)`**: Encourage drug to raise resistant-cell energy
- **Function `basin_entropy_penalty(basin_logits, *, target_uniform)`**: KL to uniform to prevent degenerate basin usage

#### `resistancemap/landscape/neural_ode_flow.py`
Neural-ODE forward operator over learned scalar Waddington potential. Replaces v10 single-Euler-step.

- **Dataclass `NeuralODEFlowConfig`**: `method="dopri5"`, `rtol=1e-5`, `atol=1e-7`, `use_adjoint=False`, `max_norm=1e3`
- **Class `_GradFieldRHS(nn.Module)`**: RHS of z_dot = -grad U(z) with gradient norm clamping
- **Class `NeuralODEFlow(nn.Module)`**: Adaptive ODE flow using torchdiffeq
  - `integrate(z0, t_span)`: Full trajectory
  - `forward_to_T(z0, T)`: Terminal state only
  - `energy_decay(z0, T)`: (U(z0), U(z(T))) for Lyapunov check
  - `euler_step(z0, T)`: v10 single-Euler comparison
- **Function `truncation_error(flow, z0, T, n_steps_grid)`**: Compare Euler chain vs adaptive Dopri5

#### `resistancemap/landscape/rwr_propagation.py`
Random Walk with Restart (RWR) + NPI z-score correction on STRING graph.

- **Function `column_normalize(A)`**: Column-stochastic transition P = A D^{-1}
- **Function `rwr_power(P, s, alpha=0.7, max_iter=200, tol=1e-8)`**: Power iteration; returns (r, info)
- **Function `build_seed_vector(seed_genes, gene_to_idx, n_nodes, weights)`**: Normalized seed indicator
- **Function `degree_stratified_null(deg, seed_indices, n_perm, n_bins, rng_seed)`**: Size-and-degree-matched random seed sets for NPI
- **Function `npi_zscore(P, deg, seed_indices, n_perm, alpha, ...)`**: Full NPI z-score computation; returns (r_obs, mean_null, std_null)
- **Function `top_k_by_zscore(r_obs, mean_null, std_null, gene_names, k, ...)`**: Top-k genes by z-score with variance-stabilizing floor

#### `resistancemap/landscape/gat_propagation.py`
Multi-alpha attention propagation over STRING -- ensemble generalization of v10 RWR.

- **Dataclass `MultiAlphaConfig`**: `alphas=(0.3, 0.5, 0.7, 0.9)`, `max_iter=200`, `tol=1e-8`, `aggregator="mean"`
- **Class `MultiAlphaAttentionPropagation`**: Training-free ensemble of RWR with H different alpha values
  - `_rwr_single(s, alpha)`: Single-alpha RWR
  - `propagate(s)`: Ensemble aggregation (mean/max/softmax)
- **Function `reproduce_rwr_alpha_07(P, s, tol)`**: Sanity check that H=1 alpha=0.7 matches v10 RWR

#### `resistancemap/landscape/hbayes_outer.py`
Hierarchical-Bayes outer layer for v10 Sprint 2 -- stratified partial pooling over 5 cytogenetic strata.

- **Constant `STRATA_DEFAULT`**: `["del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"]`
- **Dataclass `HBayesConfig`**: `n_strata=5`, `latent_dim=64`, `prior_mu_scale=1.0`, `prior_tau_scale=1.0`, `obs_noise_init=1.0`
- **Function `hbayes_model(z_obs, strata, cfg)`**: numpyro model with `mu ~ N(0, mu_scale)`, `tau ~ HalfCauchy`, `theta_s ~ N(mu, tau^2)`, `obs ~ N(theta_s[strata], sigma^2)`
- **Function `fit_svi(z_obs, strata, cfg, n_steps, lr, seed)`**: SVI fit with AutoNormal guide
- **`__main__` block**: Smoke test with JAX-RNG-derived data (NOT training data)

#### `resistancemap/landscape/waddington_visualizer.py`
2-D contour visualization of learned Waddington landscape.

- **Function `landscape_grid(potential, *, extent, n, proj, drug)`**: Returns (X, Y, U) grids for plt.contourf
- **Function `plot_landscape(potential, *, out_path, title, cells, cell_labels, proj, ...)`**: Full 2-D contour plot with cell overlay and attractor markers

#### `resistancemap/landscape/predictor.py`
L5 resistance landscape prediction module.

- **Dataclass `LandscapeResult`**: `sample_id`, `fused_representation`, `drug_resistance_{3,6,12}m` dicts, `top_intervention_targets`, `resistance_state`, `basin_of_attraction`, `transition_probabilities`, `confidence_score`, `epistemic_uncertainty`, `aleatoric_uncertainty`, `evidence_strength`
- **Class `TargetRanker`**: Ranks proteins by actionability (0.6 * drug_target_status + 0.4 * PPI_centrality) * resistance_contribution
- **Class `EvidentialResistanceHead(nn.Module)`**: Dirichlet-based uncertainty head
  - `forward(x)`: Returns (alpha, probs, epistemic_unc)
  - `compute_aleatoric_uncertainty(alpha)`: Expected entropy
- **Class `ResistanceLandscape(nn.Module)`**: Full L5 module with 4 heads:
  - Drug resistance head: (n_drugs * n_timepoints) sigmoid outputs
  - State classification head: Standard softmax OR EvidentialResistanceHead
  - Transition matrix head: (n_states, n_states) softmax
  - Target ranking head: (n_proteins) sigmoid scores
- **Class `ResistanceLandscapePredictor`**: End-to-end predictor wrapping model + ranker
  - `predict_single(fused_representation, sample_id, ...)`: Full single-sample prediction with evidential uncertainty
  - `predict_batch(fused_representations, ...)`: Vectorized batch prediction

---

### 10. `resistancemap/latent_compute/` (2 files, ~595 lines)

Latent space computation scheduling.

#### `resistancemap/latent_compute/__init__.py`
Exports: `ComputeJob`, `JobStatus`, `LatentComputeScheduler`

#### `resistancemap/latent_compute/scheduler.py`
Async job scheduler with priority queues, GPU memory budget, and dependency tracking.

- **Enum `JobStatus`**: QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
- **Dataclass `ComputeJob`**: `job_id`, `operation`, `priority` (lower=higher), `estimated_flops`, `estimated_memory_gb`, `status`, `result`, timestamps, `fn`, `args`, `kwargs`, `dependencies`; comparable via `__lt__`; `duration_seconds()`, `wait_time_seconds()`
- **Class `LatentComputeScheduler`**:
  - Config: `gpu_memory_budget_gb=24.0`, `max_concurrent_jobs=2`, `max_queue_size=1000`
  - `submit(operation, fn, args, kwargs, priority, ...)`: Submit job, returns job_id
  - `start()` / `stop(wait_for_completion)`: Async scheduler lifecycle
  - `_run_scheduler()`: Main loop checking completions + submitting available jobs
  - `_are_dependencies_satisfied(job)`: Dependency check
  - `_execute_job(job)`: Execute with GPU memory tracking
  - `cancel_job(job_id)`, `get_job_status(job_id)`, `estimate_completion_time(job_id)`
  - Pre-configured operations: `schedule_esm2_embedding()`, `schedule_ppi_subgraph_extraction()`, `schedule_trajectory_ensemble()`
  - Stats: `queue_depth()`, `running_count()`, `completed_count()`, `failed_count()`, `gpu_memory_utilization()`, `get_scheduler_stats()`, `get_running_jobs_summary()`, `get_queue_preview(limit)`

---

### 11. `resistancemap/utils/` (4 files, ~1397 lines)

Common utilities for checkpointing, metrics, and logging.

#### `resistancemap/utils/__init__.py`
Docstring only: "Infrastructure for saving/loading trained models, computing evaluation metrics, and structured logging with W&B integration."

#### `resistancemap/utils/checkpoint.py`
Checkpoint management for L0-L5 pipeline stages (adapted from MyeloMemory).

- **Class `CheckpointManager`**:
  - Stage names: data_prep, vae_pretrain, vae_finetune, trajectory_calibrate, protein_net_train, fusion_train, landscape_train, validate
  - `path(stage_name)`: File path for stage checkpoint
  - `exists(stage_name)`: Check existence + non-empty
  - `save(stage_name, data)`: Save with auto-metadata (timestamp, git hash); converts dataclass configs to dicts
  - `load(stage_name, map_location)`: Load with `weights_only=False`
  - `get_latest_completed_stage(stage_names)`: Find most recent completed stage
  - `list_checkpoints()`: List all with metadata (name, path, size_mb, modified)
  - `delete(stage_name)`: Remove checkpoint
  - `_get_git_hash()` (static): Subprocess call to git rev-parse HEAD

#### `resistancemap/utils/logging_utils.py`
Structured logging with optional W&B integration (adapted from MyeloMemory).

- **Function `setup_logger(log_dir, wandb_project, config)`**: Console + file handlers; W&B init
- **Function `log_stage_start(stage_name)`**: Start timing + W&B log
- **Function `log_stage_end(stage_name, metrics)`**: End timing + metrics logging
- **Function `log_metrics(metrics, step)`**: Console + W&B metric logging
- **Function `log_summary(title, summary_dict)`**: Formatted summary block

#### `resistancemap/utils/metrics.py`
Comprehensive evaluation metrics across all pipeline stages.

**VAE Metrics**:
- `reconstruction_mse(predicted, target)`: MSE for VAE reconstruction
- `latent_space_metrics(mu, log_var)`: KL divergence, active units, latent variance

**Stability Metrics**:
- `stability_calibration_metrics(predicted_scores, drug_sensitivity_variance)`: Spearman rho + calibration MSE

**Survival Metrics** (requires sksurv):
- `concordance_index(y_event, y_time, predictions)`: Harrell's C-index
- `time_dependent_auc(y_event, y_time, predictions, times)`: Dynamic AUC at specific timepoints
- `integrated_brier_score(y_event, y_time, survival_probs, times)`: IBS calibration metric

**ResistanceMap-specific**:
- `trajectory_accuracy(predicted_states, actual_states)`: State sequence match fraction
- `target_hit_rate(predicted_targets, validated_targets)`: Fraction of predicted targets in validated set
- `drug_resistance_metrics(predicted_ic50, true_ic50, drug_names, threshold)`: Per-drug AUROC/AUPRC/MSE

**Fairness & Equity Metrics** (Agent 3):
- `ppv_parity(predictions, labels, group_ids)`: PPV per demographic group + max disparity
- `equalized_odds(predictions, labels, group_ids)`: TPR/FPR parity per group + max gaps
- `worst_subgroup_performance(predictions, labels, group_ids, threshold)`: Worst-group AUROC + safety flagging
- `false_negative_analysis(predictions, labels, group_ids, drug_ids, threshold)`: FN rate per group + per-drug; flags high-risk groups
- `calibration_parity(predicted_probs, labels, group_ids, n_bins)`: ECE per group + disparity
- `auroc_with_ci(predictions, labels, n_bootstrap, confidence_level)`: Bootstrap AUROC + 95% CI
- `delong_test(predictions_a, predictions_b, labels)`: Simplified DeLong AUC comparison

**Aggregate**:
- `compute_full_metrics(predictions, dataset, split, actual_states, validated_targets, group_ids)`: All metrics including fairness if `group_ids` provided; confidence score bootstrap CIs; state distribution; trajectory accuracy; target hit rate

---

## Appendix H: Top-Level Scripts

# Part 8: Top-Level Scripts (`scripts/*.py`)

All paths relative to `ResistanceMap/` (BASE).

---

## 1. Core Pipeline Scripts (Legacy v6 Architecture)

### scripts/train.py
- **Purpose**: End-to-end training for the L0-L5 ResistanceMap pipeline (legacy architecture).
- **Pipeline stage**: Training orchestrator
- **CLI args**: `--config` (YAML path), `--device` (cpu/cuda), `--data-dir`, `--checkpoint-dir`, `--stages` (comma-separated L0-L5 or "all"), `--synthetic` (flag)
- **Data read**: CCLE proteomics/epigenomics via `load_ccle_proteomics`/`load_ccle_epigenomics`, STRING PPI via `load_string_ppi`, drug sensitivity via `load_drug_sensitivity`, optional scRNA-seq + MMRF data
- **Data written**: Checkpoints to `checkpoints/l1/` through `checkpoints/l5/`, logs to `logs/train.log`
- **Key functions**:
  - `generate_synthetic_data()` — creates random multi-omics data for testing (100 samples, 1000 proteins, 5000 epi features, 6 drugs)
  - `load_real_data(config)` — loads and aligns real CCLE + STRING + drug sensitivity data; finds intersection of sample IDs across modalities
  - `create_train_val_test_splits()` — random 70/15/15 split with configurable fractions
  - `train_stage_l1_vae()` — trains ProteomeToEpigenomeVAE with pan-cancer pretrain then hematological finetune
  - `train_stage_l2_stability()` — computes ODE-based chromatin bistability scores (requires torchdiffeq)
  - `train_stage_l3_gnn()` — protein network GNN training (requires torch_geometric; currently stub/placeholder)
  - `train_stage_l4_fusion()` — multi-modal fusion combining epigenetic+trajectory+protein+stability
  - `train_stage_l5_landscape()` — resistance landscape predictor (per-drug resistance at 3/6/12 months)
  - `evaluate_pipeline()` — basic test-set MSE evaluation
- **Dependencies**: `resistancemap.config`, `resistancemap.data.loaders`, `resistancemap.models.vae`, `resistancemap.models.trajectory`, `resistancemap.models.protein_network`, `resistancemap.models.fusion`, `resistancemap.landscape.predictor`, `resistancemap.utils.checkpoint`
- **Type**: Training script

### scripts/run_experiments.py
- **Purpose**: Master experiment runner for ICML/ICLR submission — orchestrates 8 phases of the full reproducible pipeline.
- **Pipeline stage**: Full experiment orchestrator
- **CLI args**: `--config` (required YAML), `--phase` (1-8, optional single phase), `--resume` (flag), `--dry-run` (flag), `--seed` (override)
- **Data read**: YAML config; Phase 1 reads RNA-seq h5ad + clinical CSV; cached preprocessed pickles
- **Data written**: `data/cached/preprocessed_<hash>.pkl`, `results/metrics/training_summary.json`, `results/metrics/statistical_tests.json`, `results/experiment_manifest.json`, paper figures + LaTeX tables
- **Key functions**:
  - `set_seed()` — sets Python/numpy/torch seeds for reproducibility
  - `load_config()` / `config_hash()` — YAML loading with SHA256 cache keys
  - `phase1_preprocess()` — data loading with scanpy normalization + HVG selection + train/val/test split; caches results via pickle
  - `phase2_train_model()` — trains ResistanceMap model or falls back to sklearn GradientBoosting surrogate
  - `phase3_train_baselines()` — trains baselines in parallel via joblib; sklearn fallbacks for all model types
  - `phase4_ablation_study()` — ablation across conditions and seeds using `configs/ablation_configs.py`
  - `phase5_evaluate()` — test-set evaluation with bootstrap CIs (AUROC, AUPRC, F1, MCC, Brier, ECE)
  - `phase6_statistical_tests()` — pairwise tests with Bonferroni correction, Cohen's d effect sizes
  - `phase7_generate_figures()` — delegates to `PaperFigureGenerator`
  - `phase8_generate_tables()` — delegates to `LatexTableGenerator`
- **Dependencies**: `resistancemap.model`, `training.trainer`, `configs.ablation_configs`, sklearn, scipy, yaml
- **Type**: Experiment orchestration script

### scripts/run_baselines_real.py
- **Purpose**: Train regression baselines on REAL data and compare to ResistanceMap's test_mse.
- **Pipeline stage**: Evaluation / baseline comparison
- **CLI args**: None (paths hardcoded)
- **Data read**: `checkpoints/data_ready.pt` (dataset + splits), `checkpoints/pipeline_validated.pt` (ResistanceMap metrics)
- **Data written**: `paper/tables/baseline_comparison.json`, `paper/tables/baseline_comparison.md`
- **Key functions**:
  - `load_real_data()` — loads dataset from checkpoint, concatenates proteomics+epigenomics, extracts drug sensitivity
  - `masked_mse_per_drug()` — trains one model per drug on observed (non-NaN) cells, pools squared residuals
  - `baseline_global_mean()` / `baseline_global_zero()` — trivial baselines
  - `run_panel()` — runs Ridge, ElasticNet, Lasso, RF, GBR, XGBoost, MLP baselines with PCA-256 features
- **Dependencies**: sklearn, xgboost, torch
- **Type**: Evaluation script
- **Note**: NaN-masked MSE evaluation matches `resistancemap/main.py:1440-1443`

### scripts/run_real_figures.py
- **Purpose**: Generate only paper figures from REAL data; skips any figure that would require synthetic/RNG-generated numbers.
- **Pipeline stage**: Reporting
- **CLI args**: None
- **Data read**: `checkpoints/pipeline_validated.pt`, `checkpoints/vae_pretrained.pt`
- **Data written**: `paper/figures/figure1_architecture.pdf`, `paper/figures/figure_real_metrics_summary.pdf`, `paper/figures/real_metrics.json`
- **Key functions**:
  - `load_real_metrics()` — extracts test_mse, target_drugs from checkpoint
  - `load_stage_val_losses()` — extracts vae best_metric from checkpoint
  - `figure_real_metrics_summary()` — renders only the real scalar test_mse + sample counts
- **Dependencies**: `generate_paper_figures.PaperFigureGenerator`, torch, matplotlib
- **Type**: Reporting script
- **Note**: Explicitly skips figure2-figure5 and all supplementary figures with documented reasons (no fabrication policy)

---

## 2. Paper Figure and Table Generation

### scripts/generate_paper_figures.py
- **Purpose**: Generate ALL paper figures in Nature/Science style (3.5"/7" wide, 7pt font, PDF 300 DPI, colorblind-friendly palette).
- **Pipeline stage**: Reporting
- **CLI args**: `--metrics` (dir), `--output` (dir), `--config` (YAML), `--format` (pdf/png/svg)
- **Data read**: `results/metrics/evaluation_metrics.json`, `results/metrics/ablation_results.json`
- **Data written**: 10 figures: `figure1_architecture`, `figure2_main_results`, `figure3_interpretability`, `figure4_temporal`, `figure5_ablation`, `figure_s1_disentanglement`, `figure_s2_identifiability`, `figure_s3_stability`, `figure_s4_per_drug`, `figure_s5_hyperparameter`
- **Key functions**:
  - `set_nature_style()` — configures matplotlib rcParams for publication style
  - `PaperFigureGenerator` class with methods for each figure
  - `figure1_architecture()` — model overview diagram with arrows, boxes for modalities/encoders/fusion/heads
  - `figure2_main_results()` — per-drug heatmap + AUROC/AUPRC bars + calibration curves
  - `figure3_interpretability()` — Waddington landscape (3D surface) + pathway attribution + counterfactual + gene network
  - `figure4_temporal()` — trajectory scatter + uncertainty fan plot + intervention windows
  - `figure5_ablation()` — component/modality ablation bars + data scaling curve
- **Dependencies**: matplotlib, numpy, scipy.stats
- **Type**: Reporting script
- **WARNING**: Most panels use `np.random` when real metrics are empty. Do not call `generate_all()` for real figures — see memory note `project_figure_fabrication_risk.md`.

### scripts/generate_latex_tables.py
- **Purpose**: Generate 5 publication-ready LaTeX tables with booktabs style.
- **Pipeline stage**: Reporting
- **CLI args**: `--metrics` (dir), `--output` (dir)
- **Data read**: `results/metrics/evaluation_metrics.json`, `results/metrics/ablation_results.json`, `results/metrics/statistical_tests.json`
- **Data written**: `paper/tables/table1_main_comparison.tex`, `table2_ablation.tex`, `table3_per_drug.tex`, `table4_interpretability.tex`, `table5_computational_cost.tex`
- **Key functions**:
  - `LatexTableGenerator` class
  - `table1_main_comparison()` — ResistanceMap vs 8 baselines with 6 metrics, CIs, significance markers
  - `table2_ablation()` — 13 ablation conditions with delta from full model, p-values from paired t-test
  - `table3_per_drug()` — 11 drugs x 4 models
  - `table4_interpretability()` — pathway recovery + counterfactual + disentanglement metrics
  - `table5_computational_cost()` — parameter counts, training time, GPU memory, inference time
- **Dependencies**: numpy, scipy.stats
- **Type**: Reporting script

### scripts/generate_actionability_report.py
- **Purpose**: Generate a clinical actionability decision matrix per drug (v8 artifact).
- **Pipeline stage**: Reporting
- **CLI args**: None
- **Data read**: `checkpoints/pipeline_validated.pt` (per_drug_metrics, actionability_summary)
- **Data written**: `paper/v8_artifacts/actionability_matrix.md`
- **Key functions**:
  - `main()` — reads per-drug Spearman, MSE, MAE from checkpoint; applies thresholds (Spearman >= 0.25 AND n >= 30 for screening; additionally MSE < 1.0 for calibrated); writes markdown with per-drug verdicts, drug class aggregates, and use/don't-use recommendations
- **Dependencies**: torch
- **Type**: Reporting script

### scripts/krishnaswamy_visualizations.py
- **Purpose**: Krishnaswamy-style (PHATE/UMAP) visualizations of VAE latents and drug sensitivity patterns.
- **Pipeline stage**: Visualization / reporting
- **CLI args**: None
- **Data read**: `checkpoints/data_ready.pt`, `checkpoints/vae_finetuned.pt`, `checkpoints/fusion_trained.pt`, `checkpoints/pipeline_validated.pt`
- **Data written**: 5 PNG figures + caption markdown files under `paper/v8_artifacts/visualizations/`
- **Key functions**:
  - `reduce_2d()` — manifold reduction with graceful PHATE -> UMAP -> PCA fallback
  - `fig1_latent_manifold()` — PHATE/UMAP of 886 cell-line VAE latents colored by stability score
  - `fig2_pseudotime_per_drug()` — per-drug pred-vs-truth scatter colored by stability pseudotime
  - `fig3_meld_condition_density()` — MELD-style drug-class density overlays on manifold
  - `fig4_residuals_heatmap()` — (pred-truth) per cell-line per drug heatmap
  - `fig5_resistance_landscape()` — kernel-smoothed resistance score on PHATE manifold
- **Dependencies**: torch, sklearn, matplotlib, optional phate/umap-learn
- **Type**: Visualization script

---

## 3. Data Ingestion and Preprocessing (MMRF / scRNA / GDC)

### scripts/preprocess_mmrf_gdc.py
- **Purpose**: Merge per-sample MMRF-CoMMpass GDC downloads into the single-file schema consumed by `load_mmrf_data`.
- **Pipeline stage**: Data ingestion
- **CLI args**: `--mmrf-dir` (default: `data/raw/mmrf_commpass`), `--manifest-dir` (default: `~/.gdc`), `--reuse-lookup` (flag), `--skip` (rna/snv/cnv, repeatable)
- **Data read**: Per-sample STAR counts TSVs (rna/), MAF.gz files (snv/), CNV segment files (cnv/); GDC manifest TSVs
- **Data written**: `file_to_case.tsv`, `gene_expression.tsv` (genes x aliquots), `mutations.tsv` (long-form), `copy_number.tsv` (long-form)
- **Key functions**:
  - `build_file_to_case_lookup()` — resolves file_id -> (patient, sample, aliquot, sample_type) via batched GDC `/files` API with retries
  - `build_gene_expression()` — merges STAR TSVs into (genes x aliquots) matrix with version-stripped Ensembl IDs, raw TPM
  - `build_mutations()` — concatenates MAFs, keeps Hugo_Symbol + variant fields
  - `build_copy_number()` — concatenates CNV segment files
- **Dependencies**: pandas, numpy, urllib (GDC API)
- **Type**: Data ingestion script

### scripts/pull_mmrf_clinical.py
- **Purpose**: Pull MMRF-CoMMpass clinical + treatment + sample timeline from GDC `/cases` API.
- **Pipeline stage**: Data ingestion
- **CLI args**: `--output-dir` (default: `data/raw/mmrf_commpass`), `--log-level`
- **Data read**: GDC Cases API (MMRF-COMMPASS project, paginated)
- **Data written**: `clinical.tsv` (1 row/case), `treatments.tsv` (1 row/(case,treatment)), `samples.tsv` (1 row/(case,sample,aliquot))
- **Key functions**:
  - `iter_cases()` — paginates through all ~995 MMRF cases with full expansion (demographic, diagnoses, treatments, follow_ups, samples)
  - `flatten_clinical_row()` — extracts ISS stage, diagnosis, demographics, survival, treatment counts
  - `flatten_treatment_rows()` — extracts regimen, therapeutic agents, days_to_treatment_start/end
  - `flatten_sample_rows()` — extracts aliquot submitter IDs (join key against RNA/SNV/CNV files)
- **Dependencies**: urllib (stdlib)
- **Type**: Data ingestion script (open-access, no token required)

### scripts/derive_cytogenetics.py
- **Purpose**: Derive per-patient binary flags for 5 MM-resistance cytogenetic events: del17p, chr1q21 gain, del13q, t(4;14), t(11;14).
- **Pipeline stage**: Data preprocessing
- **CLI args**: None (paths derived from script location)
- **Data read**: `data/raw/mmrf_commpass/copy_number.tsv`, `gene_expression.tsv`, `samples.tsv`
- **Data written**: `data/raw/mmrf_commpass/cytogenetics.tsv` (submitter_id + 5 flags + source_aliquot + notes)
- **Key functions**:
  - CNV calling using GRCh38 coordinates: TP53 region (chr17:7565097-7590856), 1q21 (chr1:153.5-155.5Mb), RB1 (chr13:48.3-48.5Mb) with log2-ratio thresholds (loss < -0.3, gain > +0.3)
  - Translocation surrogates via RNA expression: GMM-based calling for FGFR3/NSD2 (t(4;14)) and CCND1 (t(11;14)) with literature prevalence priors
  - `expr_call_posterior()` — 2-component GMM on log1p(TPM) with fallback to top-quantile cutoff
  - Validation log comparing call rates against literature bands (e.g., del17p 10-15%, t(11;14) 15-20%)
- **Dependencies**: pandas, numpy, optional sklearn (GaussianMixture)
- **Type**: Data preprocessing script

### scripts/build_scrna_summary.py
- **Purpose**: Build per-(sample, disease_stage) pseudobulk summaries from on-disk scRNA h5ad files.
- **Pipeline stage**: Data preprocessing
- **CLI args**: None
- **Data read**: `data/raw/gse124310.h5ad` (27,796 cells), `data/raw/gse271107.h5ad` (143,748 cells)
- **Data written**: `checkpoints/scrna_summary.pt` (torch dict with GSE124310, GSE271107, harmonized pseudobulk arrays + metadata)
- **Key functions**:
  - `_normalize_log_cpm()` — log1p(CPM) per row, stays sparse
  - `_pseudobulk()` — mean log1p(CPM) per group via sparse group-membership matrix; filters groups with < 30 cells
  - `harmonize()` — intersects gene sets, stacks pseudobulk across datasets, normalizes stage labels (NBM->HD, healthy->HD)
- **Dependencies**: anndata, scipy.sparse, numpy, torch
- **Type**: Data preprocessing script

### scripts/gdc_download.py
- **Purpose**: Download GDC open-access files from a manifest TSV with parallel HTTP, MD5 verification, and resume-by-existence.
- **Pipeline stage**: Data ingestion
- **CLI args**: `--manifest` (required), `--out-dir` (required), `--workers` (default 8), `--token` (optional, for controlled data)
- **Data read**: GDC manifest TSV (id, filename, md5, size, state)
- **Data written**: Downloaded files to `--out-dir`, with `.part` temp files during download
- **Key functions**:
  - `md5_of()` — computes MD5 of a file
  - `fetch_one()` — downloads one file with retry, MD5 verification, and size check; returns (uuid, status, size)
  - `main()` — ThreadPoolExecutor with progress logging
- **Dependencies**: urllib, hashlib, concurrent.futures
- **Type**: Data ingestion script

---

## 4. Analysis and Validation Scripts

### scripts/pilot_ppi_propagation.py
- **Purpose**: RWR (Random Walk with Restart) on STRING PPI network vs DepMap MM essentiality validation.
- **Pipeline stage**: Analysis / validation
- **CLI args**: None
- **Data read**: `data/raw/string_ppi.txt`, `data/raw/string/9606.protein.info.v12.0.txt.gz`, `data/raw/depmap/CRISPRGeneEffect.csv`, `data/raw/drug_metadata.json`
- **Data written**: `docs/ppi_cross_drug_corr.csv`, `docs/ppi_pilot_results.csv`, `docs/PPI_PROPAGATION_PILOT.md`
- **Key functions**:
  - `load_string_info()` — ENSP -> gene name mapping from STRING protein info
  - `build_graph()` — STRING undirected weighted graph (gene-name nodes, confidence >= 700)
  - `rwr()` — Random Walk with Restart, column-normalized adjacency, alpha=0.7
  - `drug_validation()` — per-drug: top-k RWR genes vs random column subsets of real DepMap data; one-sided Wilcoxon rank-sum
  - `cross_drug_specificity()` — pairwise Spearman on RWR rankings
  - `bortezomib_proteasome_check()` — validates proteasome subunit ranks in Bortezomib RWR
- **Dependencies**: numpy, pandas, networkx, scipy.stats
- **Type**: Analysis / validation script

### scripts/data_integration_audit.py
- **Purpose**: Head-to-head comparison of ResistanceMap's CrossModalFusionNet against 4 integration baselines on real CCLE data.
- **Pipeline stage**: Evaluation
- **CLI args**: None
- **Data read**: `checkpoints/data_ready.pt`, `checkpoints/pipeline_validated.pt`
- **Data written**: `paper/v8_artifacts/data_integration_audit.json`, `paper/v8_artifacts/data_integration_audit.md`
- **Key functions**:
  - `baseline_concat_ridge()` — naive concat -> Ridge
  - `baseline_pca_per_modality()` — PCA(256/20) per modality -> Ridge
  - `baseline_mofa_like()` — shared sparse factors via MiniBatchDictionaryLearning -> Ridge
  - `baseline_late_fusion()` — per-modality Ridge predictions averaged
  - `batch_effect_metrics()` — simplified kBET, iLISI, silhouette on embedding
- **Dependencies**: sklearn, torch, numpy
- **Type**: Evaluation script

### scripts/check_docs_consistent.py
- **Purpose**: CI check ensuring documentation numbers match `resistancemap._facts.FACTS`.
- **Pipeline stage**: CI / documentation consistency
- **CLI args**: None
- **Data read**: `README.md`, `ARCHITECTURE.md`, `AGENT_ORCHESTRATOR.md`
- **Data written**: None (prints pass/fail)
- **Key rules**:
  - PPI sizes from FACTS must appear in any doc mentioning STRING v12
  - Forbidden phrases: "Adams-Bashforth" (dopri5 is Dormand-Prince), "NIST SP 800-207", "first-principles math"
  - Fusion hidden_dim must be consistent across docs
- **Dependencies**: `resistancemap._facts`
- **Type**: CI validation script

---

## 5. MORT-FM Data Ingestion Scripts

### scripts/mortfm_download_public_data.py
- **Purpose**: Download (or symlink) all public raw inputs for MORT-FM pipeline.
- **Pipeline stage**: Data download orchestrator
- **CLI args**: `--data-root`, `--only` (comma-separated subset), `--skip`, `--depmap-release`, `--chembl-max`, `--link-legacy`, `--dry-run`
- **Sources**: DepMap (Model.csv, expression TPM, CRISPRGeneEffect), PRISM (secondary screen), GDSC (dose-response), STRING (links + aliases), UniProt (human FASTA), Reactome (pathway membership), ChEMBL (drug-target via REST API), GEO single-cell archives
- **Data written**: Files under `data/raw_public/{depmap,prism,gdsc,string,uniprot,reactome,chembl,geo_single_cell}/`
- **Type**: Data download script

### scripts/mortfm_ingest_all_blocks.py
- **Purpose**: Run all Block A/B/C/D ingestion scripts in order via subprocess.
- **Pipeline stage**: Ingestion orchestrator
- **Steps executed**: DepMap, GDSC, PRISM, STRING, UniProt, Reactome, ChEMBL, biological graph, Beat AML, GEO single-cell
- **Type**: Orchestration script

### scripts/mortfm_ingest_depmap.py
- **Purpose**: Block A — DepMap/CCLE ingestion. RNA TPM matrix + CRISPR gene-effect.
- **Data written**: `data/processed/omics/rna_matrix.parquet`, `data/processed/perturbation/crispr_gene_effect.parquet`, `data/processed/metadata/samples_depmap.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_gdsc.py
- **Purpose**: Block A — GDSC drug-response ingestion. Dose-response curves to long-form ln_IC50 + AUC.
- **Data written**: `data/processed/drug_response/gdsc_response_long.parquet`
- **Type**: Data ingestion

### scripts/mortfm_ingest_prism.py
- **Purpose**: Block A — PRISM repurposing-hub drug-response ingestion.
- **Data written**: `data/processed/drug_response/prism_response_long.parquet`
- **Type**: Data ingestion

### scripts/mortfm_ingest_string.py
- **Purpose**: Block A — STRING PPI ingestion. Edges filtered at combined_score >= 700 + UniProt alias mapping.
- **Data written**: `data/processed/graphs/string_edges.parquet`, `data/processed/graphs/string_aliases.parquet`
- **Type**: Data ingestion

### scripts/mortfm_ingest_uniprot.py
- **Purpose**: Block A — UniProt protein-sequence ingestion. Parses FASTA for (uniprot_id, gene_symbol, sequence).
- **Data written**: `data/processed/graphs/protein_sequences.parquet`, `data/processed/graphs/protein_nodes.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_reactome.py
- **Purpose**: Block A — Reactome pathway-membership ingestion.
- **Data written**: `data/processed/graphs/reactome_membership.parquet`
- **Type**: Data ingestion

### scripts/mortfm_ingest_chembl.py
- **Purpose**: Block A — ChEMBL drug-target ingestion. Normalizes column names, emits drug_targets.csv + drug_ontology.csv.
- **Data written**: `data/processed/drugs/drug_targets.csv`, `data/processed/drugs/drug_ontology.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_beataml.py
- **Purpose**: Block B — Beat AML 1.0 ingestion with raw/processed/auto modes.
- **CLI args**: `--raw-dir`, `--processed-dir`, `--out-dir`, `--mode` (auto/raw/processed)
- **Data written**: `data/processed/beataml/beataml_expression.parquet`, `beataml_drug_response.parquet`, `beataml_clinical.csv`, `beataml_sample_map.csv`, `metadata/samples_beataml.csv`
- **Acceptance check**: >= 300 expression specimens AND >= 100 drug-response specimens
- **Type**: Data ingestion

### scripts/mortfm_ingest_crispr.py
- **Purpose**: Block D — DepMap CRISPRGeneEffect.csv into tidy long-form parquet for causal validator.
- **Data written**: `data/processed/crispr/gene_effect.parquet` (long-form), `data/processed/crispr/essentiality_summary.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_geo_singlecell.py
- **Purpose**: Block C — GEO single-cell ingestion (scRNA/scATAC for MM and AML). Converts .h5ad to canonical format.
- **Data written**: `data/processed/single_cell/scrna/<GSE>.h5ad`, `data/processed/metadata/samples_<GSE>.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_perturbation.py
- **Purpose**: Block D — perturbation/CRISPR/Perturb-seq ingestion into canonical long table.
- **Data written**: `data/processed/perturbation/perturbation_response.csv`
- **Type**: Data ingestion

### scripts/mortfm_ingest_padimac.py
- **Purpose**: Ingest GSE116324 (PADIMAC: bortezomib-based treatment in newly diagnosed MM, 44 patients).
- **Note**: Single-baseline-timepoint, NOT paired longitudinal.
- **Data written**: `data/processed/padimac/padimac_expression.parquet`, aligned version, `logs/mortfm/padimac_ingest_report.json`
- **Type**: Data ingestion

---

## 6. MORT-FM Data Preprocessing and Graph Building

### scripts/mortfm_harmonize_identifiers.py
- **Purpose**: Block D-1 — improve gene-symbol to UniProt coverage past the 70% gate using STRING aliases.
- **Data written**: `data/processed/metadata/gene_protein_identifier_map.csv`, `logs/mortfm/identifier_mapping_report.json`
- **Tier system**: Tier 1 (HGNC exact), Tier 5 (STRING alias), Tier 0 (unmapped)
- **Type**: Data preprocessing

### scripts/mortfm_resolve_drug_names.py
- **Purpose**: Block D-2/B-3 — resolve drug names across GDSC + PRISM + BeatAML to canonical identifiers (MM ontology or ChEMBL).
- **Match tiers**: 1 (MM ontology synonym-exact), 2 (normalized-name), 3 (ChEMBL exact), 4 (ChEMBL normalized), 0 (unmapped)
- **Data written**: `data/processed/drugs/drug_identifier_map.csv`, `logs/mortfm/drug_resolution_report.json`
- **Type**: Data preprocessing

### scripts/mortfm_build_biological_graph.py
- **Purpose**: Block D — assemble heterogeneous biological graph from STRING PPI + ChEMBL drug-target + Reactome pathway membership + UniProt sequences.
- **Data written**: `data/processed/graphs/biological_edges.parquet`, `protein_nodes_from_graph.csv`, `drug_nodes.csv`, `pathway_nodes.csv`, `logs/mortfm/graph_coverage_report.json`
- **Claim gates**: `pathway_claim_allowed` = has_pathway_membership AND has_drug_target; `sequence_aware_claim_allowed` = has UniProt sequences
- **Type**: Graph construction

### scripts/mortfm_compute_esm2_embeddings.py
- **Purpose**: Block D — ESM-2 embedding cache for feature gene UniProt sequences.
- **CLI args**: `--model` (default: `facebook/esm2_t6_8M_UR50D`), `--max-sequences`, `--max-len` (1024)
- **Data written**: `data/processed/proteins/esm_embeddings__*.pt`, `logs/mortfm/uniprot_coverage_report.json`, `logs/mortfm/block_d_summary.json`
- **Type**: Feature computation

### scripts/mortfm_build_scrna_manifest.py
- **Purpose**: Block C-2 — build canonical scRNA manifest from processed .h5ad files.
- **Data written**: `data/processed/single_cell/scrna_manifest.csv`
- **Fields**: dataset_id, h5ad_path, n_cells, n_genes, n_hvgs, n_samples, disease_stages, has_counts_layer, has_real_time_units, has_pseudotime_only
- **Type**: Data preprocessing

### scripts/mortfm_preprocess_scrna.py
- **Purpose**: Run MORT-FM single-cell preprocessing (QC + log-normalization + HVG selection) on raw-counts .h5ad files.
- **CLI args**: `--input`, `--gse`, `--sample-col`, `--disease-stage-col`
- **Data written**: `data/processed/single_cell/scrna/<gse>.h5ad`, `<gse>_qc_report.json`
- **Type**: Data preprocessing

### scripts/mortfm_build_longitudinal_dataset.py
- **Purpose**: Lane 4+5 — build MORT-FM longitudinal training substrate from MMRF paired-patient + outcomes tables.
- **Data written**: `data/processed/longitudinal/temporal_pairs.parquet`, `logs/mortfm/longitudinal_data_report.json`
- **Gate decisions**: longitudinal_trajectory (needs 100 pairs, 50 unique patients, real time units) and resistance_emergence_TT2L (10 pairs with endpoint, 5 observed events)
- **Type**: Data preprocessing

### scripts/mortfm_build_temporal_pairs.py
- **Purpose**: Build TemporalTrainingPair objects from snapshot + outcome pickles.
- **CLI args**: `--snapshots` (required), `--outcomes` (required), `--out` (required), `--allowed-gaps`, `--gap-tolerance`, `--include-unlabelled`
- **Data written**: Pickled temporal pairs
- **Type**: Data preprocessing

### scripts/mortfm_prepare_mmrf.py
- **Purpose**: Build MORT-FM-ready cohort from MMRF CoMMpass dump (clinical outcomes).
- **CLI args**: `--data-dir` (required), `--out-dir`, `--endpoint` (pfs/os/relapse)
- **Data written**: `data/processed/mortfm/outcomes.pkl`, `summary.json`
- **Type**: Data preprocessing

### scripts/mortfm_align_beataml_features.py
- **Purpose**: Block B-2 — align BeatAML expression matrix onto Block-A checkpoint's top-2000 RNA feature space.
- **Data written**: `data/processed/beataml/beataml_expression_block_a_features.parquet`, `beataml_feature_mask.parquet`, `logs/mortfm/beataml_feature_alignment_report.json`
- **Gate**: coverage_rate >= 0.80 for block_a_gate_pass
- **Type**: Feature alignment

---

## 7. MORT-FM Training Scripts

### scripts/mortfm_pretrain_foundation.py
- **Purpose**: Pre-train MORT-FM foundation encoder (Stages A + B) on cell-line multi-omics cohort.
- **Dependencies**: `resistancemap.mortfm.{data_module, model, schemas, trainer}`
- **Data written**: `cfg.checkpoint_dir/stage_B_complete.pt`
- **Type**: Training script

### scripts/mortfm_train_cellline_foundation.py
- **Purpose**: Block A — Cell-line foundation pretraining + drug-response on merged DepMap (RNA) x GDSC (drug response) cohort.
- **Data read**: `data/processed/omics/rna_matrix.parquet` (1775 cell-lines x 19220 genes), `data/processed/drug_response/gdsc_response_long.parquet` (484k rows)
- **Data written**: `checkpoints/mortfm/block_a_cellline_foundation.pt`, `logs/mortfm/block_a_*.json/csv`
- **Type**: Training script

### scripts/mortfm_train_beataml_finetune.py
- **Purpose**: Block B — BeatAML patient/specimen-level drug-response finetune (static_drug_response endpoint).
- **Data read**: BeatAML expression (451 specimens x 22843 genes), drug response (95,300 rows), clinical, biological graph
- **Data written**: `checkpoints/mortfm/beataml_finetuned.pt`, `logs/mortfm/beataml_*.{json,csv}`
- **Type**: Training script

### scripts/mortfm_train_beataml_drug_conditioned.py
- **Purpose**: v17.6 — Block B with drug-conditioned head. Runs scratch vs init-A variants side-by-side on patient-disjoint splits.
- **Data written**: `checkpoints/mortfm/beataml_v17_{scratch,init_a}.pt`, per-drug CSVs
- **Type**: Training script

### scripts/mortfm_train_scrna_state_encoder.py
- **Purpose**: Block C — pre-train cell-state RNA encoder on processed single-cell AnnDatas.
- **Constraint**: No calendar-time labels (pseudotime only); encoder+fusion weights exported for Block E integration.
- **Type**: Training script

### scripts/mortfm_train_block_c_contrastive.py
- **Purpose**: v17.5 — Block C re-train with stage-aware objectives (SupCon, stage classification, pseudotime ordinal loss).
- **Acceptance bands**: macro-F1 > 0.25 minimum, > 0.35 useful, > 0.45 strong
- **Type**: Training script

### scripts/mortfm_train_lens_resistance.py
- **Purpose**: Phase A.2 — train MORT-FM-LENS on MMRF baseline->followup paired latents (n=29) with TT2L resistance endpoint.
- **Method**: LOO training with GraphEnergyResistanceSDE + CompetingRiskHead; Harrell C-index over LOO predictions
- **Constraint**: n=29, high variance; resistance_emergence claim only if C-index lower CI > 0.5
- **Type**: Training script

### scripts/mortfm_train_survival.py
- **Purpose**: Train MORT-FM Stage F (survival/time-to-resistance) + Stage H (calibration).
- **Constraint**: Refuses survival claims if patient count below `cfg.min_patient_n_for_survival_claim`
- **Dependencies**: `resistancemap.mortfm.{data_module, model, schemas, trainer}`
- **Type**: Training script

### scripts/mortfm_train_trajectory.py
- **Purpose**: Train MORT-FM Stage E (longitudinal trajectory).
- **Constraint**: Requires temporal pairs with `x_t_delta is not None` for supervision
- **Type**: Training script

---

## 8. MORT-FM Evaluation and Validation Scripts

### scripts/mortfm_beataml_compare.py
- **Purpose**: Lane 1 — head-to-head comparison of Block B scratch vs init-A variants on BeatAML.
- **Data written**: `results/mortfm/beataml_transfer_vs_scratch.csv`, `beataml_drug_family_metrics.csv`, `beataml_top_k_recovery.csv`, `logs/mortfm/beataml_compare_summary.json`, figures
- **Gate**: hematologic_specimen_drug_response — requires both variants median Spearman >= 0.10 AND init-A beats scratch by >= 0.02 in paired bootstrap
- **Type**: Evaluation script

### scripts/mortfm_validate_beataml.py
- **Purpose**: Block B-1 — validate processed BeatAML data contract (structural validator, no training).
- **Type**: Validation script

### scripts/mortfm_block_d_validation.py
- **Purpose**: Lane 3 — Block D extended validation: drug-target coverage, ESM-2 cache hit rate, pathway x drug coverage matrix.
- **Data written**: `results/mortfm/drug_family_target_coverage.csv`, `feature_gene_esm2_hit_rate.csv`, `pathway_x_drug_family_coverage.csv`, `logs/mortfm/block_d_validation_summary.json`
- **Type**: Validation script

### scripts/mortfm_causal_smoke_test.py
- **Purpose**: Phase B smoke — exercise causal counterfactual stack on real biological-graph edges + real MMRF z0 latents. Does NOT claim causal_mechanism.
- **Data written**: `results/mortfm/causal_edge_effects.csv`, `logs/mortfm/causal_smoke_summary.json`
- **Type**: Smoke test

### scripts/mortfm_causal_evidence_report_v2.py
- **Purpose**: v17.8 — three-way evidence join (CRISPR + ChEMBL drug-target + Reactome pathway) with permutation-test enrichment.
- **Data written**: `results/mortfm/causal_edge_evidence_v2.csv`, `logs/mortfm/causal_evidence_v2_summary.json`
- **Gate**: causal_mechanism requires >= 2 essentials AND >= 2 drug-target-supported edges AND enriched pathway p < 0.05
- **Type**: Evaluation script

### scripts/mortfm_survival_baseline_suite.py
- **Purpose**: v17.3 — mandatory survival baseline suite for MMRF 29-patient TT2L cohort. Clinical-only Cox PH + permutation null vs LENS variants.
- **Data written**: `results/mortfm/survival_baseline_suite.csv`, `logs/mortfm/survival_baseline_suite.json`
- **Verdict**: No patient-level claim unless full model beats clinical-only Cox AND permutation null
- **Type**: Evaluation script

### scripts/mortfm_regularized_lens_eval.py
- **Purpose**: v17.7 — evaluate regularised LENS baseline on MMRF 29-patient TT2L cohort with bootstrap optimism-corrected C-index.
- **Substrates**: Clinical-only (5 features) and clinical + z0 (5 + 64)
- **Type**: Evaluation script

### scripts/mortfm_longitudinal_pair_audit.py
- **Purpose**: v17.4 — audit longitudinal pair pool: cohort registry, censoring policy, patient-disjoint + temporal-holdout splits.
- **Type**: Audit script

### scripts/mortfm_lens_variant_compare.py
- **Purpose**: Aggregate four LENS resistance-training variant summaries into one comparison table. No new training.
- **Type**: Evaluation consolidation

### scripts/mortfm_scrna_latent_atlas.py
- **Purpose**: Lane 2 — build scRNA latent atlas from Block C encoder. PCA/UMAP embedding, stage classifier (5-fold CV), PNG diagnostics.
- **Type**: Analysis / visualization

### scripts/mortfm_v17_unified_smoke.py
- **Purpose**: v17.4 — verify MORTFM model exposes both legacy `forward()` and new `forward_lens()` paths on real MMRF z0 latents.
- **Type**: Contract / smoke test

### scripts/mortfm_v17_gate_revalidate.py
- **Purpose**: v17.9 — final gate revalidation across 12 claim levels. Reads all v17 results and produces auditable grant/block table.
- **Type**: Gate validation

### scripts/mortfm_validate_integrated_checkpoint.py
- **Purpose**: Cross-check `checkpoints/mortfm/mortfm_integrated.pt` against artifact, feature-space, endpoint, and claim-gate registries.
- **Type**: Validation script

---

## 9. MORT-FM Integration and Registry Scripts

### scripts/mortfm_integrate_blocks_bcd.py
- **Purpose**: Block E — integrate Blocks A+B+C+D into a single provenance-tagged checkpoint.
- **Logic**: Loads Block A foundation, overlays Block B BeatAML-tuned weights, records pointers to Block C encoder + Block D ESM-2 cache. Records shape mismatches. Emits a single `.pt` + JSON manifest.
- **Data written**: Integrated checkpoint + JSON manifest
- **Type**: Integration script

### scripts/mortfm_emit_registries.py
- **Purpose**: Emit the four MORT-FM registries from current on-disk state (feature spaces, endpoints, claim gates, artifacts).
- **Data written**: `data/processed/features/mortfm_feature_spaces.json`, `logs/mortfm/endpoint_registry.json`, `logs/mortfm/claim_gate_registry.json`, `logs/mortfm/artifact_registry.json`
- **Type**: Registry emission

### scripts/mortfm_run_real_endtoend.py
- **Purpose**: Run MORT-FM pipeline end-to-end on REAL public data already on disk. Trains at `acceptance_level="debug"` only. No patient-level claim from this run.
- **Data sources**: MMRF clinical (995 rows), MMRF RNA (optional), GSE271107 (143k cells), BeatAML processed, STRING PPI (473k edges), GDSC (484k rows), PRISM (701k rows)
- **Data written**: `checkpoints/mortfm/real_endtoend.pt`, `logs/mortfm/real_endtoend_summary.json`
- **Type**: End-to-end integration test

---

## Summary Statistics

| Category | Count |
|---|---|
| Total scripts in `scripts/` | 64 |
| Legacy pipeline (v6) | 4 (train, run_experiments, run_baselines_real, run_real_figures) |
| Paper figures/tables/reporting | 5 (generate_paper_figures, generate_latex_tables, generate_actionability_report, krishnaswamy_visualizations, run_real_figures) |
| Data ingestion (MMRF/GDC/scRNA) | 5 (preprocess_mmrf_gdc, pull_mmrf_clinical, derive_cytogenetics, build_scrna_summary, gdc_download) |
| Analysis/validation (legacy) | 3 (pilot_ppi_propagation, data_integration_audit, check_docs_consistent) |
| MORT-FM data ingestion | 14 (mortfm_ingest_*, mortfm_download_public_data, mortfm_ingest_all_blocks) |
| MORT-FM preprocessing/graph | 9 (mortfm_harmonize_*, mortfm_resolve_*, mortfm_build_*, mortfm_align_*, mortfm_compute_*, mortfm_preprocess_*, mortfm_prepare_*) |
| MORT-FM training | 9 (mortfm_train_*, mortfm_pretrain_*) |
| MORT-FM evaluation/validation | 11 (mortfm_validate_*, mortfm_causal_*, mortfm_survival_*, mortfm_regularized_*, mortfm_beataml_compare, mortfm_lens_variant_compare, mortfm_scrna_latent_atlas, mortfm_longitudinal_pair_audit, mortfm_v17_*) |
| MORT-FM integration/registry | 4 (mortfm_integrate_blocks_bcd, mortfm_emit_registries, mortfm_run_real_endtoend, mortfm_validate_integrated_checkpoint) |

---

## Appendix I: Script Subdirectories (v10, v11, v12, mortfm canonical)

# ResistanceMap Scripts & Paper Artifacts -- Exhaustive File Catalogue

## 1. `scripts/v10/` -- V10 Sprint Scripts (38 files, ~6800 lines)

### Sprint 1: Foundation Scaffold (s1a-s1e)

#### `scripts/v10/s1a_build_mmrf_baseline_matrix.py`
- **Purpose**: Build per-patient earliest-visit baseline expression matrix from MMRF CoMMpass RNA-seq.
- **CLI arguments**: None (hardcoded paths).
- **Key logic**: Parses aliquot IDs to extract timepoint via `_T{N}_` regex; groups by patient and keeps earliest timepoint; transposes gene-expression matrix to patients-x-genes; drops zero-variance genes.
- **Output**: `data/processed/mmrf_baseline_expression.parquet`, `data/processed/mmrf_baseline_metadata.tsv`, `paper/v8_artifacts/v10_sprint1/s1a_summary.json`.
- **Dependencies**: pandas, numpy. Reads `data/raw/mmrf_commpass/gene_expression.tsv` and `file_to_case.tsv`.

#### `scripts/v10/s1b_build_ppi_laplacian.py`
- **Purpose**: Build symmetric-normalized PPI Laplacian from STRING v12 with non-leakage filter (combined >= 700, experiments+database >= 400).
- **CLI arguments**: `--links-file`, `--tag`, `--sprint-dir`.
- **Key logic**: Streams STRING links file; filters by combined score and non-observational channels; maps ENSP to HGNC via protein.info; builds sparse adjacency; computes L = I - D^{-1/2} A D^{-1/2}.
- **Output**: `data/processed/ppi_laplacian{tag}.npz`, `data/processed/ppi_gene_index{tag}.tsv`, summary JSON.
- **Dependencies**: scipy.sparse, gzip, pandas.

#### `scripts/v10/s1c_encode_mmrf_z64.py`
- **Purpose**: Encode MMRF baseline expression to 64-d latents via PCA (Sprint-1 substitute for scGPT).
- **CLI arguments**: None.
- **Key logic**: log1p-TPM transform; selects top-5000 genes by mean expression; standardizes per-gene; fits 64-d PCA.
- **Output**: `data/processed/mmrf_z64.npy` (n_patients, 64), `mmrf_z64_sample_ids.json`, `mmrf_z64_pca.npz`, `mmrf_gene_features.tsv`.
- **Dependencies**: sklearn.decomposition.PCA.

#### `scripts/v10/s1d_train_scalar_potential.py`
- **Purpose**: Train scalar Waddington potential U_theta via denoising score matching with PPI Tikhonov regularization.
- **CLI arguments**: `--epochs` (400), `--batch-size` (128), `--lr` (1e-3), `--lambda-latgrad`, `--lambda-ppi`, `--no-ppi`, `--sigma-min/max`, `--seed`, `--device`, `--tag`, `--laplacian-suffix`, `--sprint-dir`.
- **Key logic**: Loss = L_DSM(sigma) + lambda_lat * ||grad_z U||^2 + lambda_ppi * <grad_z U, L_PPI grad_z U>_gene. PPI Tikhonov operates on gene-space projection of latent gradient via frozen PCA decoder. AdamW + cosine annealing.
- **Output**: `checkpoints/u_theta_{tag}.pt`, training log JSON.
- **Dependencies**: `resistancemap.landscape.scalar_potential` (ScalarPotential, dsm_loss).

#### `scripts/v10/s1e_falsification_gates.py`
- **Purpose**: Run F4 (CurlFraction) and F5 (median-r identifiability) falsification gates.
- **CLI arguments**: `--tag`, `--sprint-dir`.
- **Key logic**: F4 uses discrete Helmholtz-Hodge decomposition on k-NN simplicial complex. Computes empirical drift (mean-shift), model drift (-grad U), residual, then Hodge lstsq decomposition. CurlFraction threshold 0.30. F5 simulates gradient flow from held-out z0 toward random z* targets; median ||z_hat - z*|| / ||z0 - z*|| >= 0.9 means refutation.
- **Output**: `paper/v8_artifacts/{sprint_dir}/falsification_gates_{tag}.json`.
- **Dependencies**: sklearn.neighbors.NearestNeighbors, scipy.sparse.linalg.

### Sprint 2: Paired Patients + HBayes (s2b-s2f)

#### `scripts/v10/s2b_paired_patients.py`
- **Purpose**: Identify paired baseline-to-relapse MMRF patients (>=2 RNA-seq timepoints).
- **Key logic**: Groups aliquots by patient; keeps earliest two distinct timepoints; checks for 2nd-line therapy records in treatments.tsv. Actual: 29 paired, 20 strict-paired (spec assumed 42/319).
- **Output**: `data/processed/mmrf_paired_patients.tsv`, summary JSON.

#### `scripts/v10/s2c_encode_paired.py`
- **Purpose**: Encode paired-patient (z0, z1) latents using Sprint-1 frozen PCA decoder.
- **Key logic**: Applies same PCA transform (standardize + project) to both timepoint aliquots.
- **Output**: `data/processed/mmrf_paired_z64.npz` (z0, z1, patient_id, timepoints, has_2nd_line).

#### `scripts/v10/s2d_f5_paired.py`
- **Purpose**: Re-run F5 (median-r) on REAL paired (z0, z1) pairs instead of random targets.
- **CLI arguments**: `--tag`.
- **Key logic**: Euler flow z_hat = z0 - eta * grad_U(z0) over grid of (n_steps, step_size). Reports median ||z_hat - z1|| / ||z0 - z1||. Refutation if >= 0.9.
- **Output**: `paper/v8_artifacts/v10_sprint2/f5_paired_{tag}.json`.

#### `scripts/v10/s2e_hbayes_inference.py`
- **Purpose**: Hierarchical-Bayes SVI on MMRF cytogenetic strata (5 flags: del17p, chr1q21_gain, del13q, t_4_14, t_11_14).
- **Key logic**: Multi-label HBayes model: mu ~ N(0,1), tau ~ HalfCauchy(1), theta_s ~ N(mu, tau^2), z_p ~ N(mu + M*theta, sigma_obs^2). AutoNormal guide, SVI with ELBO, 2000 steps.
- **Output**: `paper/v8_artifacts/v10_sprint2/hbayes_posterior.npz`, summary JSON.
- **Dependencies**: jax, numpyro.

#### `scripts/v10/s2f_f1_f2_gates.py`
- **Purpose**: F1 (cross-stratum coverage) + F2 (baseline-ablation log-lik) falsification gates.
- **Key logic**: F1: hold out each stratum, fit HBayes on rest, check 90% predictive CI coverage >= 70%. F2: proper paired LOO -- for each paired patient, fit on full cohort minus held-out vs fit on paired-only minus held-out; bootstrap on delta log-lik vector. F2 PASSES iff 95% CI excludes 0 and delta > 0.
- **Output**: `paper/v8_artifacts/v10_sprint2/f1_f2_gates.json`.

### Sprint 3: PPI Propagation + Falsification F3/F6/F7 (s3a-s3f + fixes)

#### `scripts/v10/s3a_build_ppi_adjacency.py`
- **Purpose**: Build PPI weighted adjacency (un-normalized) from STRING for RWR propagation.
- **CLI arguments**: `--links-file`, `--gene-index`, `--tag`.
- **Output**: `data/processed/ppi_adjacency_{tag}.npz`, `ppi_degree_{tag}.npy`.

#### `scripts/v10/s3c_drug_seed_propagation.py`
- **Purpose**: RWR propagation for 10 MM-relevant drugs using NPI degree-stratified null (alpha=0.7, 1000 permutations).
- **Key logic**: 10 drugs with seed gene sets (Bortezomib: PSMB5/1/2; Panobinostat: HDAC1/2/3/6; etc.). Uses `rwr_power()` and `npi_zscore()` from `resistancemap.landscape.rwr_propagation`.
- **Output**: `paper/v8_artifacts/v10_sprint3/drug_propagation.npz`, summary JSON.

#### `scripts/v10/s3d_f3_crispr_oracle.py`
- **Purpose**: F3 -- CRISPR rank-sum oracle for 10-drug panel. Tests whether RWR top-50 driver genes have more negative Chronos essentiality than random gene sets.
- **Key logic**: One-sided Wilcoxon rank-sum + 1000-permutation test per drug. Bonferroni correction across drugs. Spec threshold: >= 4/10 Bonferroni-significant.
- **Output**: `paper/v8_artifacts/v10_sprint3/f3_crispr_oracle.json`.

#### `scripts/v10/s3e_f6_driver_recall.py`
- **Purpose**: F6 -- MM driver gene recall test. Tests whether RWR seeded by consensus MM drivers recovers other drivers in top-100.
- **Key logic**: 46 consensus MM drivers from Walker/Lohr/Bolli/Manier. Per-driver RWR, count consensus in top-100. 1000 random-seed permutation null. Spec: permutation p < 0.001.
- **Output**: `paper/v8_artifacts/v10_sprint3/f6_driver_recall.json`.

#### `scripts/v10/s3f_f7_multiseed.py`
- **Purpose**: F7 -- multi-seed pooling fixes for HDAC drugs. Compares single-seed (HDAC1) vs pooled (HDAC1+2+3+6) Chronos rank-sum.
- **Key logic**: F7 PASSES iff pooled p <= single p for both Panobinostat and Vorinostat.
- **Output**: `paper/v8_artifacts/v10_sprint3/f7_multiseed.json`.

#### `scripts/v10/s3prep_depmap_haem_subset.py`
- **Purpose**: Subset DepMap CRISPRGeneEffect.csv to haematologic cell lines (~64 lines).
- **Output**: `data/processed/chronos_haem.parquet`, `chronos_haem_lineage.tsv`.

#### `scripts/v10/s3prep_crispr_oracle.py`
- **Purpose**: CRISPR-oracle smoke test using proteasome 26S subunits as positive control.
- **Output**: `paper/v8_artifacts/v10_sprint1/s3prep_oracle_smoke.json`.

#### `scripts/v10/s3fix_diagnostic.py`
- **Purpose**: Diagnostic investigation of F3/F6/F7 failures. Sweeps include-seeds x MM-vs-haem x top-K for F3; alias rescue for F6; partial-seed leave-N-out for F6.
- **Output**: `paper/v8_artifacts/v10_sprint3/s3fix_diagnostic.json`.

#### `scripts/v10/s3fix_apply.py`
- **Purpose**: Apply diagnosed fixes and re-run F3/F6/F7. F3: include seeds in top-K, restrict to MM-only DepMap, Vorinostat HDAC1+2+3 only. F6: partial-seed K-of-N test with alias-rescued consensus. F7: multiple RNG seeds per condition.
- **Output**: `f3_crispr_oracle_fixed.json`, `f6_driver_recall_fixed.json`, `f7_multiseed_fixed.json`.

#### `scripts/v10/s3fix_v2.py`
- **Purpose**: Align drug panel with pilot's essentiality-MoA composition (adds Doxorubicin, Etoposide, Dinaciclib; drops Daratumumab/Venetoclax from F3 panel). F6 with degree-preserving null + K-sweep.
- **Output**: `f3_crispr_oracle_v2.json`, `f6_driver_recall_v2.json`.

#### `scripts/v10/s3fix_v3.py`
- **Purpose**: F6 redesigned with continuous rank statistic + degree-matched label-permutation null (B=10000). Uses mean rank of other drivers as T_obs, label permutation preserving seed-side topology.
- **Output**: `paper/v8_artifacts/v10_sprint3/f6_driver_recall_v3.json`.

#### `scripts/v10/s3fix_v3_f7.py`
- **Purpose**: F7 v3 with mechanism-correct seeds. Vorinostat gets HDAC1+2+3 only (HDAC6 IC50 1.5uM). Bortezomib as positive control. Evaluated on haem64 + mm19 panels.
- **Output**: `paper/v8_artifacts/v10_sprint3/f7_multiseed_v3.json`.

#### `scripts/v10/s3_fix_f7_prism_oracle.py`
- **Purpose**: F7 v4 -- replace Chronos essentiality oracle with PRISM drug-sensitivity oracle (Spearman(proteomics, -AUC)) for HDAC drugs lacking CRISPR essentiality signal due to paralog buffering.
- **Output**: `paper/v8_artifacts/v10_sprint3/f7_multiseed_v4_prism.json`.
- **Dependencies**: PRISM secondary screen, CCLE proteomics.

### Sprint 4: Causal Mediation NIE (s4a-s4e + fixes)

#### `scripts/v10/s4a_build_outcome_treatment.py`
- **Purpose**: Build analysis-ready Sprint-4 dataset with Bortezomib 1L indicator and right-censored TT2L.
- **Output**: `data/processed/mmrf_outcomes_treatment.tsv`, `mmrf_ensembl_to_symbol.tsv`.

#### `scripts/v10/s4b_build_mediator.py`
- **Purpose**: Build per-patient proteasome mediator score from baseline expression (z-score-weighted mean of Bortezomib RWR top-50 genes).
- **Output**: `data/processed/mmrf_proteasome_score.tsv`.

#### `scripts/v10/s4c_merge_analysis_table.py`
- **Purpose**: Merge outcome+treatment, mediator, and confounders into one analysis-ready table.
- **Output**: `data/processed/mmrf_sprint4_analysis.tsv` (columns: submitter_id, bort_1L, M_proteasome, tt2L_days, had_2L, confounders).

#### `scripts/v10/s4d_nie_estimation.py`
- **Purpose**: Lange-Hansen-style hazard-scale NIE for proteasome -> TT2L on Bort cohort. Pearl SCM with Cox PH interaction model.
- **Key logic**: Cox PH with A + M + A*M + C. NDE = beta_A + beta_AM * E[M|A=0], NIE = (beta_M + beta_AM) * delta_M. Bootstrap B=1000 patient-level resamples. E-value sensitivity (VanderWeele-Ding 2017).
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_proteasome_tt2L.json`.
- **Dependencies**: lifelines.CoxPHFitter.

#### `scripts/v10/s4e_falsification_neg_control.py`
- **Purpose**: F9 -- negative-control mediator NIE with B=20 random degree-matched gene sets. Tests that proteasome NIE is not a generic false positive.
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_negative_control.json`.

#### `scripts/v10/s4e_sensitivity_alt_mediators.py`
- **Purpose**: Alternative mediator constructions (M_seed3: PSMB5+1+2 only; M_uniform: top-50 uniform weights) to rule out construction-specific artifacts.
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_alt_mediators.json`.

#### `scripts/v10/s4e_v2_focused_neg_control.py`
- **Purpose**: Re-run F8/F9/F10 with the FOCUSED 3-gene proteasome mediator (PSMB5+PSMB1+PSMB2). B=50 negative-control 3-gene sets. Bootstrap B=1000.
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_focused_falsification.json`.

#### `scripts/v10/s4_fix_f10_richer_mediator.py`
- **Purpose**: Test richer proteasome mediators (20S beta, 20S alpha, 19S ATPase, 19S non-ATPase, full 26S complex) for stronger NIE -> higher E-value.
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_richer_mediators.json`.

#### `scripts/v10/s4_fix_f10_tipping_point.py`
- **Purpose**: Tipping-point sensitivity analysis (RR_AU, RR_UY) contour for unmeasured confounding. Shows structural F10 fail is correct given realistic MM confounder strengths.
- **Output**: `paper/v8_artifacts/v10_sprint4/nie_tipping_point.json`.

### Sprint 5: Mondrian Conformal (s5)

#### `scripts/v10/s5_mondrian_jackknife_plus.py`
- **Purpose**: Mondrian jackknife+ conformal wrapper for per-stratum 90% coverage on log(TT2L+1).
- **Key logic**: 5 mutually-exclusive cyto strata. Ridge base learner. Full JK+ LOO (N LOO fits). Per-stratum quantile intervals from same-stratum calibration scores. Spec target: >= 3 strata within +/-3% of 90%, all 5 within +/-5%.
- **Output**: `paper/v8_artifacts/v10_sprint5/mondrian_jk_plus_coverage.json`.
- **Dependencies**: sklearn.linear_model.Ridge.

### Sprint 6: Cross-Disease (Beat AML) (s6a-s6c)

#### `scripts/v10/s6a_extract_beataml.py`
- **Purpose**: Extract Beat AML 1.0 (Tyner 2018) supplement tables (Clinical, RPKM, Drug Responses).
- **Output**: `data/processed/beataml_clinical.tsv`, `beataml_rpkm.parquet`, `beataml_drug_response.tsv`.

#### `scripts/v10/s6b_cross_disease_f3.py`
- **Purpose**: Cross-disease F3 -- MM-derived RWR top-50 evaluated against Beat AML Bortezomib/Panobinostat ex-vivo drug response. Oracle: Spearman(RPKM, -AUC).
- **Output**: `paper/v8_artifacts/v10_sprint6/cross_disease_f3.json`.

#### `scripts/v10/s6b_v2_aml_native.py`
- **Purpose**: Native-AML F3 -- RWR seeded on AML driver genes (FLT3, BCL2, TOP2A, DNMT) evaluated against Beat AML drug-response. Tests whether the framework works on AML when seeded with AML-relevant genes.
- **Output**: `paper/v8_artifacts/v10_sprint6/cross_disease_f3_aml_native.json`.

#### `scripts/v10/s6c_conformal_beataml.py`
- **Purpose**: Sprint 5 Mondrian JK+ applied to Beat AML Bortezomib AUC prediction. ELN-2017 cytogenetic risk strata.
- **Output**: `paper/v8_artifacts/v10_sprint6/conformal_beataml_coverage.json`.

### Sprint 7: Diffusion Posterior (s7)

#### `scripts/v10/s7_diffusion_posterior.py`
- **Purpose**: Single-snapshot diffusion-posterior wrapper + F8 LOO energy distance test. Three predictors: prior-dynamics (Euler step + noise), constant baseline, population-mean baseline. Mann-Whitney U for ED comparison.
- **CLI arguments**: None explicit (grid search internal).
- **Key logic**: Sweeps (eta, sigma) grid. Energy distance between K=200 posterior samples and singleton z1. F8 PASS iff MWU p < 0.05 vs BOTH baselines.
- **Output**: `paper/v8_artifacts/v10_sprint7/f8_energy_distance.json`.

---

## 2. `scripts/v11/` -- V11 Scripts (11 files, ~3000 lines)

#### `scripts/v11/s1c_encode_mmrf_v11.py`
- **Purpose**: Foundation-model encoder upgrade from PCA to Geneformer V1-10M. Downloads model from HuggingFace, applies rank-value encoding to bulk TPM (pseudobulk-as-cell), mean-pools last hidden states, PCA(256->64).
- **Key logic**: Honest documentation of scFoundation blockers (Python 3.7 pin, missing loaders). Geneformer chosen as tractable proxy. Off-distribution caveat documented.
- **Output**: `data/processed/mmrf_z64_v11.npy`, `mmrf_z64_v11_pca.npz`, `mmrf_z64_v11_sample_ids.json`, `paper/v8_artifacts/v11_sprint1/encoder_upgrade.json`.
- **Dependencies**: huggingface_hub, transformers.BertForMaskedLM.

#### `scripts/v11/s1f_ode_forward_diagnostic.py`
- **Purpose**: Neural-ODE forward operator diagnostic (3 checks): D1 energy-decay (Lyapunov M1), D2 truncation-error study (Euler vs Dopri5), D3 drift-direction agreement (cosine v10 vs v11).
- **Output**: `paper/v8_artifacts/v11_sprint1/ode_forward_diagnostic.json`.
- **Dependencies**: `resistancemap.landscape.neural_ode_flow`.

#### `scripts/v11/s2f_f1_f2_gates_v11.py`
- **Purpose**: Zero-trust leak check -- re-run F1/F2 on v11 Geneformer-derived latents. If F2 flips from REFUTED to PASS, encoder is leaking paired-patient identity.
- **Output**: `paper/v8_artifacts/v11_sprint1/f1_f2_gates_v11.json`.

#### `scripts/v11/s3_gat_f6_replication.py`
- **Purpose**: Re-run F6 v3 with multi-alpha attention-propagation ensemble (alpha in {0.3, 0.5, 0.7, 0.9}). Must not regress the F6 strict-pass.
- **Output**: `paper/v8_artifacts/v11_sprint3/f6_driver_recall_gat.json`.
- **Dependencies**: `resistancemap.landscape.gat_propagation`.

#### `scripts/v11/s5_v11_conformal_waddington_features.py`
- **Purpose**: Mondrian JK+ with Waddington-derived features. Three base learners: Ridge_v10 (11 features), Ridge_v11features (11 + 4 Waddington: U(z0), U(z(T)), ||grad U||, ||z(T)-z0||), GBM_v10. Tests whether features or model flexibility helps.
- **Output**: `paper/v8_artifacts/v11_sprint5/mondrian_jk_plus_v11.json`.

#### `scripts/v11/s5c_discrimination_metrics.py`
- **Purpose**: Discrimination metrics (C-index, AUROC at 12mo/24mo) on JK+ point predictions. Three base learners with LOO point predictions. Comparison to mmSYGNAL C-index.
- **Output**: `paper/v8_artifacts/v11_sprint5/discrimination_metrics.json`.

#### `scripts/v11/s5d_cox_discrimination.py`
- **Purpose**: Cox PH base learner for survival-aware discrimination. Three variants: Cox_v10, Cox_v11features, Cox_v11_richer (with expression PCs). LOO per-patient log-hazard, Harrell C-index + AUROC.
- **Output**: `paper/v8_artifacts/v11_sprint5/cox_discrimination.json`.

#### `scripts/v11/s5e_cox_with_programs.py`
- **Purpose**: Close the +0.040 mmSYGNAL gap by adding transcriptional-program features. Tests Cox with routed mmSYGNAL score, 6 raw mmSYGNAL submodel scores, and program-activity PCs.
- **Output**: `paper/v8_artifacts/v11_sprint5/cox_with_programs.json`, `cox_per_patient_log_hazards.npz`.

#### `scripts/v11/s5f_mmsygnal_complete_routing.py`
- **Purpose**: Complete mmSYGNAL routing with derived del(1p36) and FGFR3 calls from copy-number and expression data. Re-routes using corrected subtype calls and paired bootstrap delta vs v11.5 best.
- **Output**: `paper/v8_artifacts/v11_sprint5/mmsygnal_complete_routing.json`.

#### `scripts/v11/s5g_paired_cindex_test.py`
- **Purpose**: Sharpen v11.5 vs mmSYGNAL TIE finding. Implements both U-statistic z-test (Kang/Tian/Cai 2015) and B=10000 paired percentile bootstrap for paired C-index difference.
- **Output**: `paper/v8_artifacts/v11_sprint5/paired_cindex_test.json`.

#### `scripts/v11/s7_v11_dps_neural_ode.py`
- **Purpose**: Re-run F8 DPS with Neural-ODE forward operator. Expected F8 STILL FAIL per Stone minimax bound at N_paired=29. Demonstrates v10's refutation was robust to operator choice.
- **Output**: `paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`.

---

## 3. `scripts/v12/` -- V12 Scripts (4 files, ~1500 lines)

#### `scripts/v12/paired_wilcoxon_rm_vs_latefusion.py`
- **Purpose**: Paired Wilcoxon test of ResistanceMap vs Late-Fusion (avg-Ridge) per-drug MSE. Tests H0: median(MSE_RM - MSE_LF) >= 0 via one-sided Wilcoxon signed-rank. Optional `--multi-seed` for 5-seed retraining.
- **Output**: `paper/v8_artifacts/v12_sprint1/paired_wilcoxon_latefusion.json`.
- **Dependencies**: scipy.stats, sklearn.linear_model.Ridge, statsmodels.

#### `scripts/v12/s_v12_hbayes_ppc.py`
- **Purpose**: Posterior-predictive checks (Gelman 1996) on Sprint-2 HBayes 5-stratum model. Draws M=1000 posterior samples from AutoNormal guide, simulates z_rep, computes Bayesian PPC p-values for stratum-centroid magnitude and dispersion.
- **Output**: `paper/v8_artifacts/v12_sprint1/hbayes_ppc.json`.

#### `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`
- **Purpose**: Formal head-to-head: MOFA+ shared factors vs v11.5 Cox PH features on MMRF PFS endpoint. Fits MOFA+ (mofapy2 or MiniBatchDictionaryLearning fallback) on log1p(TPM) top 5000 HVGs, extracts 64 factors, trains LOO Cox PH on three feature sets: MOFA-only, Cox_v11_richer, combined. Paired bootstrap delta.
- **Output**: `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_pfs.json`.

#### `scripts/v12/s_v12_t1114_specialist_head.py`
- **Purpose**: t(11;14) specialist head. Adds BCL2/MCL1/CCND1/BCL2L1 z-scores and their interactions with cyto_t_11_14 to the v11.5 Cox model. Tests whether S4 stratum C-index lifts without disturbing marginal coverage.
- **Output**: `paper/v8_artifacts/v12_sprint1/t1114_specialist_head.json`.

---

## 4. `scripts/v12_refactor/` -- Refactoring (1 file, ~815 lines)

#### `scripts/v12_refactor/krishnaswamy_visualizations.py`
- **Purpose**: Krishnaswamy-lab-style scientific visualizations for v12-refactor audit. All figures from on-disk checkpoints only. Embedding fallback hierarchy: PHATE -> UMAP -> PCA. Produces publication-quality multi-panel figures.
- **Output**: `paper/v8_artifacts/v12_refactor_audit/visualizations/{fig}.png/.pdf/.caption.md`, `docs/V12_REFACTOR_AUDIT/visualizations.md`.
- **Dependencies**: matplotlib, scipy, torch. Optional: phate, umap-learn.

---

## 5. `scripts/archive/` -- Archived Scripts (3 files, ~474 lines)

#### `scripts/archive/mortfm_causal_evidence_report.py`
- **Purpose**: v17 per-edge causal-evidence report with HGNC bridging and external-evidence joins (DepMap, drug targets, Reactome).
- **CLI arguments**: `--edges`, `--crispr-summary`, `--drug-targets`, `--out`, `--out-summary`.
- **Dependencies**: `resistancemap.mortfm.graph.IDHarmonizer`.

#### `scripts/archive/mortfm_lens_smoke_test.py`
- **Purpose**: Smoke-test MORT-FM-LENS GraphEnergyResistanceSDE + ResistanceBasin + HittingTime on a real Block C latent batch. Verifies finite trajectories, basin probability sum=1, monotone hitting-time CDF, LossRouter skip behavior.
- **Dependencies**: `resistancemap.mortfm.trajectory`, `resistancemap.training.loss_router`.

#### `scripts/archive/mortfm_smoke_test.py`
- **Purpose**: End-to-end MORT-FM smoke test -- exercises v15 stack (schemas, loaders, encoders, fusion, dynamics, landscape, heads, losses, trainer) on tiny structurally valid examples. Confirms import, data contract, forward/backward, checkpoint save/load.
- **Dependencies**: `resistancemap.mortfm.model.MORTFM`, `resistancemap.mortfm.trainer.MORTFMTrainer`.

---

## 6. `scripts/baselines/` -- Baseline Runners (2 files, ~682 lines)

#### `scripts/baselines/elasticnet_baseline.py`
- **Purpose**: ElasticNet baseline for drug-response prediction. Loads MultiOmicsDataset checkpoint, builds per-sample feature matrix from proteomics + epigenomics + CRISPR (NaN->0 imputed), fits per-drug ElasticNetCV. Top-K=2000 proteomics feature selection by Pearson correlation.
- **CLI arguments**: `--ckpt`, `--out`.
- **Output**: `paper/v8_artifacts/baselines/elasticnet_results.json`.
- **Dependencies**: sklearn.linear_model.ElasticNetCV.

#### `scripts/baselines/mofa_baseline.py`
- **Purpose**: MOFA+ baseline against ResistanceMap. ~20 latent factors over 3 views (proteomics, epigenomics, crispr_effect). Trains on same train split, projects val/test, fits per-drug Ridge from MOFA factors to drug-response. Falls back to sklearn MiniBatchDictionaryLearning if mofapy2 unavailable.
- **Output**: `paper/v8_artifacts/baselines/mofa_factors.npy`, `mofa_sample_ids.json`, `mofa_results.json`.

---

## 7. `scripts/dev/` -- Dev Utilities (1 file, ~358 lines)

#### `scripts/dev/dead_code_audit.py`
- **Purpose**: stdlib-only dead-code audit catching unused imports (F401), unused local variables (F841), and orphan modules. Uses Python AST parsing. Reads YAML allowlist for known exemptions.
- **CLI arguments**: `--roots`, `--allowlist`, `--out`.
- **Output**: `results/dev/dead_code_report.json`.
- **Dependencies**: ast (stdlib only).

---

## 8. `scripts/mortfm/` -- MORT-FM Canonical Workflow (12 files, ~369 lines)

All numbered steps (00-09) are thin v18 canonical-workflow shims that delegate to underlying scripts via `runpy.run_path()`. `__init__.py` is empty.

#### `scripts/mortfm/00_download_public_data.py`
- **Delegates to**: `scripts/mortfm_download_public_data.py`
- **Stage**: Raw public-data acquisition (GEO, MMRF, BeatAML).

#### `scripts/mortfm/01_harmonize_identifiers.py`
- **Delegates to**: `scripts/mortfm_harmonize_identifiers.py`
- **Stage**: Identifier harmonization (STRING/HGNC bridge, drug aliases).

#### `scripts/mortfm/02_build_longitudinal_dataset.py`
- **Delegates to**: `scripts/mortfm_build_longitudinal_dataset.py`
- **Stage**: Build longitudinal patient pair dataset.

#### `scripts/mortfm/03_prepare_mmrf.py`
- **Delegates to**: `scripts/mortfm_prepare_mmrf.py`
- **Stage**: MMRF RNA/clinical preparation.

#### `scripts/mortfm/04_pretrain_foundation.py`
- **Delegates to**: `scripts/mortfm_pretrain_foundation.py`
- **Stage**: Block A foundation pretraining.

#### `scripts/mortfm/05_train_block_c_contrastive.py`
- **Delegates to**: `scripts/mortfm_train_block_c_contrastive.py`
- **Stage**: Block C SupCon contrastive stage training.

#### `scripts/mortfm/06_train_lens_resistance.py`
- **Delegates to**: `scripts/mortfm_train_lens_resistance.py`
- **Stage**: LENS resistance head training.

#### `scripts/mortfm/07_train_survival.py`
- **Delegates to**: `scripts/mortfm_train_survival.py`
- **Stage**: Discrete-time survival head training.

#### `scripts/mortfm/08_eval_causal_evidence_v2.py`
- **Delegates to**: `scripts/mortfm_causal_evidence_report_v2.py` (not yet wired; Phase 8 dependency).
- **Stage**: Causal evidence v2 report. Falls back to legacy v17 evaluation if available.

#### `scripts/mortfm/09_gate_revalidate.py`
- **Delegates to**: `scripts/mortfm_v17_gate_revalidate.py`
- **Stage**: v17/v18 gate revalidation (12 claim levels).

#### `scripts/mortfm/10_run_baselines.py`
- **Purpose**: Phase 10 baseline-runner skeleton. CLI orchestrator that runs registered baselines on same splits as training scripts. Currently scaffolding-only (`--dry-run` mode only; real fitting raises NotImplementedError).
- **CLI arguments**: `--config`, `--baselines`, `--splits`, `--out-parquet`, `--seeds`, `--dry-run`.
- **Dependencies**: `resistancemap.baselines.registry.REGISTRY`, imports `resistancemap.baselines.{deep_static, sequence, static_ml}`.

#### `scripts/mortfm/__init__.py`
- Empty file (package marker).

---

## 9. `paper/v8_artifacts/v11_sprint5/mmsygnal/` -- Paper Figure/Table Generation (2 files, ~259 lines)

#### `paper/v8_artifacts/v11_sprint5/mmsygnal/score_and_evaluate.py`
- **Purpose**: Apply mmSYGNAL subtype-aware routing and compute head-to-head C-index vs v11 Cox. Routes using A>B>C grade priority (t(4;14)[A], amp(1q)[B], del(13)[B], agnostic[C] fallback). Evaluates per-stratum + marginal Harrell C-index on same 5-stratum partition as v11.
- **Output**: `paper/v8_artifacts/v11_sprint5/mmsygnal/head_to_head_results.json`.
- **Dependencies**: lifelines.concordance_index.

#### `paper/v8_artifacts/v11_sprint5/mmsygnal/bootstrap_ci.py`
- **Purpose**: Bootstrap 95% CI for mmSYGNAL marginal C-index on real 787-patient MMRF cohort. Resamples patient indices (not synthetic data) via sklearn.utils.resample. Also computes per-stratum bootstrap CIs.
- **Output**: `paper/v8_artifacts/v11_sprint5/mmsygnal/bootstrap_ci.json`.

---

## Cross-Cutting Dependencies

| Module | Used by |
|---|---|
| `resistancemap.landscape.scalar_potential` | s1d, s1e, s2d, s7, s1f(v11), s7(v11) |
| `resistancemap.landscape.rwr_propagation` | s3c, s3d, s3e, s3f, s3fix_*, s6b, s3_gat(v11) |
| `resistancemap.landscape.neural_ode_flow` | s1f(v11), s7(v11) |
| `resistancemap.landscape.gat_propagation` | s3_gat(v11) |
| `resistancemap.mortfm.*` | archive scripts, mortfm shims |
| `resistancemap.baselines.registry` | 10_run_baselines.py |
| `lifelines.CoxPHFitter` | s4d, s4e_*, s5d(v11), s5e(v11), s5f(v11), v12 scripts |
| `numpyro` / `jax` | s2e, s2f, s2f_v11, v12_hbayes_ppc |
| `sklearn` | s1c, s3prep, s5, s5_v11, s5c/d(v11), baselines |

## Falsification Test Index

| Test | Script(s) | Description | Threshold |
|---|---|---|---|
| F1 | s2f, s2f_v11 | Cross-stratum HBayes coverage | 90% CI coverage >= 70% |
| F2 | s2f, s2f_v11 | Baseline-ablation log-lik | 95% CI of delta excludes 0, delta > 0 |
| F3 | s3d, s3fix_*, s3fix_v2, s6b, s6b_v2 | CRISPR/drug-response rank-sum oracle | >= 4/10 Bonferroni-pass |
| F4 | s1e | CurlFraction (Helmholtz-Hodge) | <= 0.30 |
| F5 | s1e, s2d | Median-r identifiability | < 0.9 |
| F6 | s3e, s3fix_*, s3fix_v2, s3fix_v3, s3_gat(v11) | MM driver gene recall | permutation p < 0.001 |
| F7 | s3f, s3fix_*, s3fix_v3_f7, s3_fix_f7_prism | Multi-seed pooling improvement | pooled p <= single p |
| F8 | s4d/s4e_v2, s7, s7_v11 | NIE CI excludes zero / Energy distance | CI excludes 0 / MWU p < 0.05 |
| F9 | s4e, s4e_v2 | Negative-control NIE | < 10% of neg trials reach observed |
| F10 | s4d/s4e_v2, s4_fix_* | E-value sensitivity | E-value at CI bound >= 1.5 |
| F_S5 | s5, s5_v11 | Mondrian conformal coverage | >= 3 strata within +/-3%, all within +/-5% |

---

## Appendix J: Test Suite

## Test Suite

### Pytest Configuration

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:
- **testpaths**: `["tests"]`
- **python_files**: `["test_*.py", "*_test.py"]`
- **addopts**: `-v --cov=resistancemap --cov-report=term-missing`
- **markers**: `unit`, `integration`, `slow`, `gpu`
- **coverage source**: `resistancemap` (omits `tests/`, `__main__.py`, `setup.py`)

The `tests/mortfm/__init__.py` docstring: "MORT-FM (v15) tests. These exercise the data contract, encoders, fusion, dynamics, heads, losses, trainer, and leakage / patient-disjoint splits."

---

### Top-Level Tests (`tests/`)

#### `tests/test_agents.py`
- **Modules tested**: `resistancemap.agents.base` (AgentState, AgentResult, BaseAgent), `resistancemap.agents.orchestrator` (AgentDAG, Orchestrator), `resistancemap.agents.specialized` (DataValidationAgent, DataPrepAgent, VAEPretrainAgent, ValidationAgent), `resistancemap.config.ResistanceMapConfig`
- **Fixtures**: `MockAgent` (subclass of BaseAgent with trivial execute)
- **Test classes and methods**:
  - `TestHashComputation` (unit):
    - `test_hash_dict` -- deterministic SHA for identical dicts, different for changed values
    - `test_hash_list` -- same for lists
    - `test_hash_tensor` -- same for PyTorch tensors
    - `test_hash_string` -- same for strings
    - `test_hash_none` -- None hashes consistently
  - `TestAgentResult` (unit):
    - `test_agent_result_creation` -- fields populated, error is None
    - `test_agent_result_to_dict` -- serialisation preserves agent_name, status string, hash, metadata
  - `TestMockAgentExecution` (unit, async):
    - `test_agent_execute` -- MockAgent.execute sets flag, returns COMPLETED
    - `test_agent_safe_execute` -- _safe_execute records non-empty verification_hash
    - `test_agent_input_validation` -- non-dict input causes FAILED status with "validation failed" error
  - `TestAgentDAG` (unit):
    - `test_dag_add_agent` -- agents and edges registered
    - `test_dag_duplicate_agent` -- raises ValueError("already registered")
    - `test_dag_topological_sort` -- 3-layer DAG produces correct parallel layers
    - `test_dag_cycle_detection` -- raises ValueError("Cycle detected")
    - `test_dag_missing_dependency` -- raises ValueError("unknown agent")
    - `test_dag_validate` -- valid DAG returns (True, "")
  - `TestOrchestrator` (unit + async):
    - `test_orchestrator_creation` -- empty dag, results, verification chain
    - `test_orchestrator_add_agents` -- agents appear in dag
    - `test_orchestrator_prepare_execution` -- valid plan with correct layer count
    - `test_orchestrator_get_execution_plan` -- parallel agents in layer 0, dependent in layer 1
    - `test_orchestrator_run` -- both agents COMPLETED
    - `test_orchestrator_verification_chain` -- chain length == 2, each hash is 64 hex chars
    - `test_orchestrator_timing_report` -- timing dict has start_time, end_time, elapsed_seconds >= 0
    - `test_orchestrator_results_summary` -- total_agents, status_counts, empty failed_agents
  - `TestSpecializedAgents` (unit, async):
    - `test_data_validation_agent` -- name, empty deps, execute returns result
    - `test_data_prep_agent` -- name, depends on data_validation, verify_inputs fails without dep output
    - `test_vae_pretrain_agent` -- name, depends on data_prep
  - `TestIntegration` (integration, async):
    - `test_three_layer_pipeline` -- 3-agent serial pipeline, all COMPLETED, chain length 3
    - `test_parallel_pipeline` -- 2 parallel + 1 dependent, plan has 2 layers with 2 in first

---

#### `tests/test_evaluation_baselines.py`
- **Modules tested**: `resistancemap.evaluation.baselines.base.Baseline`, `.clinical_only.ClinicalOnlyBaseline`, `.transcriptome_only.TranscriptomeOnlyBaseline`, `.last_observation.LastObservationBaseline`, `.simple_survival.SimpleSurvivalBaseline`
- **Skip condition**: `pytest.importorskip("resistancemap.evaluation")` and `resistancemap.evaluation.baselines`
- **Parametrized over**: `ALL_BASELINE_CLASSES` (4 classes)
- **Tests** (all parametrized, unit):
  - `test_baseline_subclasses_baseline_abc` -- issubclass(cls, Baseline)
  - `test_baseline_has_required_methods` -- callable fit, predict
  - `test_baseline_has_name_attribute` -- non-empty .name string
  - `test_baseline_fit_rejects_none_inputs` -- fit(None, None) raises ValueError/TypeError
  - `test_baseline_fit_rejects_empty_inputs` -- fit(empty_array, empty_array) raises ValueError/TypeError

---

#### `tests/test_evaluation_charter_and_rubric.py`
- **Modules tested**: `resistancemap.evaluation` (load_default_charter, Rubric)
- **Skip condition**: `pytest.importorskip("resistancemap.evaluation")`
- **Tests** (unit):
  - `test_default_charter_has_three_claims` -- charter.claims has exactly 3 with names {state, when, pathway}
  - `test_charter_documents_prospective_epistemic_limit_for_when_claim` -- epistemic_limits text mentions "prospective" or "time-ordered"
  - `test_every_claim_has_at_least_one_falsifier` -- each claim has a non-empty falsifiers list
  - `test_rubric_score_applies_weights_multiplicatively` -- weighted score for data_adequacy agent: (1.0*2.0 + 0.0*0.5)/(2.0+0.5) = 0.8
  - `test_rubric_yaml_roundtrip_preserves_structure` -- to_yaml then from_yaml produces identical dict or identical scores

---

#### `tests/test_evaluation_leakage.py`
- **Modules tested**: `resistancemap.evaluation.leakage` (audit_patient_split, detect_temporal_leakage)
- **Skip condition**: `pytest.importorskip("resistancemap.evaluation")`
- **Tests** (unit):
  - `test_audit_patient_split_detects_overlap` -- P002 in train+val, P003 in train+test detected, report.clean is False
  - `test_audit_patient_split_clean_split_is_clean` -- disjoint sets produce report.clean=True
  - `test_detect_temporal_leakage_flags_train_after_test` -- train timestamp 100 > test timestamp 80 for P001 flagged, clean=False
  - `test_detect_temporal_leakage_clean_split` -- correctly time-ordered split produces clean=True

---

#### `tests/test_evaluation_metrics.py`
- **Modules tested**: `resistancemap.evaluation.metrics.calibration` (expected_calibration_error), `.survival` (integrated_brier_score, concordance_index), `.forecasting` (pinball_loss), `.pathway_stability` (attribution_overlap), `.shift` (maximum_mean_discrepancy)
- **Skip condition**: `pytest.importorskip("resistancemap.evaluation")`
- **Tests** (unit, known-answer):
  - `test_ece_perfectly_calibrated_is_zero` -- 20 predictions in 2 bins, empirical frequency matches predicted prob => ECE==0
  - `test_ece_worst_case_is_one` -- predict 1.0 when label is 0 => ECE==1.0
  - `test_integrated_brier_score_known_values` -- 5-sample fixture, IBS == 0.04
  - `test_pinball_loss_at_median_is_half_mean_abs_error` -- tau=0.5, verified against 0.5*mean(|y-q|)
  - `test_attribution_overlap_identical_inputs_is_one` -- same gene list => 1.0
  - `test_attribution_overlap_disjoint_inputs_is_zero` -- disjoint gene lists => 0.0
  - `test_mmd_same_distribution_is_near_zero` -- identical samples => MMD ~0.0
  - `test_concordance_index_perfect_ranking_is_one` -- correct hazard ordering => c=1.0
  - `test_concordance_index_anti_ranked_is_zero` -- inverted hazard => c=0.0

---

#### `tests/test_evaluation_orchestrator.py`
- **Modules tested**: `resistancemap.evaluation` (EvalAgent, EvalFinding, Verdict, EvalOrchestrator, EvalReport), `resistancemap.agents.base` (AgentResult, AgentState), `resistancemap.config.ResistanceMapConfig`
- **Skip condition**: `pytest.importorskip("resistancemap.evaluation")`
- **Fixtures**: `FakeEvalAgent` (EvalAgent subclass with configurable tier, verdict, delay, score; records wall-clock start/end timestamps)
- **Tests** (async, integration):
  - `test_orchestrator_executes_tiers_in_order` -- A->B->C->D, each tier finishes before next starts
  - `test_orchestrator_hard_stops_on_tier_a_failure` -- a_fail FAIL => b1/c1/d1 BLOCKED, never executed (start_ts is None)
  - `test_orchestrator_parallelizes_within_tier` -- two Tier A agents overlap in wall-clock time, elapsed < 2*delay
  - `test_orchestrator_eval_report_per_tier_verdicts_populated` -- per_tier_verdicts has keys A/B/C/D, all Verdict instances
  - `test_orchestrator_verification_chain_length_matches_executed_agents` -- chain length == 5, each is 64-char hex
  - `test_orchestrator_timing_recorded_for_executed_agents` -- timing dict has entries for a1 and b1

---

#### `tests/test_leakage.py`
- **Modules tested**: `resistancemap.data.splits` (make_split, assert_no_overlap, assert_folds_disjoint, LeakageError)
- **Fixtures**: `synthetic_meta` -- DataFrame with 600 rows, 50 patients, 4 lineage subtypes, 11 drugs (RNG seeded)
- **Tests** (unit):
  - `test_patient_split_folds_disjoint` -- 5-fold patient split, folds are patient-disjoint
  - `test_lineage_split_folds_disjoint` -- 4-fold lineage split, folds are lineage-disjoint
  - `test_drug_holdout_generalisation` -- 5-fold drug holdout, no shared drugs between train/test per fold
  - `test_pretrain_eval_cohort_disjoint_happy` -- disjoint patient IDs pass assert_no_overlap
  - `test_pretrain_eval_cohort_disjoint_raises` -- overlapping patient IDs raise LeakageError
  - `test_bad_kind_rejected` -- make_split(kind="sample") raises ValueError
  - `test_split_manifest_reproducible` -- same seed => same group_hash and fold_sizes
  - `test_esm2_training_contamination_placeholder` -- pytest.skip placeholder for ESM-2 finetuning corpus check

---

#### `tests/test_metrics_and_calibration.py`
- **Modules tested**: `resistancemap.evaluation.metrics.per_drug` (per_drug_metrics, pooled_with_ci, require_per_drug_with_pooled), `resistancemap.evaluation.calibration` (empirical_coverage, interval_score, conformal_quantile, conformal_trajectory_intervals)
- **Tests** (unit):
  - `test_per_drug_returns_row_per_drug` -- 200 samples across 3 drugs, each drug gets a row
  - `test_per_drug_flags_insufficient_samples` -- fewer than min_each_class=3 positives => "insufficient" note
  - `test_pooled_ci_narrows_with_n` -- CI width for n=5000 < CI width for n=50
  - `test_require_per_drug_with_pooled_contract` -- returns dict with "pooled" and "per_drug" keys
  - `test_empirical_coverage_matches_nominal_when_well_calibrated` -- intervals always contain y => coverage=1.0
  - `test_interval_score_penalises_miscoverage` -- missed intervals score higher than covering intervals
  - `test_conformal_quantile_monotone_in_alpha` -- tighter alpha (0.1) => larger quantile than looser (0.3)
  - `test_conformal_intervals_shape` -- output lo/hi have same shape as pred, hi >= lo

---

#### `tests/test_ml_integration.py`
- **Modules tested**: `resistancemap.models.vae` (ProteomeToEpigenomeVAE, GradientReversalLayer), `resistancemap.models.trajectory` (SinkhornOT, ChromatinODE, StabilityConfig), `resistancemap.models.fusion` (CrossModalFusionNet), `resistancemap.landscape.predictor` (ResistanceLandscape, EvidentialResistanceHead), `resistancemap.inference.pipeline` (ResistanceMapPipeline), `resistancemap.config.VAEConfig`
- **Tests** (integration, CPU):
  - `test_vae_training_loss_decreases` -- 10 epochs on random data, loss[0] > loss[-1] and loss[0] > loss[4]
  - `test_vae_encode_decode_shapes` -- encode returns (B, latent_dim) mu/logvar; decode returns (B, epigenome_dim); full forward returns all three
  - `test_gradient_reversal_actually_reverses` -- GRL negates gradients: grad_reversed == -grad_normal
  - `test_sinkhorn_ot_transport_plan` -- transport plan shape (m, n), non-negative, row sums ~1, positive total mass; interpolation shape (m, d)
  - `test_landscape_predictor_forward_pass` -- output dict has drug_resistance (B, n_drugs*n_timepoints), resistance_state (B, n_states), transition_matrix (B, n_states, n_states), target_scores
  - `test_landscape_predictor_batch_vs_single` -- batch prediction numerically equals loop of single predictions (atol=1e-5)
  - `test_fusion_cross_attention` -- 4-modality CrossModalFusionNet outputs (B, output_dim) fused tensor + attention weights dict
  - `test_fusion_missing_modality_handling` -- one all-zero modality does not crash, output is finite
  - `test_chromatin_ode_integration` -- 10-step Euler integration stays finite, output shape (B, state_dim)
  - `test_evidential_head_uncertainty` -- alpha > 0, probs sum to 1, epistemic > 0, aleatoric is finite
  - `test_pipeline_end_to_end` -- ResistanceMapPipeline created with landscape model, forward produces valid outputs, pipeline has correct drug/protein/state names
  - `test_vae_stochastic_decoder` -- stochastic decoder mode returns (recon_mean, recon_logvar, mu, log_var) with correct shapes

---

#### `tests/test_observability_instrumentation.py`
- **Modules tested**: `resistancemap.observability.instrumentation` (Recorder, TraceContext, get_recorder, record_guardrail_violation, record_handoff, record_tool_call, set_recorder)
- **Fixture**: `recorder(tmp_path)` -- installs fresh Recorder with run_id="smoketest_run" at temp dir, tears down after
- **Budget constraint**: < 2 seconds, no network, no GPU, no model loading
- **Tests** (smoke):
  - `test_helpers_are_inert_when_no_recorder` -- record_* with no recorder creates no files, no raise
  - `test_records_real_events_and_schema_matches` -- records 7 events (1 trace_context + 3 handoffs + 2 tool_calls + 1 guardrail_violation); validates top-level keys {schema_version, run_id, event_type, recorded_at, payload}; validates per-type payload keys; verifies event counts; spot-checks from_agent values, succeeded booleans, rule+severity content
  - `test_jsonl_is_append_only` -- second batch grows file size, total line count == 3

---

#### `tests/test_ope_and_guardrails.py`
- **Modules tested**: `resistancemap.models.rl.ope` (weighted_importance_sampling, doubly_robust, positivity_check, evaluate), `resistancemap.infrastructure.guardrails` (adjudicate, GUARDRAILS)
- **Tests** (unit):
  - `test_wis_matches_on_policy` -- when behaviour==target policy, WIS estimate matches mean reward within 0.05
  - `test_positivity_check_flags_low_prob` -- log_prob(0.001) flagged at min_prob=0.01
  - `test_evaluate_fails_on_positivity_violation` -- very low behaviour probs => result.passed=False, "positivity" in notes
  - `test_evaluate_passes_when_agreed` -- matched policies and rewards => result.passed=True
  - `test_probability_range_rule` -- scores in [0,1] pass; negative score triggers "probability_range"
  - `test_stochastic_ci_rule_blocks_degenerate_interval` -- zero-width CI triggers "stochastic_ci"
  - `test_cohort_access_tag_rule` -- controlled_access=True + public_safe=False triggers "cohort_access_tag"
  - `test_monotone_dose_rule` -- decreasing viability passes; increasing viability triggers "monotone_dose"

---

### MORT-FM Tests (`tests/mortfm/`)

#### `tests/mortfm/test_acceptance_gate.py`
- **Modules tested**: `resistancemap.mortfm.acceptance_gate` (AcceptanceError, AcceptanceReport, VALID_LEVELS, evaluate_cohort, require_level), `resistancemap.mortfm.schemas` (DrugContext, ModalityTensor, MORTFMConfig, PatientCellSnapshot, ResistanceOutcome), `resistancemap.data.trajectory_pair_builder.build_temporal_pairs`
- **Helpers**: `_snap(pid, t)` builds PatientCellSnapshot with rna(4)+proteomics(3); `_cohort(n_patients, events)` builds n_patients with 2 timepoints each
- **Tests** (unit):
  - `test_unknown_level_raises` -- ValueError for level="paper-ready"
  - `test_empty_cohort_fails_all_levels` -- empty list fails all VALID_LEVELS
  - `test_5_patients_fails_debug` -- 5 patients below debug threshold, blocking reason mentions "debug threshold"
  - `test_15_patients_passes_debug_but_fails_research` -- 15 patients pass debug, fail research; level_achieved in {debug, technical}
  - `test_require_level_raises_on_block` -- require_level raises AcceptanceError with passed=False
  - `test_strong_level_always_blocks_by_default` -- even 250-patient cohort cannot auto-pass "strong" level
  - `test_report_dataclass_carries_metrics` -- AcceptanceReport has n_unique_patients=20, n_modalities>=1, has_survival_labels=True, has_drug_response_labels=False

---

#### `tests/mortfm/test_baseline_registry.py`
- **Modules tested**: `resistancemap.baselines.registry` (REGISTRY, BaselineModel), `resistancemap.baselines.static_ml`, `resistancemap.baselines.deep_static`, `resistancemap.baselines.sequence`
- **Fixtures**: `synthetic_static_xy` (N=24, F=10 Gaussian, deterministic linear y); `synthetic_sequence_xy` (N=20, T=3, F=6 visits)
- **Tests**:
  - Registry-level:
    - `test_registry_lists_static_ml_baselines` -- elasticnet, ridge, random_forest registered
    - `test_registry_lists_deep_static_baselines` -- mlp, cnn_feature_order, ft_transformer registered
    - `test_registry_lists_sequence_baselines` -- lstm_visits, gru_visits registered
    - `test_registry_resolves_to_baseline_contract` -- every name resolves to object with fit/predict/predict_survival/save/load/name; model.name matches registry key
  - Determinism (parametrized):
    - `test_static_ml_determinism` [elasticnet, ridge, random_forest] -- same seed => bit-identical predictions
    - `test_deep_static_determinism` [mlp, cnn_feature_order, ft_transformer] -- same seed, 2 epochs => predictions match (rtol=1e-5)
    - `test_sequence_determinism` [lstm_visits, gru_visits] -- same seed => predictions match (rtol=1e-5)
  - Optional deps:
    - `test_xgboost_baseline_or_skip` -- importorskip xgboost, name matches
    - `test_lightgbm_baseline_or_skip` -- importorskip lightgbm, name matches
  - Save/load round-trip:
    - `test_static_ml_save_load_roundtrip` -- ridge save to joblib, reload, predictions match
    - `test_deep_static_save_load_roundtrip` -- mlp save to .pt, reload, predictions match (rtol=1e-5)

---

#### `tests/mortfm/test_canonical_forward.py`
- **Modules tested**: `resistancemap.mortfm.model.MORTFM`, `resistancemap.mortfm.schemas.MORTBatch`, `resistancemap.mortfm.schemas.MORTFMConfig`
- **Helper**: `_minimal_batch(d_latent, B)` -- zero-filled MORTBatch with rna+drug modalities
- **Tests** (unit):
  - `test_canonical_forward_returns_lens_dict` -- model(batch) returns dict with keys: z_traj, z_samples, hazard, survival_curve, basin_probs, hitting_cdf, hitting_mean_tau, hitting_frac_hit; z_traj shape (B, n_time_grid, 64); basin_probs first two dims (B, n_time_grid)
  - `test_forward_legacy_preserved` -- forward_legacy returns TrajectoryPrediction with z_path.shape[0] == n_time_grid
  - `test_forward_foundation_encode_only` -- forward_foundation returns FoundationState with z0 shape (B, 64)
  - `test_canonical_and_legacy_are_distinct_methods` -- forward is not forward_legacy; forward_lens attribute gone

---

#### `tests/mortfm/test_canonical_loss_router.py`
- **Modules tested**: `resistancemap.training.canonical_mortfm_losses` (REQUIRED_SUPERVISION_PER_STAGE, STAGE_LOSS_SPEC, assemble_canonical_loss, basin_transition_loss, canonical_trajectory_loss, hitting_time_loss, lens_survival_loss, sde_path_loss), `resistancemap.mortfm.canonical_trainer` (CanonicalMORTFMTrainer, MissingSupervisionError)
- **Helpers**: `_mk_outputs(B, T, d, K, n_basins)` -- synthetic forward outputs dict; `_mk_batch_with_all_supervision(B, d)` -- MORTBatch with event_time, event_observed, resistance_label, future_state; `_mk_batch_with_no_supervision(B)` -- MORTBatch with patient_ids only
- **Tests** (unit):
  - Individual losses -- n_supervised contract:
    - `test_lens_survival_loss_reports_n_supervised` -- n_supervised==4, finite loss, contains loss_survival_nll + loss_brier
    - `test_lens_survival_loss_returns_zero_when_unsupervised` -- n_supervised==0, loss==0.0
    - `test_hitting_time_loss_n_supervised_counts_observed_events_only` -- event_observed=[1,0,1,1] => n_supervised==3
    - `test_hitting_time_loss_returns_zero_when_no_observed` -- all censored => n_supervised==0, loss==0.0
    - `test_basin_transition_loss_counts_valid_labels` -- 4 labels < 5 basins => n_supervised==4
    - `test_basin_transition_loss_zero_when_no_label` -- no labels => n_supervised==0, loss==0.0
    - `test_sde_path_loss_is_always_finite` -- regularisation, n_supervised==B even without labels
    - `test_canonical_trajectory_loss_inv_dt_weighting` -- n_supervised==4, finite loss
    - `test_canonical_trajectory_loss_zero_on_nonpositive_dt` -- dt=[0, -1, NaN] => n_supervised==0, loss==0.0
  - Stage dispatcher (parametrized):
    - `test_assemble_canonical_loss_dispatches_per_stage` -- E: (trajectory, sde_path, basin_transition); F: (survival, hitting, hitting_nll, surv_hit_consistency); G: (basin_transition); H: (survival); each term has n_supervised + loss; loss_total is finite
    - `test_assemble_canonical_loss_rejects_unknown_stage` -- stage "Z" raises ValueError
    - `test_required_supervision_table_matches_stage_spec` -- every entry in REQUIRED_SUPERVISION_PER_STAGE points to a term in STAGE_LOSS_SPEC
  - Strict-mode trigger:
    - `test_trainer_strict_raises_when_required_term_unsupervised` -- strict trainer with stage F + empty batch raises MissingSupervisionError (uses _DummyModel stub)

---

#### `tests/mortfm/test_canonical_trainer_smoke.py`
- **Modules tested**: `resistancemap.mortfm.canonical_trainer.CanonicalMORTFMTrainer`, `resistancemap.mortfm.model.MORTFM`, `resistancemap.mortfm.data_module.make_data_module`, `resistancemap.mortfm.trainer.MORTFMTrainer`
- **Helpers**: `_mk_snap`, `_cohort(n=10)`, `_debug_cfg(checkpoint_dir)`, `_make_model_trainer(tmp_path, strict, seed)` -- builds tiny MORTFM + canonical trainer with deterministic seed
- **Tests** (smoke):
  - `test_losses_for_batch_returns_canonical_bundle` [parametrized: E, F, G, H] -- canonical-bundle has loss_total (finite scalar), components, per_term_n_supervised, outputs, required_term; outputs contain z_traj, hazard, survival_curve, basin_probs, hitting_cdf, hitting_mean_tau, t_grid
  - `test_canonical_trainer_never_calls_forward_legacy` -- monkeypatch spy on forward_legacy, drive all 4 stages, assert never called
  - `test_fit_stage_F_runs_one_epoch_and_records_supervision` -- fit_stage("F", 1) returns list of 1 metric dict with stage, epoch, n_batches, per_term_n_supervised["survival"] > 0
  - `test_strict_mode_raises_on_missing_supervision` -- wipe event_time/event_observed from batch, stage F raises MissingSupervisionError
  - Bug B6 wireup:
    - `test_with_canonical_factory_attaches_canonical_trainer` -- MORTFMTrainer.with_canonical sets canonical_trainer attribute, is CanonicalMORTFMTrainer instance
    - `test_with_canonical_stage_E_never_calls_forward_legacy` -- through with_canonical shim, stage E routes through canonical forward, never forward_legacy

---

#### `tests/mortfm/test_canonical_trainer_validate_for_loss.py`
- **Regression test for Bug B14**: validate_for_loss was never invoked by canonical trainer
- **Modules tested**: `resistancemap.mortfm.canonical_trainer` (CanonicalMORTFMTrainer, MissingSupervisionError), `resistancemap.mortfm.model.MORTFM`
- **Tests** (regression):
  - `test_strict_supervision_raises_when_event_time_missing` -- stage F + strict + missing event_time/event_observed => MissingSupervisionError(match="validate_for_loss")
  - `test_non_strict_skips_survival_term_without_raising` -- strict=False + missing labels => no raise, survival term n_supervised==0
  - `test_strict_supervision_alias_back_to_strict` -- strict_supervision=True maps to strict=True; False maps to False

---

#### `tests/mortfm/test_counterfactual_engine_deprecation.py`
- **Regression test for Bug B4**: WhatIfAnalyzer uses latent clamping (associational, not interventional)
- **Modules tested**: `resistancemap.interpretability.counterfactual_engine` (CounterfactualConfig, WhatIfAnalyzer)
- **Helpers**: `_TinyODEModel` -- minimal nn.Module with ode_func + predict
- **Tests** (unit):
  - `test_whatif_init_emits_deprecation_warning` -- __init__ emits DeprecationWarning matching "associational, not interventional"
  - `test_whatif_inhibit_pathway_emits_deprecation_warning` -- inhibit_pathway emits same DeprecationWarning (method-call may raise after warning, caught)

---

#### `tests/mortfm/test_edge_evidence_kofn.py`
- **Regression test for Bug B13**: causal-mechanism gate previously hard-AND'd 3 channels; now k-of-N with default k=2
- **Modules tested**: `resistancemap.mortfm.causal.edge_evidence_report` (DEFAULT_K_OF_N, N_EVIDENCE_CHANNELS, evaluate_edge_evidence_gate)
- **Tests** (unit):
  - `test_default_k_is_two` -- DEFAULT_K_OF_N==2, N_EVIDENCE_CHANNELS==3
  - `test_two_channels_supporting_passes_at_k2` -- 2/3 passes; channels_supported == {crispr, drug_target}
  - `test_two_channels_supporting_fails_at_k3` -- 2/3 fails at k=3
  - `test_one_channel_supporting_fails_at_k2` -- 1/3 fails at k=2
  - `test_one_channel_supporting_passes_at_k1` -- 1/3 passes at k=1
  - `test_all_three_pass_at_k3` -- 3/3 passes, channels_not_supported==[]
  - `test_all_three_fail_at_k1` -- 0/3 fails even at k=1

---

#### `tests/mortfm/test_endpoint_validator.py`
- **Modules tested**: `resistancemap.governance.endpoint_validator` (EndpointSemanticsError, VALID_ENDPOINTS, require_resistance_endpoint, require_trajectory_endpoint, validate_endpoint_semantics), `resistancemap.governance.claim_gates` (RunEvidence, run_gates)
- **Tests** (unit):
  - `test_unknown_endpoint_raises` -- ValueError for "time_to_grant_funding"
  - `test_os_blocks_resistance_and_trajectory` -- overall_survival: survival allowed, resistance + trajectory blocked; require_resistance_endpoint raises
  - `test_pfs_allows_resistance_with_warning` -- progression_free_survival: resistance + trajectory allowed; blocking_reasons mentions "PFS proxies resistance only when explicitly framed"
  - `test_drug_resistance_label_requires_longitudinal_for_trajectory` -- n_longitudinal_pairs=10: resistance yes, trajectory no; n=200: both yes
  - `test_auc_is_static_drug_response` -- claim_level=="static_drug_response", drug_response allowed, resistance + trajectory blocked
  - `test_all_valid_endpoints_have_a_claim_level` -- every endpoint has claim_level in known set
  - `test_claim_gates_block_os_resistance_combo` -- OS endpoint blocks resistance even when all other gates pass; output JSON written

---

#### `tests/mortfm/test_eval_causal_script_stub.py`
- **Regression test for Bug B5**: 08_eval_causal_evidence_v2.py exited code 2 instead of 1 when target missing
- **Tests** (smoke, subprocess):
  - `test_script_exists` -- scripts/mortfm/08_eval_causal_evidence_v2.py exists
  - `test_script_help_exits_with_code_0_or_1_never_2` -- --help returns 0 (target exists) or 1 (missing with Phase-8 hint), never 2
  - `test_actionable_message_when_target_missing` -- copy shim to sandbox without target; exits 1; stderr has "Phase 8" and "not yet wired" or "phase_8_counterfactual.md"

---

#### `tests/mortfm/test_identifier_mapping.py`
- **Modules tested**: `resistancemap.data.identifier_mapping` (load_identifier_maps, unify_identifiers)
- **Helper**: `_write_maps(tmp_path)` -- writes 5 TSV files (HGNC->Ensembl, UniProt xref, STRING alias, Reactome membership, ChEMBL target)
- **Tests** (unit):
  - `test_missing_file_raises` -- missing HGNC TSV raises FileNotFoundError
  - `test_round_trip_lookups` -- hgnc_to_ensembl["CD38"]=="ENSG0001", ensembl_to_hgnc, uniprot_to_hgnc, string_to_uniprot
  - `test_harmonise_handles_all_namespaces` -- HGNC symbol, Ensembl ID, UniProt ID, STRING ID all resolve to correct HGNC symbol; unknown returns None
  - `test_pathways_and_drug_targets` -- CD38 in pathways R1+R2; CHEMBL_D1 targets P28907+Q02223
  - `test_unify_identifiers_returns_none_for_unmapped` -- CD38->CD38, P28907->CD38, GIBBERISH->None

---

#### `tests/mortfm/test_intervention_graph_determinism.py`
- **Regression test for Bug B3**: per-edge basis seed used Python hash() which is process-random; fix uses SHA-256
- **Modules tested**: `resistancemap.mortfm.causal.intervention_graph` (_deterministic_edge_seed, deterministic_edge_basis)
- **Tests** (unit + subprocess):
  - `test_same_process_same_eid_returns_equal_seed` -- two calls same edge ID => same seed
  - `test_same_process_same_eid_returns_equal_basis` -- two calls same edge ID => torch.equal basis vectors
  - `test_different_eid_returns_different_seed` -- different edge IDs => different seeds
  - `test_subprocess_returns_same_seed_as_parent` -- spawn child process, compare integer seed; fails with old hash() implementation
  - `test_subprocess_returns_same_basis_as_parent` -- spawn child, compare float basis vector values (atol=1e-9)

---

#### `tests/mortfm/test_manifest_schemas.py`
- **Modules tested**: `resistancemap.data.manifest_schemas` (load_dataset_manifest, load_feature_manifest, load_public_data_manifest, load_samples_table)
- **Helper**: `_write_samples(tmp_path, df)` -- writes CSV
- **Tests** (unit):
  - `test_missing_file_raises` -- FileNotFoundError for all three loaders
  - `test_samples_missing_required_column_raises` -- ValueError("missing required columns")
  - `test_samples_duplicate_sample_id_raises` -- ValueError("sample_id must be unique")
  - `test_samples_invalid_split_raises` -- ValueError("invalid values") for split_group="INVALID"
  - `test_samples_patient_disjoint_violation_raises` -- ValueError("Patient-disjoint") when same patient in train+test
  - `test_samples_valid_loads` -- 6 samples, modality_coverage rna=1.0, proteomics=2/6, atac present
  - `test_feature_manifest_load` -- 2 RNA features, map_to_hgnc ENSG1->CD38, unknown->None
  - `test_dataset_manifest_load` -- 2 datasets, datasets_by_modality("drug_response") returns 1

---

#### `tests/mortfm/test_mortfm_losses.py`
- **Modules tested**: `resistancemap.training.mortfm_losses` (reconstruction_loss, drug_response_loss, survival_loss, resistance_state_loss, trajectory_distribution_loss, contrastive_alignment_loss, landscape_regularization_loss, counterfactual_consistency_loss, assemble_total_loss, calibration_loss, masked_modality_loss, pathway_attribution_loss, perturbation_consistency_loss), `resistancemap.landscape.potential.WaddingtonPotential`, `resistancemap.legacy.v15_planned_mortfm.models.heads.time_to_resistance_head` (CoxSurvivalHead, DiscreteTimeSurvivalHead)
- **Tests** (unit):
  - `test_reconstruction_gaussian` -- Gaussian recon loss is finite
  - `test_reconstruction_nb_with_real_counts` -- Negative binomial with Poisson counts is finite and > 0
  - `test_drug_response_loss_skips_nan_targets` -- NaN target skipped, 3 perfect => loss==0.0
  - `test_cox_survival_loss_handles_censoring` -- all censored => loss==0.0
  - `test_discrete_survival_loss_runs_with_some_events` -- 3 events, 4-bin discrete head, finite loss > 0
  - `test_resistance_state_loss_ignores_negative_targets` -- target -1 ignored, 3 valid rows, finite
  - `test_trajectory_mmd_zero_for_identical_distributions` -- MMD(x, x)==0.0
  - `test_contrastive_alignment_perfect_correspondence` -- x aligned with x at temp=0.01 => loss < 0.1
  - `test_landscape_regulariser_returns_finite_value` -- WaddingtonPotential(8, 3) regulariser is finite
  - `test_counterfactual_consistency_zero_when_rankings_match` -- matched rankings => loss < 1.0
  - `test_assemble_total_loss_skips_zero_weight` -- weight=0 excluded, missing component excluded, total==2.0
  - `test_calibration_loss_zero_for_perfect_calibration` -- ~0 ECE for well-calibrated predictions, finite

---

#### `tests/mortfm/test_mortfm_trainer_smoke.py`
- **Modules tested**: `resistancemap.mortfm.model.MORTFM`, `resistancemap.mortfm.trainer.MORTFMTrainer`, `resistancemap.mortfm.data_module.make_data_module`, `resistancemap.data.trajectory_pair_builder.build_temporal_pairs`
- **Helpers**: `_mk_snap`, `_cohort(n=10)` with 2 timepoints per patient, `_debug_cfg` -- tiny MORTFMConfig
- **Tests** (smoke, end-to-end):
  - `test_smoke_stage_A` -- 1 epoch of stage A, "recon" in components, train_loss > 0
  - `test_smoke_stage_F_survival` -- 1 epoch of stage F, "survival" or "state" in components
  - `test_smoke_stage_E_trajectory` -- 1 epoch of stage E, "trajectory" in components
  - `test_smoke_checkpoint_roundtrip` -- save after stage A, reload into new model, all parameters match
  - `test_smoke_full_forward_emits_all_heads` -- forward_legacy with compute_counterfactuals=True; z_path shape[0]==n_time_grid, resistance_state_logits (B, 4), pathway_protein_scores (B, 20), drug_specific_risk (B, 5), counterfactual_rankings not None

---

#### `tests/mortfm/test_patient_holdout_split.py`
- **Modules tested**: `resistancemap.mortfm.data_module` (split_patients, make_data_module), `resistancemap.data.trajectory_pair_builder.build_temporal_pairs`
- **Tests** (unit):
  - `test_split_is_patient_disjoint` -- 20-patient split: train, val, test patient sets have no intersection
  - `test_split_is_seed_deterministic` -- same seed => same train patients; different seed => different
  - `test_followup_snapshots_go_with_baselines` -- x_t_delta always has same patient_id as x_t (no cross-patient leakage)
  - `test_dataloader_creation_does_not_crash_on_minimum_cohort` -- 8 patients, batch_size=2, loaders iterate without exception

---

#### `tests/mortfm/test_phase1_trajectory_target.py`
- **v19 Phase 1 tests**: latent trajectory target
- **Modules tested**: `resistancemap.data.mortfm_dataset.mort_collate`, `resistancemap.training.canonical_mortfm_losses.latent_future_state_loss`, `resistancemap.mortfm.canonical_trainer.CanonicalMORTFMTrainer`, `resistancemap.mortfm.model.MORTFM`
- **Tests**:
  - Collate-fix:
    - `test_mort_collate_populates_future_snapshot_batch_when_paired` -- paired snapshots produce future_snapshot_batch (MORTBatch) + delta_t_days (finite, positive)
    - `test_mort_collate_does_not_fabricate_future_state_from_raw_rna` -- future_state is None even when x_t_delta is populated (raw-RNA fallback deleted)
    - `test_mort_collate_future_snapshot_batch_is_none_when_no_followup` -- static pairs produce None future_snapshot_batch + None delta_t_days
  - latent_future_state_loss:
    - `test_latent_future_state_loss_shape_mismatch_raises` -- mismatched d raises ValueError("same encoder")
    - `test_latent_future_state_loss_zero_when_predicted_equals_target` -- l2==0.0, mmd < 1e-5
    - `test_latent_future_state_loss_masks_nan_and_nonpositive_dt` -- dt=[NaN, -1, 0, 30] => n_supervised==1
    - `test_latent_future_state_loss_mini_overfit_decreases_over_5_steps` -- SGD on fixed target, loss monotonically decreases
    - `test_latent_future_state_loss_mmd_gated_on_min_batch` -- B < mmd_min_batch => mmd==0.0; with mmd_min_batch=2, mmd >= 0.0
  - _trajectory_step:
    - `test_trajectory_step_returns_zero_when_no_followup` -- n_supervised==0, loss==0.0
    - `test_trajectory_step_supervises_n_equals_batch_when_followup_present` -- n_supervised==batch_size, loss.requires_grad
  - Bug B1 axis-order:
    - `test_bug_b1_canonical_z_traj_terminal_slice_returns_b_d` -- z_traj (B, T, d_latent); z_traj[:, -1, :] is (B, d_latent)
    - `test_bug_b1_legacy_trajectory_mean_terminal_slice_returns_n_d` -- trajectory_mean is (T, N, d_latent); [-1] is (N, d_latent)

---

#### `tests/mortfm/test_phase7_canonical_grid.py`
- **v19 Phase 7 tests**: shared canonical_time_grid + survival/hitting consistency
- **Modules tested**: `resistancemap.mortfm.trajectory.grid` (CanonicalTimeGridConfig, canonical_time_grid, time_grid_meta), `resistancemap.mortfm.trajectory.graph_energy_sde.GraphEnergyResistanceSDE`, `resistancemap.mortfm.survival.competing_risk_head.CompetingRiskHead`, `resistancemap.training.canonical_mortfm_losses` (hitting_time_nll, survival_hitting_consistency_loss), `resistancemap.mortfm.model.MORTFM`
- **Tests**:
  - canonical_time_grid:
    - `test_canonical_time_grid_default_shape_and_endpoints` -- shape (25,), [0.0, 12.0]
    - `test_canonical_time_grid_custom_shape_and_endpoints` -- (13,), [0.0, 6.0]
    - `test_canonical_time_grid_rejects_bad_args` -- negative t_max or n_steps=1 raises ValueError
    - `test_time_grid_meta_extracts_dt` -- t_max==10.0, n_steps==6, dt==2.0
  - SDE + survival head sharing:
    - `test_sde_and_survival_head_share_grid_shape` -- both have shape (25,), torch.equal
    - `test_sde_without_config_warns_and_uses_legacy_grid` -- DeprecationWarning, shape (8,), [0.0, 1.0]
    - `test_survival_head_without_config_warns_and_uses_legacy_n_bins` -- DeprecationWarning, n_bins==8
  - hitting_time_nll:
    - `test_hitting_time_nll_low_when_mass_concentrated_at_event_bin` -- n_supervised==2, NLL < 0.5
    - `test_hitting_time_nll_grows_when_mass_shifted_away_from_event_bin` -- misplaced mass gives higher NLL
    - `test_hitting_time_nll_censored_uses_survival_at_t` -- censored row: -log(1-F(t_censor)) == -log(0.8)
    - `test_hitting_time_nll_grid_length_mismatch_raises` -- ValueError("canonical_time_grid")
  - survival_hitting_consistency_loss:
    - `test_consistency_loss_zero_when_consistent` -- S(t) <= 1-F_hit(t) => loss==0.0
    - `test_consistency_loss_positive_when_inconsistent` -- S(t) > 1-F_hit(t) => loss > 0.0
    - `test_consistency_loss_with_competing_risk_correction_relaxes_bound` -- cif_other_events makes previously inconsistent case consistent
    - `test_consistency_loss_shape_mismatch_raises` -- ValueError("grids unaligned")
  - Bug B19 SDE/projector d_graph parity:
    - `test_b19_sde_and_projector_share_d_graph` -- SDE drift d_graph == projector out_features
    - `test_b19_canonical_forward_runs_end_to_end_with_shared_grid` -- end-to-end forward; z_traj (B, n_time_grid, d_latent); hitting_cdf + survival_curve shapes match; SDE and CR head share t_grid

---

#### `tests/mortfm/test_schemas.py`
- **Modules tested**: `resistancemap.mortfm.schemas` (DrugContext, MODALITY_ORDER, ModalityTensor, MORTBatch, MORTFMConfig, ModalityName, PatientCellSnapshot, ResistanceOutcome, TemporalTrainingPair)
- **Tests** (unit):
  - `test_modality_tensor_shape_validation` -- correct feature_names accepted; wrong count raises ValueError("feature dim is 5"); 4D tensor raises ValueError("must be 2-D")
  - `test_modality_tensor_mask_shape_validation` -- mismatched mask shape raises ValueError("mask shape")
  - `test_modality_order_is_stable` -- exact 8-modality tuple order: rna, atac, methylation, histone_ptm, proteomics, phosphoproteomics, clinical, drug
  - `test_patient_snapshot_available_modalities` -- rna-only snapshot lists [RNA]; adding drug adds DRUG
  - `test_resistance_outcome_label_methods` -- has_survival_label True when event_time set; has_future_state True when future_state set; both False otherwise
  - `test_temporal_pair_supervision_flag` -- has_any_supervision False initially; True after setting event_time
  - `test_mortbatch_validate_for_loss_rejects_missing_labels` -- "survival" needs event_time; "trajectory" needs future_state; "resistance_state" needs resistance_label; "drug_response" needs drug_response
  - `test_mortbatch_to_device_preserves_tensors` -- .to(cpu) preserves rna tensor + modality_mask device
  - `test_mortfmconfig_defaults` -- d_latent > 0; survival_kind in {cox, discrete, competing_risks}; dynamics_kind in {neural_ode, neural_sde, jump_sde}

---

#### `tests/mortfm/test_single_cell_preprocessing.py`
- **Modules tested**: `resistancemap.data.single_cell_preprocessing` (DEFAULT_DISEASE_STAGE_MAP, QCReport, _derive_patient_id, _stage_to_pseudotime, preprocess_scrna)
- **Skip condition**: `pytest.importorskip("anndata")` and `pytest.importorskip("scanpy")`
- **Helper**: `_toy_anndata(n_cells=200, n_genes=300)` -- Poisson counts with MT genes, 4 disease stages
- **Tests** (unit):
  - `test_refuses_non_count_input` -- non-integer X raises ValueError("raw counts")
  - `test_stage_to_pseudotime_mapping` -- MM->3.0, MGUS->1.0, healthy->0.0, SMM->2.0, MM_2->3.0 (prefix match), unknown->NaN
  - `test_derive_patient_id_strips_gsm_and_tail` -- GSM prefixes stripped, suffixes handled
  - `test_pipeline_adds_canonical_obs_and_uns` -- output has layers["counts"], obs columns patient_id/pseudotime_disease_stage/disease_subtype, all 4 pseudotime ordinals present, uns["mortfm_qc"] + uns["pseudotime_note"] contains "NOT calendar time"
  - `test_pipeline_preserves_counts_layer` -- X is log-normalised (< 50), counts layer still integer-like

---

#### `tests/mortfm/test_stage_supervision_strict.py`
- **Modules tested**: `resistancemap.mortfm.trainer` (MORTFMTrainer, MissingSupervisionError, STRICT_SUPERVISION_COVERAGE, StageSupervisionReport)
- **Helpers**: `_unsupervised_pairs(n)` -- cohort with event_time=None for everyone; include_drug_response_only=True
- **Tests** (unit):
  - `test_stage_F_raises_when_no_event_labels` -- train_epoch(stage="F") raises MissingSupervisionError on unsupervised cohort
  - `test_stage_A_does_not_raise_without_supervision` -- stage A is unsupervised, completes without error, returns metric.stage=="A"
  - `test_strict_thresholds_are_documented` -- STRICT_SUPERVISION_COVERAGE has E, F, G; F >= E; F >= G
  - `test_supervision_report_coverage_math` -- coverage = n_supervised_batches / n_batches; 7/10=0.7; 0/0=0.0

---

#### `tests/mortfm/test_temporal_pair_builder.py`
- **Modules tested**: `resistancemap.data.trajectory_pair_builder` (build_temporal_pairs, summarise_pairs)
- **Tests** (unit):
  - `test_empty_inputs_return_empty_list` -- ([], []) => []
  - `test_pairs_respect_allowed_time_gaps` -- 3 snapshots at 0/6/12, allowed_gap=6 => 2 longitudinal pairs ((0,6) and (6,12))
  - `test_outcome_outside_tolerance_skips_pair` -- outcome at baseline_time=0 matches t=0 pair but not others
  - `test_no_self_pairs` -- x_t.sample_id != x_t_delta.sample_id always
  - `test_no_cross_patient_pairs` -- P1 at t=0, P2 at t=6 => 0 longitudinal pairs
  - `test_survival_only_rows_emitted` -- single snapshot + outcome => 1 pair with x_t_delta=None
  - `test_unlabelled_rows_only_when_opt_in` -- include_unlabelled=False => 0; True => 1
  - `test_summarise_pairs_counts` -- n_unique_patients==2, n_longitudinal_pairs >= 1

---

### Dev Tests (`tests/dev/`)

#### `tests/dev/test_dead_code_audit.py`
- **Modules tested**: `scripts/dev/dead_code_audit.py` (via subprocess)
- **Tests** (smoke, subprocess):
  - `test_audit_runs_and_emits_report` -- runs script with --out and --repo-root, exit code 0, output JSON has keys: run_id, n_files_scanned, n_files_allowlisted, by_kind, findings, wall_time_s; by_kind histogram matches findings.kind counts
  - `test_audit_respects_allowlist` -- loads results/dev/dead_code_report.json (skips if not present); verifies scripts/mortfm/09_gate_revalidate.py, scripts/mortfm/__init__.py, resistancemap/__init__.py are NOT flagged

---

### Summary Statistics

| Directory | Files | Test Classes | Test Functions | Test Type |
|-----------|-------|-------------|----------------|-----------|
| `tests/` (top-level) | 10 | 9 classes + standalone | ~67 | Unit, integration, smoke, async |
| `tests/mortfm/` | 23 (+1 `__init__`) | ~5 classes + standalone | ~130 | Unit, regression, smoke, subprocess |
| `tests/dev/` | 1 | 0 | 2 | Smoke, subprocess |
| **Total** | **34 test files** | — | **~199 test functions** | — |

### Key Regression Tests (Bug IDs)
- **B1**: Axis-order mismatch in trajectory terminal slice (test_phase1_trajectory_target.py)
- **B3**: Hash randomization in edge basis vectors (test_intervention_graph_determinism.py)
- **B4**: WhatIfAnalyzer associational-not-interventional deprecation (test_counterfactual_engine_deprecation.py)
- **B5**: Script exit code 2 instead of 1 (test_eval_causal_script_stub.py)
- **B6**: MORTFMTrainer.with_canonical factory wireup (test_canonical_trainer_smoke.py)
- **B13**: Hard-AND causal gate replaced with k-of-N (test_edge_evidence_kofn.py)
- **B14**: validate_for_loss never invoked (test_canonical_trainer_validate_for_loss.py)
- **B19**: SDE and graph projector d_graph mismatch (test_phase7_canonical_grid.py)

