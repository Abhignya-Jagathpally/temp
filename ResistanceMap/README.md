# ResistanceMap

A reproducibility-first multi-omics machine-learning pipeline for **compound-prioritization screening** of MM-relevant cell lines against 11 GDSC drugs. Built on real CCLE proteomics + epigenomics + STRING PPI + GDSC drug-sensitivity data, with cryptographic boundary hashing, AgentOps observability, and Tier-A→D evaluation governance.

> **v15-mort-fm branch — MORT-FM research track.** A separate, *not-yet-trained* multi-modal foundation model (Multi-Omic Resistance Trajectory Foundation Model) lives on the `v15-mort-fm` branch under `resistancemap/mortfm/`. MORT-FM is a different scientific contribution from ResistanceMap v14: it targets patient-level longitudinal resistance trajectories using a graph-conditioned Neural SDE on a learned Waddington potential. Architecture, data contract, encoders, fusion, dynamics, landscape, heads, losses, trainer, configs, smoke test, and 38-test pytest suite have landed; **no patient-level metrics are claimable yet** because the model has not been trained on a real cohort. See `docs/MORTFM_ARCHITECTURE.md`, `docs/MORTFM_DATA_CONTRACT.md`, and `docs/MORTFM_LIMITATIONS.md`.

> **What ResistanceMap is.** A trained, calibrated, benchmarked tool that *ranks* cell lines by predicted drug resistance for **9 of 11 GDSC drugs** (82%), with documented failure modes on 2 HDAC-class drugs (Panobinostat, Romidepsin). All claims in this README are anchored to on-disk checkpoints and reproducible from `python main.py --config configs/default.yaml`.

> **What ResistanceMap is *not*.** It is **not** a patient-level treatment-response predictor (no patient training data has been ingested), **not** a longitudinal forecaster — training data consists of single-time-point cell-line snapshots; no time-ordered `(X_{t=0}, X_{t=Δ})` training pairs exist, so time-to-event and temporal-state claims are outside the model's training scope — and **not** a substitute for the IMWG response criteria in MM clinical decision-making. Earlier README versions overstated the scope; this v8-corrected README reports only what the actual code produces.

## Headline result (from `checkpoints/pipeline_validated.pt`, last run 2026-05-02)

```
Trained on:        622 CCLE cell lines × 11 GDSC drugs (3,635 observed IC50 cells)
Test:              132 cell lines, 736 observed (drug, cell-line) pairs
Aggregate test_mse:  2.3731   (NaN-masked, pooled across all drugs/cells)
Per-drug Spearman:   0.045 — 0.396 (n=11), median 0.328
                     (0.045 is Panobinostat — the documented failure; 9/11 drugs ≥ 0.305)
Actionable for screening:  9/11 drugs (82%)  ← Spearman ≥ 0.25 AND n_test ≥ 30
Well-calibrated for IC50:  8/11 drugs (73%)  ← additionally MSE < 1.0
Documented failure modes:  Panobinostat, Romidepsin (HDAC class)
Baseline rank:             4 / 11 in `paper/tables/baseline_comparison.md`
                           — beaten by Zero-predictor (2.3678) and Late-fusion (2.3653)
                           on aggregate, due to Panobinostat MSE=22.3 dragging the mean
```

The aggregate `test_mse` looks underwhelming because **one drug (Panobinostat) has MSE 22.3 and Spearman 0.045** — essentially noise. Removing Panobinostat from the average puts ResistanceMap firmly above the trivial baselines. We surface this honestly via the per-drug actionability matrix below rather than hiding it in the aggregate.

## Decision matrix — when to use ResistanceMap

| Drug | Class | n_test | MSE | Spearman | Screening | Calibrated | Verdict |
|---|---|---|---|---|---|---|---|
| Venetoclax | BCL2 | 68 | 0.010 | 0.328 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Bortezomib | Proteasome | 69 | 0.032 | 0.338 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Dinaciclib | CDK | 66 | 0.156 | 0.373 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Palbociclib | CDK | 69 | 0.197 | 0.305 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Cyclophosphamide | DNA-damage | 68 | 0.279 | 0.358 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Vorinostat | HDAC | 69 | 0.413 | 0.310 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Lenalidomide | IMiD | 69 | 0.499 | 0.396 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Etoposide | DNA-damage | 67 | 0.531 | 0.328 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Doxorubicin | DNA-damage | 69 | 1.666 | 0.339 | ✅ | — | Use for **screening only** (rank-meaningful; absolute IC50 unreliable) |
| Romidepsin | HDAC | 56 | 0.237 | 0.240 | ❌ | — | **DO NOT USE — failure mode documented** |
| Panobinostat | HDAC | 66 | 22.335 | 0.045 | ❌ | — | **DO NOT USE — failure mode documented** |

**Drug-class summary:**

| Class | Drugs | Actionable | Mean Spearman | Note |
|---|---|---|---|---|
| Proteasome | Bortezomib | 1/1 | 0.338 | strong |
| BCL2 | Venetoclax | 1/1 | 0.328 | strongest absolute MSE (0.010) |
| CDK | Dinaciclib, Palbociclib | 2/2 | 0.339 | strong |
| IMiD | Lenalidomide | 1/1 | 0.396 | best rank correlation |
| DNA-damage | Cyclophosphamide, Doxorubicin, Etoposide | 3/3 | 0.341 | useful for ranking |
| HDAC | Panobinostat, Romidepsin, Vorinostat | **1/3** | 0.198 | **class-level failure mode** |

Full matrix: [`paper/v8_artifacts/actionability_matrix.md`](paper/v8_artifacts/actionability_matrix.md).

## What ResistanceMap is genuinely good for

1. **Wet-lab compound prioritization on MM-relevant cell lines** — for 9/11 GDSC drugs, ResistanceMap ranks candidates with Spearman 0.30–0.40, sufficient to focus follow-up IC50 testing. Most useful cases: Bortezomib (frontline MM standard-of-care, MSE 0.032), Venetoclax (t(11;14) MM, MSE 0.010), Lenalidomide (MM IMiD backbone, Spearman 0.396).
2. **A re-usable pre-trained 64-d cell-line latent embedding** (`checkpoints/vae_finetuned.pt`) over 886 hematologic-relevant cell lines — usable as a feature extractor for downstream studies.
3. **A reproducibility-first multi-omics integration benchmark** with cryptographic boundary hashing, automatic baseline comparison (`scripts/run_baselines_real.py`), and Tier-A→D evaluation governance (`paper/v8_evaluation_report.md`).
4. **A documented HDAC-class failure mode** — itself a useful negative finding for the field. The CCLE proteomics + epigenomics features available do not contain sufficient signal to rank Panobinostat/Romidepsin response. Future work should ingest histone PTM proteomics or HDAC-target enrichment scores.

## What ResistanceMap should NOT be used for

- **Patient-level treatment-response prediction** — no patient training data has been ingested. The pipeline references `gse124310.h5ad`, `gse271107.h5ad`, and `mmrf_commpass/` in config, but `checkpoints/data_ready.pt` contains only cell-line proteomics + epigenomics + PPI. Patient-cohort claims are out of scope until that data is wired in.
- **Time-to-resistance forecasting / "predict before it happens"** — earlier docs claimed this; it is not supported by training data. Training pairs are contemporaneous `(X_cell-line, IC50_observed)`; there are no `(X_t, X_{t+Δ})` longitudinal pairs.
- **Pathway-level mechanism attribution** — the architecture supports per-drug attention weights, but they are **not currently persisted** to checkpoints. Pathway-validator (`.claude/agents/proteomics-pathway-validator.md`) cannot evaluate grounding until this is fixed.
- **Substitute for IMWG MM response criteria** — even when transferred to patient samples, GDSC IC50 is a cell-line viability metric, not a clinical response. No published IC50 → IMWG mapping exists.
- **HDAC-inhibitor screening** — documented failure mode (1/3 HDAC drugs actionable).

## Architecture

The training DAG that actually runs (10 agents, parallel via `resistancemap/agents/orchestrator.py`):

```
Layer 0  data_validation         file-existence + schema check
Layer 1  data_prep               CCLE proteomics + epigenomics + STRING + GDSC harmonized → data_ready.pt (77 MB)
Layer 2  vae_pretrain ‖ esm2_embed   parallel
                                 (esm2_embed only validates protein names; real ESM-2 forward
                                  requires ds.protein_sequences which is not currently populated)
Layer 3  vae_finetune            hematologic specialization → 64-d latent
Layer 4  trajectory              calibrate stability score + train forecaster on cell-line latents
                                 (cell-line resolution; not single-cell trajectory inference)
Layer 5  protein_net             GAT on STRING PPI subgraph (no real ESM-2 features)
Layer 6  fusion                  cross-attention over (epi, traj, pnet, stab)
Layer 7  landscape               2-D UMAP layout + drug-resistance head
Layer 8  validation              per-drug MSE/MAE/Spearman + actionability flags + baselines
                                 → pipeline_validated.pt + logs/per_drug_metrics.csv + paper/tables/baseline_comparison.md

Boundary integrity:  SHA256 hash chain across all 10 agent outputs (10 hashes per run).
Observability:       AgentOps tracer/evaluator emit real spans (post-v7 fix).
Governance:          run_evaluation_governance() runs Tier A/B/C/D after the training DAG.
```

Each agent boundary computes `verification_hash = SHA256(output)`; the orchestrator validates self-consistency at handoff. The verification chain is reported in stdout and persisted in the AgentOps dashboard.

## Data — what's actually loaded

| Dataset | What's in `checkpoints/data_ready.pt` | What advertised | Status |
|---|---|---|---|
| CCLE Proteomics | 886 cell lines × 19,177 proteins | 1,393 lines × 19,177 | **64% of advertised cohort** (subset selection during harmonization) |
| CCLE Epigenomics | 886 lines × 42 features | 897 × ~42 | substantially complete |
| STRING PPI v12 | 19,177 nodes, **473,860 directed edges** | 930k edges | **~half** (confidence-cutoff 0.7 in `configs/default.yaml`) |
| GDSC drug sensitivity | 886 × 11 drugs, **53.1% NaN sparse** (3,635 observed train cells) | 11 MM-relevant drugs | feature-complete; sparse-by-design |
| scRNA GSE124310 (27,796 MM cells) | **NOT INGESTED** | listed in config | future work |
| scRNA GSE271107 (143,748 cells, HD→MGUS→SMM→MM) | **NOT INGESTED** | listed in config | future work |
| MMRF CoMMpass clinical (1,143 patients) | **NOT INGESTED** | listed in config | future work (`data/mmrf_loader.py` exists but is unwired) |
| Protein FASTA sequences for ESM-2 | **NOT POPULATED** | "1,280-d ESM-2 embeddings" | `esm2_embed` agent only validates names; real ESM-2 forward never runs |

**Splits** (from `data_ready.pt[splits]`): train 622 / val 132 / test 132 (70 / 15 / 15, random seed 42). Per-drug observed-cell counts in train range 267–348 (median ~340 / 622 = 55%).

### Data Sources Table — provenance

| Dataset | Type | Source | Public/Restricted | Currently used? |
|---|---|---|---|---|
| CCLE Proteomics | Bulk | DepMap | Public | **Yes** |
| CCLE Epigenomics | Bulk | DepMap/ENCODE | Public | **Yes** |
| STRING PPI v12 | Network | STRING-DB | Public | **Yes** |
| GDSC | Drug Screen | CancerRxGene | Public | **Yes** |
| CTRPv2 | Drug Screen | Broad | Registered | configured but not loaded |
| GSE124310 | scRNA-seq | GEO | Public | **No (planned)** |
| GSE271107 | scRNA-seq | GEO | Public | **No (planned)** |
| MMRF CoMMpass | Clinical | GDC Portal | IRB | **No (planned, IRB-gated)** |

## Quick Start

```bash
git clone <this-repo>
cd ResistanceMap
pip install -e .

# End-to-end run (requires real data in data/raw/ — see scripts/download_data.sh)
python main.py --config configs/default.yaml

# Just the validation stage (uses cached checkpoints from training)
python main.py --config configs/default.yaml --stage validate

# Generate the per-drug decision matrix from validated checkpoints
python scripts/generate_actionability_report.py

# Compare against integration baselines (concat / PCA / MOFA+-like / late-fusion)
python scripts/data_integration_audit.py

# Generate Krishnaswamy-style figures (UMAP fallback if PHATE not installed)
python scripts/krishnaswamy_visualizations.py

# Run orthogonal Tier-A→D evaluation governance
python main.py --config configs/default.yaml --evaluate
```

Configs live in `configs/`. `default.yaml` is CPU-friendly; `h100.yaml` uses bf16 + torch.compile for H100. Each stage is checkpoint-aware — a re-run skips completed stages automatically.

## Performance — real numbers, no fabrication

These are the actual numbers from `checkpoints/pipeline_validated.pt`. **Earlier README versions reported AUROC 0.82–0.89 / AUPRC 0.76–0.84 / MAE 0.09–0.18 — those numbers had no provenance in this codebase and have been removed.**

### Aggregate test_mse vs baselines (NaN-masked, same train/val/test split, n_test=132)

From `paper/tables/baseline_comparison.md` (auto-generated by `scripts/run_baselines_real.py`):

| Rank | Model | test_mse | val_mse | Notes |
|---|---|---|---|---|
| 1 | Zero (z-scored target floor) | 2.3678 | 1.8841 | predicts 0 everywhere |
| 2 | PerDrugTrainMean | 2.3678 | 1.8841 | predicts per-drug train mean |
| 3 | GradientBoosting (PCA-256) | 2.3699 | 2.1597 | strong classical baseline |
| 4 | **ResistanceMap (10-agent DAG)** | **2.3731** | n/a | this pipeline |
| 5 | RandomForest (PCA-256) | 2.4134 | 2.0232 | |
| 6 | Ridge (full proteomics + epi) | 2.4998 | 2.0536 | |
| ... | ... | ... | ... | (see baseline_comparison.md for the full panel) |

The Zero / per-drug-mean baselines beat ResistanceMap **on the aggregate metric** because of the Panobinostat outlier. On the 9 actionable drugs, ResistanceMap meaningfully outperforms a constant predictor in rank correlation (median Spearman 0.328 vs 0).

### Per-drug performance

See [Decision matrix](#decision-matrix--when-to-use-resistancemap) above. Source CSV: `logs/per_drug_metrics.csv`.

### Integration ablation (from `paper/v8_artifacts/data_integration_audit.md`)

Same train/val/test split, masked-MSE pooled across observed cells:

| Rank | Method | test_mse |
|---|---|---|
| 1 | Late fusion (avg of per-modality Ridge) | 2.3653 |
| 2 | **ResistanceMap CrossModalFusionNet** | **2.3731** |
| 3 | MOFA+-like shared factors (64) → Ridge | 2.4440 |
| 4 | Naive concat → Ridge | 2.4583 |
| 5 | PCA(256/20) per-modality → Ridge | 3.5652 |

ResistanceMap beats naive concat, PCA-Ridge, and a MOFA+-like factor model. It is beaten by simple late fusion by 0.008 MSE — the cross-attention machinery is **not yet justified** by the available data scale (622 train, ~340 obs/drug).

## Known Limitations (explicit)

1. **Data scale ceiling.** Per-drug training counts (~270–350 cells) are small for a 19,177-feature input. Doubling the cohort (e.g., adding CCLE 2024 release) is the highest-ROI next step.
2. **HDAC class is a documented failure mode.** Panobinostat (MSE 22.3, Spearman 0.045) and Romidepsin (Spearman 0.240) are below screening threshold. Vorinostat is borderline-actionable. The current omics features lack sufficient HDAC-mechanism signal.
3. **No patient cohorts.** The architecture is set up to ingest scRNA + MMRF, but no code path currently does. README's earlier patient-cohort framing was aspirational, not actual.
4. **No real ESM-2 embeddings.** `ESM2EmbedAgent` validates protein metadata; the real 1,280-d forward pass only fires inside `train_protein_network` if `dataset.protein_sequences` is non-empty (it isn't). The protein-net trains on abundance + VAE latent + stability features.
5. **No persisted attribution.** Per-drug attention weights are computed inside `CrossModalFusionNet` and `ProteinNetwork` but not saved to checkpoints. This blocks the v8 `proteomics-pathway-validator` from grounding predictions in KEGG/Reactome.
6. **Cross-attention complexity not justified.** Late fusion (avg of per-modality Ridge) beats ResistanceMap by 0.008 MSE on the same data. Without more training data or a per-drug head, the architecture's complexity is not earning its keep.
7. **No longitudinal data.** "Predict resistance state at time t" is not a supported claim; training is on snapshot pairs only.
8. **No causal grounding.** Without perturbation training data (e.g., DepMap CRISPR screens), pathway "transition" predictions are associative, not interventional.

## Reproducibility

Every numeric claim in this README is tagged to a file:

| Claim | Source |
|---|---|
| Aggregate test_mse, n_test_samples, n_drugs | `checkpoints/pipeline_validated.pt[metrics]` |
| Per-drug MSE, MAE, Spearman | `logs/per_drug_metrics.csv`, also `pipeline_validated.pt[metrics][per_drug_metrics]` |
| Actionability flags + drug-class summary | `pipeline_validated.pt[metrics][actionability_summary]` |
| Baseline ranking | `paper/tables/baseline_comparison.md` |
| Integration ablation | `paper/v8_artifacts/data_integration_audit.md` |
| SOTA literature comparison | `paper/v8_artifacts/sota_comparison.md` (PMIDs verified live via PubMed) |
| Visualizations | `paper/v8_artifacts/visualizations/*.png` |
| Strict evaluation report | `paper/v8_evaluation_report.md` |

Reproduce everything from a clean state:

```bash
rm -rf checkpoints/ logs/
python main.py --config configs/default.yaml             # full DAG run
python scripts/generate_actionability_report.py
python scripts/data_integration_audit.py
python scripts/krishnaswamy_visualizations.py
python main.py --config configs/default.yaml --evaluate   # Tier-A→D governance
```

Run-ledger policy: every numeric claim in `README.md` and `ARCHITECTURE.md` must trace back to a row in `RUNS.md` (enforced by `scripts/check_docs_consistent.py`).

## Citations and prior art

ResistanceMap stands on four pillars of prior work, cited honestly:

- **MOFA+ — Argelaguet et al. 2020** [Genome Biol, DOI](https://doi.org/10.1186/s13059-020-02015-1), PMID 32393329 — multi-omics factor analysis. ResistanceMap's integration is benchmarked against a MOFA+-like sparse-factor baseline; MOFA+ produces interpretable factors that ResistanceMap currently does not.
- **MM treatment-specific prediction model landscape — Jarrah et al. 2026** [Eur J Haematol, DOI](https://doi.org/10.1111/ejh.70177), PMID 41909977 — the field's current systematic review.
- **Dynamic biomarker MM response — Xiong et al. 2026** [J Transl Med, DOI](https://doi.org/10.1186/s12967-026-07946-0), PMID 41814396 — F1=0.75 at Cycle 4 vs R-ISS F1=0.32 (n=662 patients). The strongest current MM-clinical comparator. ResistanceMap is not a head-to-head comparator (cell line vs patient cohort) but cites this as the patient-side benchmark to migrate toward.
- **Multi-omics MM prognostic signature — Li et al. 2026** [Ann Hematol, DOI](https://doi.org/10.1007/s00277-026-06867-8), PMID 41762247 — uses overlapping data (GSE124310) that ResistanceMap should ingest in the next iteration.

Trajectory-method context (the field that "predicts cell-state evolution"):
- **TrajectoryNet** — Tong, Huang, Wolf, van Dijk, Krishnaswamy 2020 (ICML / arXiv:2002.04461) — single-cell dynamic optimal transport. Operates at single-cell resolution; ResistanceMap is cell-line resolution.
- **VGFM** — Wang et al. 2025 ([hf.co/papers/2505.13413](https://hf.co/papers/2505.13413)) — flow matching with mass growth, the modern successor to TrajectoryNet.
- **Conditional Monge Gap** — Driessen et al. 2025 ([hf.co/papers/2504.08328](https://hf.co/papers/2504.08328)) — neural OT generalizing to unseen drugs.

The detailed comparator landscape, with verified PubMed IDs and DOIs, is in [`paper/v8_artifacts/literature_review.md`](paper/v8_artifacts/literature_review.md) and [`paper/v8_artifacts/sota_comparison.md`](paper/v8_artifacts/sota_comparison.md).

## Roadmap (honest priority order)

1. **Persist per-drug attention weights** in `train_fusion` and `train_protein_network` → unlocks pathway-level interpretability and KEGG/Reactome grounding (~50 LOC).
2. **Wire `data/mmrf_loader.py`** into `prepare_data` so MMRF clinical labels feed validation. Closes the "no patient data" gap (~150 LOC + IRB).
3. **Ingest scRNA from GSE124310 + GSE271107** as additional fusion modality. The two h5ad files are already in `data/raw/`. (~300 LOC)
4. **Add per-drug head specialization** so HDAC-class failure doesn't dominate gradients, and so well-performing drugs (Venetoclax, Bortezomib) aren't pulled toward the mean. (~100 LOC)
5. **Add real ESM-2 forward pass** by populating `dataset.protein_sequences` from UniProt FASTA during `prepare_data`. (~200 LOC)
6. **Statistical significance test** vs late-fusion baseline (paired Wilcoxon over 5 random seeds) — required to claim the cross-attention machinery is justified.
7. **Replace synthetic external-cohort path** in `evaluation/external_validation.py` with real MMRF/GMMG/IFM connectors as those datasets become accessible.

## License

MIT — see `LICENSE`.

## Contributing

Issues + PRs welcome. Three project-policy rules enforced via `.claude/settings.json` hooks:

1. No `np.random.default_rng` / `np.random.seed` / `_generate_synthetic_cohort` in non-test code (blocks fabrication).
2. `scripts/generate_paper_figures.py:generate_all` is blocked at the Bash hook level (most panels use random data); use `scripts/run_real_figures.py` instead.
3. Any change to README/ARCHITECTURE must trace numeric claims back to `RUNS.md` rows (enforced by `scripts/check_docs_consistent.py`).

The v8 strict-evaluation multi-agent architecture (9 PhD-level subagents under `.claude/agents/`, with skills under `.claude/skills/`) is available for any future evaluation pass.

---

_README v8 corrected 2026-05-02. All metrics derive from `checkpoints/pipeline_validated.pt` produced by the 2026-05-02 run. Earlier README versions reported AUROC/AUPRC/MAE numbers without provenance; those have been removed and replaced with this honest, on-disk-anchored version._
