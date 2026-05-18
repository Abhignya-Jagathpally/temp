# ResistanceMap / MORT-FM

Branch `v17` (latest head; v16 archived). v17 implements the full
steering plan from the v16 honest-limitations review (9 commits):

  1. **Collapse duplicate MORT-FM paths into one executable path** (v17.3)
  2. **Wire real biological graph identifiers** — STRING alias → HGNC ↔ UniProt (v17.1)
  3. **Make clinical-only Cox the mandatory baseline** for every patient-level claim (v17.2)
  4. **Longitudinal cohort registry** with explicit "contributes_to" semantics (v17.4)
  5. **Block C contrastive/stage-aware re-train** — macro-F1 0.147 → **0.431** held-out (v17.5)
  6. **Drug-conditioned BeatAML head** — scratch 0.203 / init_a 0.276 / Δ +0.074 with CI [+0.033, +0.112] (v17.6)
  7. **Regularized LENS + Harrell optimism correction** — corrected C=0.480 (v17.7)
  8. **Three-channel causal evidence** (CRISPR + drug-target + Reactome) with permutation enrichment p=0.002 (v17.8)
  9. **Final gate revalidation: 6/12 granted (+1 vs v16)** — newly granted: `hematologic_specimen_drug_response` (v17.9)

This repository contains two scientific tracks that share an infrastructure
backbone:

1. **ResistanceMap v8** — a trained, calibrated cell-line drug-screening
   pipeline (CCLE proteomics + epigenomics + STRING PPI + GDSC). Still on
   disk; numbers anchored to `checkpoints/pipeline_validated.pt`. See
   [§ ResistanceMap v8 historical track](#resistancemap-v8-historical-track)
   below for the original headline numbers and decision matrix.
2. **MORT-FM v15+v16** — a Multi-Omic Resistance Trajectory Foundation
   Model: graph-conditioned Neural SDE on a learned Waddington landscape,
   typed registries / claim gates / endpoint registry, longitudinal MMRF
   substrate, causal counterfactual stack. **Current pass focus.** All
   results below are real, on-disk, and traceable to `RUNS.md` rows.

> No fabrication, no synthetic cohorts. The `.claude/settings.json` hook
> `scripts/hooks/block_synthetic_data.sh` blocks any `np.random` /
> `_generate_synthetic_cohort` introduction. Every numeric claim in this
> README cites either a checkpoint or a `RUNS.md` row.

---

## MORT-FM v15+v16+v17 status (current)

### v17 deltas (since v16, nine commits — final tally **6/12 granted, +1 vs v16**)

| v17 commit | Real measurable change |
|---|---|
| **v17.1** STRING-alias → HGNC ID bridge | Causal-validator HGNC coverage **0% → 100%** (200/200 edges); common-essential overlap 0% → 5.5% |
| **v17.2** Clinical-only Cox + permutation null | clinical-only Cox C=0.366 < null (p=0.956); LENS beats Cox by +0.237 but lower CI < null+2σ — gate correctly refused |
| **v17.3** Unify model.py around v16 LENS | `MORTFM.forward_lens()` calls v16 modules directly; legacy + v16 both verified |
| **v17.4** Longitudinal cohort registry | 7 cohorts typed; censoring policy + temporal splitter centralised |
| **v17.5** Block C contrastive | **macro-F1 = 0.431** held-out (vs v16 self-sup 0.147) — passes "useful" band |
| **v17.6** Drug-conditioned BeatAML head | **scratch 0.203 / init_a 0.276 / paired Δ +0.074 [+0.033, +0.112]** — **`hematologic_specimen_drug_response` NEWLY GRANTED** |
| **v17.7** Regularized LENS + Harrell optimism | clinical+z0 raw 0.515, optimism +0.035, **corrected 0.480** — still below 0.5; binding constraint = real n |
| **v17.8** Causal evidence v2 (3 channels + enrichment) | Top-20 pathway count 64 vs null 14.4 → **10.9σ, p=0.002**; drug-target support 40%; only CRISPR overlap (1/20) blocks |
| **v17.9** Final gate revalidate | **6/12 claim levels granted** (+1 vs v16) |

### Claim gates (12 levels, **6 granted, 6 blocked with concrete reasons**)

```
GRANTED (6/12):
  technical                          — pipeline runs end-to-end on real data
  static_drug_response               — Block A (DepMap × GDSC, median Spearman 0.191)
  hematologic_specimen_drug_response — v17.6 drug-conditioned head: scratch 0.203,
                                       init_a 0.276, paired Δ +0.074 [+0.033, +0.112]
                                       (NEWLY GRANTED in v17)
  single_cell_state                  — v17.5 contrastive Block C: held-out
                                       macro-F1 = 0.431 (v16 self-sup was 0.147)
  sequence_aware                     — Block D (ESM-2 cache 7,061 sequences)
  pathway_context                    — v17.8 top-20 pathway count 64 vs null 14.4,
                                       p=0.002 (10.9σ enrichment; vs v16 it was
                                       structural-only)

BLOCKED (6/12, concrete refusal reason per level):
  drug_target_mechanism      — per-source coverage 32.5/37.6/46.7% < 50% spec
  survival_prediction        — v17.7 corrected C=0.480 < 0.50 (n=29 binding;
                               raw LOO C=0.515 [0.348, 0.684])
  patient_level_clinical_prediction — needs external/temporal holdout + clinical
                                      baseline win; v17.7 verdict: False
  longitudinal_trajectory    — 29 paired patients < 100 threshold
  resistance_emergence       — lower CI 0.348 < 0.55 (needs ≥80-100 more pairs)
  causal_mechanism           — v17.8 strict gate: 1/20 common-essential targets
                               (need ≥4); BUT pathway enrichment p=0.002 +
                               drug-target support 40% — two of three channels
                               confirm
```

Refresh registries + revalidate:

```bash
python scripts/mortfm_emit_registries.py
python scripts/mortfm_validate_integrated_checkpoint.py
# verdict: PASS — 5 claim levels granted
```

### Honest result matrix (all from real, on-disk runs)

| Run | Real number | RUNS row |
|---|---|---|
| Block A cell-line foundation | Median Spearman **0.191** over 286 GDSC drugs; 124/286 sig at p<0.05; 10 at p<0.001 | `r-2026-05-17-mortfm-block-a-cellline-foundation` |
| Block B BeatAML from-scratch (30ep) | Median Spearman -0.006 [-0.063, +0.047] | `r-2026-05-18-mortfm-beataml-compare-30ep` |
| Block B BeatAML init-from-A (30ep) | Median Spearman **+0.103** [+0.070, +0.124] — clears 0.10 threshold | same |
| **Block B transfer effect** | **Paired Δ = +0.110 [+0.081, +0.133]** — CI excludes zero | same |
| Block C scRNA latent atlas | 51 pseudobulks + 4,000 cell-level latents; stage classifier macro-F1 0.147 (< 0.25 baseline — honest) | `r-2026-05-18-mortfm-scrna-latent-atlas` |
| Block D ESM-2 + drug-family | 7,061 ESM-2 embeddings; 1,975/2,000 Block-A coverage for PADIMAC | `r-2026-05-18-mortfm-block-d-validation`, `r-2026-05-18-mortfm-padimac-ingest` |
| Longitudinal MMRF substrate | 29 paired patients, 20 observed TT2L events | `r-2026-05-18-mortfm-longitudinal-substrate` |
| LENS resistance: baseline | LOO C-index 0.531 [0.395, 0.673] | `r-2026-05-18-mortfm-lens-resistance-variants` |
| LENS resistance: **+ clinical** | **LOO C-index 0.603 [0.442, 0.797]** — best variant (+0.072) | same |
| LENS resistance: + graph projector | LOO C-index 0.552 [0.394, 0.702] (+0.021) | same |
| LENS resistance: clinical+graph | LOO C-index 0.506 [0.346, 0.677] — **overfits at n=29** | same |
| DepMap CRISPR external evidence | 1,095 cell-lines × 17,931 genes → **1,057 common essentials** | `r-2026-05-18-mortfm-crispr-ingest` |
| Causal counterfactual smoke | 200 edges, top-10 vs null **separation 4.32σ** (> 2σ) | `r-2026-05-18-mortfm-causal-smoke` |

The **single biggest finding of v16**: at n=29 paired MMRF patients,
**clinical features (ISS + age + bort_1L + n_treatments) provide the only
robust C-index lift** (+0.072); the graph projector adds marginally (+0.021);
combining them overfits (-0.025). **The binding constraint is data scale,
not model capacity.**

### MORT-FM architecture

```
                              EXISTING (v15)               NEW (v16)
                              ──────────────              ─────────
Layer A. cell-line foundation │ Block A: DepMap × GDSC × STRING + Reactome + ChEMBL
                              │   checkpoints/mortfm/block_a_cellline_foundation.pt
                              │
Layer B. specimen finetune   │ Block B: BeatAML aligned to A's 2,000-gene space
                              │   137/139 params ported from A; +30ep training
                              │
Layer C. scRNA state encoder │ Block C: HVG-union 3,535 genes; pseudotime ordinal
                              │   checkpoints/mortfm/block_c_state_encoder.pt
                              │
Layer D. sequence + graph    │ Block D: ESM-2 8M cache + STRING graph (636K edges)
                              │   data/processed/proteins/esm_embeddings__*.pt
                              │                          + DepMap CRISPR (NEW)
                              │
Layer E. integrated checkpoint│ Block E: A overlaid with B (137 keys); C+D pointers
                              │   checkpoints/mortfm/mortfm_integrated.pt + manifest
                              │
Layer F. LENS resistance      │                          ▼ NEW v16 ▼
                              │   GraphEnergyResistanceSDE: dz_t =
                              │     [-∇U_θ(z,d) + f_θ(z,g,d,c)] dt + σ_θ(z,d) dW_t
                              │   + CompetingRiskHead → discrete-time hazards
                              │   + LatentToGraphProjector → patient-specific graph
                              │   + clinical_features → ISS/age/bort_1L/n_treatments
                              │   trained LOO on 29 MMRF paired patients with TT2L
                              │
Layer G. causal               │                          ▼ NEW v16 ▼
                              │   InterventionGraph(do(e=...)) over 636K real edges
                              │   CounterfactualRunner → Δ_e per edge
                              │   PathwayCausalValidator → CRISPR + drug-target
                              │     external-evidence checks; refuses claims with
                              │     concrete reasons.
                              │
META.   registries            │                          ▼ NEW v16 ▼
                              │   resistancemap/mortfm/registries/
                              │     feature_space_registry  (8 canonical spaces)
                              │     endpoint_registry       (10 typed endpoints
                              │       with allow/forbid claim lists — OS is
                              │       explicitly forbidden from resistance_emergence)
                              │     claim_gate_registry    (12 claim levels)
                              │     artifact_registry      (5 block records)
                              │   scripts/mortfm_emit_registries.py
                              │   scripts/mortfm_validate_integrated_checkpoint.py
```

### Module layout

```
resistancemap/mortfm/
  registries/           — feature_space, endpoint, claim_gate, artifact
  blocks/               — block_a..e + _common (thin adapters over checkpoints)
  longitudinal/         — schemas, MMRF temporal pair builder, clinical_features
  trajectory/           — GraphEnergyResistanceSDE, ResistanceBasin, HittingTime,
                          LatentToGraphProjector, WaddingtonPotential
  survival/             — CompetingRiskHead, nll_competing_risk, concordance_index
                          + bootstrap CI, integrated_brier_score, KM strata,
                          SurvivalCalibration (LBFGS temp-scaling),
                          ClinicalOnlyCox, permutation_null_cindex  (v17)
  graph/                — IDHarmonizer (STRING alias ↔ HGNC ↔ UniProt bridge)  (v17)
  causal/               — InterventionGraph (now ID-bridged), CounterfactualRunner,
                          PathwayCausalValidator (now CRISPR-bridged)
resistancemap/training/
  loss_router.py        — LossRouter with required-supervision guardrail
scripts/
  mortfm_emit_registries.py
  mortfm_validate_integrated_checkpoint.py
  mortfm_train_beataml_finetune.py          — Block B (with --aligned-expression,
                                              --block-a-checkpoint, --pretrain-epochs,
                                              --finetune-epochs, --per-drug-out)
  mortfm_beataml_compare.py                  — paired bootstrap CIs
  mortfm_train_scrna_state_encoder.py        — Block C
  mortfm_scrna_latent_atlas.py               — extract z0 latents + UMAP + stage CV
  mortfm_compute_esm2_embeddings.py          — Block D ESM-2 cache
  mortfm_block_d_validation.py
  mortfm_ingest_padimac.py                   — GSE116324 baseline RNA-seq
  mortfm_ingest_crispr.py                    — DepMap 23Q2 gene-effect
  mortfm_integrate_blocks_bcd.py             — Block E
  mortfm_build_longitudinal_dataset.py       — 29 MMRF temporal pairs
  mortfm_train_lens_resistance.py            — LENS LOO trainer with
                                              --use-clinical / --use-graph-projector
  mortfm_lens_variant_compare.py
  mortfm_lens_smoke_test.py
  mortfm_causal_smoke_test.py
  # v17 additions:
  mortfm_causal_evidence_report.py           — per-edge HGNC/UniProt/CRISPR table
  mortfm_survival_baseline_suite.py          — clinical-only Cox + permutation null
  mortfm_v17_unified_smoke.py                — contract test: both forward() paths
```

### Quick start (MORT-FM v16)

```bash
# 0. Emit registries from current artifacts
python scripts/mortfm_emit_registries.py

# 1. Validate the integrated checkpoint against the artifact registry
python scripts/mortfm_validate_integrated_checkpoint.py

# 2. Reproduce the LENS 4-variant ablation (29 MMRF LOO × 80 epochs × 4 = ~4 min)
for V in baseline clinical_only graph_only clinical_plus_graph; do
  case "$V" in
    baseline)            F="" ;;
    clinical_only)       F="--use-clinical" ;;
    graph_only)          F="--use-graph-projector" ;;
    clinical_plus_graph) F="--use-clinical --use-graph-projector" ;;
  esac
  python scripts/mortfm_train_lens_resistance.py --epochs 80 \
    --out "logs/mortfm/lens_resistance_${V}.json" $F
done
python scripts/mortfm_lens_variant_compare.py

# 3. Reproduce the Block B 30-epoch transfer comparison
python scripts/mortfm_train_beataml_finetune.py \
  --aligned-expression data/processed/beataml/beataml_expression_block_a_features.parquet \
  --pretrain-epochs 30 --finetune-epochs 30 --restrict-to-mapped \
  --summary-out logs/mortfm/beataml_block_b_scratch_30ep_summary.json \
  --checkpoint-name beataml_block_b_scratch_30ep.pt \
  --per-drug-out logs/mortfm/beataml_per_drug_scratch_30ep.csv

python scripts/mortfm_train_beataml_finetune.py \
  --block-a-checkpoint checkpoints/mortfm/block_a_cellline_foundation.pt \
  --aligned-expression data/processed/beataml/beataml_expression_block_a_features.parquet \
  --pretrain-epochs 30 --finetune-epochs 30 --restrict-to-mapped \
  --summary-out logs/mortfm/beataml_block_b_init_30ep_summary.json \
  --checkpoint-name beataml_block_b_finetuned.pt \
  --per-drug-out logs/mortfm/beataml_per_drug_init_30ep.csv

python scripts/mortfm_beataml_compare.py \
  --scratch-csv logs/mortfm/beataml_per_drug_scratch_30ep.csv \
  --init-a-csv logs/mortfm/beataml_per_drug_init_30ep.csv

# 4. Causal counterfactual smoke (200 edges × 8 real MMRF patients)
python scripts/mortfm_causal_smoke_test.py --n-edges 200
```

### MORT-FM honest limitations

1. **n=29 is the binding constraint on resistance_emergence.** Clinical-only
   LENS reaches C-index 0.603 [0.442, 0.797]; lower CI 0.442 misses the 0.50
   survival gate. With ~100 paired patients this would likely pass. See
   `docs/LONGITUDINAL_DATA_INVENTORY.md` for the public-cohort path.
2. **Block B transfer gain is real but the gate's "both variants" rule still fails.**
   At 30 epochs, paired Δ = +0.110 [+0.081, +0.133] (CI excludes zero), init-A
   median = 0.103 (above 0.10 threshold). Scratch median = -0.006 (below threshold).
   The gate requires both ≥ 0.10. Transfer effect is robust; gate-as-written is
   conservative.
3. **STRING source_id ↔ HGNC symbol mapping is unfinished.** The causal
   validator's CRISPR cross-check reads 0% essential overlap for top edges
   because edge IDs use STRING ENSP-style identifiers, not HGNC symbols.
   Bridging via the harmonization map is a label-mapping fix, not a model fix.
4. **Block C latent stage-classifier macro-F1 = 0.147** (below 0.25 random baseline).
   3-epoch self-supervised reconstruction does not produce stage-discriminative
   latents. Either needs many more epochs, a contrastive/stage-conditional
   objective, or richer encoding (e.g., not zero-fill HVG-union absent genes).
5. **The LENS graph_emb is a learned projection from z0, not Block A's real
   biological graph.** Wiring the real graph through to MMRF latents (which
   requires raw MMRF RNA in the Block-A feature space) is a real next step.
6. **All Block A/B numbers are at debug-tier capacity (small d_latent=64,
   d_token=32, no graph-conditioned drift on the patient side).** Scale-ups
   are honest follow-ups, not currently performed.

### Path to clear remaining gates

| Gate | What's needed | Estimated effort |
|---|---|---|
| `hematologic_specimen_drug_response` | (a) drug-conditioned head replacing mean-aggregated; (b) per-drug supervision; (c) ≥ 60 epochs | code change + day-scale training |
| `drug_target_mechanism` | wider ChEMBL ingest (current 5,092 drugs; full ChEMBL v34 has 2M+) | ~1 day data work |
| `survival_prediction` / `resistance_emergence` | ~80-100 more paired patients (MMRF IA-19, GSE39754, Tirier scRNA) | dbGaP IRB for MMRF; rest is public |
| `longitudinal_trajectory` | same as above + calendar-time labels | same |
| `causal_mechanism` | STRING ENSP → HGNC bridge in InterventionGraph; ≥ 20% CRISPR essential overlap in top-10 edges | ~half-day code |
| `patient_level_clinical_prediction` | external/temporal holdout + clinical-only Cox baseline | day-scale once data lands |

---

## ResistanceMap v8 historical track

The v8 ResistanceMap cell-line pipeline is still reproducible from
`checkpoints/pipeline_validated.pt` (last full run 2026-05-02). It targets
a different scientific contribution: **wet-lab compound prioritisation on
MM-relevant cell lines for 9 of 11 GDSC drugs**.

### v8 headline result

```
Trained on:        622 CCLE cell lines × 11 GDSC drugs (3,635 observed IC50 cells)
Test:              132 cell lines, 736 observed (drug, cell-line) pairs
Aggregate test_mse:  2.3731 (NaN-masked, pooled across all drugs/cells)
Per-drug Spearman:   0.045 — 0.396 (median 0.328)
Actionable for screening:  9/11 drugs (82%)  ← Spearman ≥ 0.25 AND n_test ≥ 30
Well-calibrated for IC50:  8/11 drugs (73%)  ← additionally MSE < 1.0
Documented failure modes:  Panobinostat, Romidepsin (HDAC class)
```

### v8 per-drug decision matrix

| Drug | Class | n_test | MSE | Spearman | Verdict |
|---|---|---|---|---|---|
| Venetoclax | BCL2 | 68 | 0.010 | 0.328 | screen + IC50 |
| Bortezomib | Proteasome | 69 | 0.032 | 0.338 | screen + IC50 |
| Dinaciclib | CDK | 66 | 0.156 | 0.373 | screen + IC50 |
| Palbociclib | CDK | 69 | 0.197 | 0.305 | screen + IC50 |
| Cyclophosphamide | DNA-damage | 68 | 0.279 | 0.358 | screen + IC50 |
| Vorinostat | HDAC | 69 | 0.413 | 0.310 | screen + IC50 |
| Lenalidomide | IMiD | 69 | 0.499 | 0.396 | screen + IC50 |
| Etoposide | DNA-damage | 67 | 0.531 | 0.328 | screen + IC50 |
| Doxorubicin | DNA-damage | 69 | 1.666 | 0.339 | screen only |
| Romidepsin | HDAC | 56 | 0.237 | 0.240 | **DO NOT USE** |
| Panobinostat | HDAC | 66 | 22.335 | 0.045 | **DO NOT USE** |

Full v8 matrix + drug-class summary: `paper/v8_artifacts/actionability_matrix.md`.

### v8 baseline comparison

ResistanceMap v8 is ranked 4 / 11 on aggregate `test_mse` against simple
baselines (Zero, PerDrugTrainMean, GradientBoosting, ResistanceMap,
RandomForest, Ridge), beaten by Zero / PerDrugTrainMean **on aggregate
because of the Panobinostat MSE=22.3 outlier**. On the 9 actionable drugs,
ResistanceMap beats a constant predictor in rank correlation (median Spearman
0.328 vs 0). See `paper/tables/baseline_comparison.md`.

### v8 architecture (10-agent training DAG)

```
Layer 0  data_validation         file-existence + schema check
Layer 1  data_prep               CCLE proteomics + epigenomics + STRING + GDSC harmonized
Layer 2  vae_pretrain ‖ esm2_embed   (esm2_embed currently validates names only;
                                      see "what's not loaded" note below)
Layer 3  vae_finetune            hematologic specialization → 64-d latent
Layer 4  trajectory              calibrate stability score + train forecaster
Layer 5  protein_net             GAT on STRING PPI subgraph
Layer 6  fusion                  cross-attention over (epi, traj, pnet, stab)
Layer 7  landscape               2-D UMAP layout + drug-resistance head
Layer 8  validation              per-drug MSE/MAE/Spearman + actionability flags
```

v8 quick-start:

```bash
python main.py --config configs/default.yaml             # full DAG run
python scripts/generate_actionability_report.py
python scripts/data_integration_audit.py
python main.py --config configs/default.yaml --evaluate  # Tier-A→D governance
```

### v8 what's NOT loaded (carry-over honest disclaimer)

| Dataset | Status |
|---|---|
| scRNA GSE124310 (27,796 MM cells) | Not in v8 (loaded in v15+ Block C) |
| scRNA GSE271107 (143,748 cells, HD→MGUS→SMM→MM) | Not in v8 (loaded in v15+ Block C) |
| MMRF CoMMpass (1,143 patients) | Not in v8 (loaded in v15+ longitudinal) |
| Protein FASTA / real ESM-2 forward | Not in v8 (loaded in v15+ Block D) |

---

## Reproducibility / run-ledger policy

Every numeric claim in this README cites either a checkpoint or a row in
`RUNS.md`. The CI check `scripts/check_docs_consistent.py` greps the README
and `ARCHITECTURE.md` for numeric claims and refuses the build if they
cannot be traced.

```bash
rm -rf checkpoints/ logs/
# v8 path:
python main.py --config configs/default.yaml
# v16 MORT-FM path:
python scripts/mortfm_ingest_depmap.py
python scripts/mortfm_ingest_chembl.py
python scripts/mortfm_ingest_reactome.py
python scripts/mortfm_ingest_uniprot.py
python scripts/mortfm_ingest_string.py
python scripts/mortfm_ingest_beataml.py --mode auto
python scripts/mortfm_train_cellline_foundation.py    # Block A
python scripts/mortfm_align_beataml_features.py
python scripts/mortfm_train_beataml_finetune.py \
  --block-a-checkpoint checkpoints/mortfm/block_a_cellline_foundation.pt \
  --aligned-expression data/processed/beataml/beataml_expression_block_a_features.parquet \
  --pretrain-epochs 30 --finetune-epochs 30 --restrict-to-mapped \
  --checkpoint-name beataml_block_b_finetuned.pt
python scripts/mortfm_train_scrna_state_encoder.py   # Block C
python scripts/mortfm_compute_esm2_embeddings.py     # Block D
python scripts/mortfm_integrate_blocks_bcd.py        # Block E
python scripts/mortfm_build_longitudinal_dataset.py  # Lane 4+5
python scripts/mortfm_ingest_crispr.py               # external causal evidence
python scripts/mortfm_train_lens_resistance.py --use-clinical --epochs 80
python scripts/mortfm_emit_registries.py
python scripts/mortfm_validate_integrated_checkpoint.py
```

---

## Citations / prior art

- **MOFA+** — Argelaguet et al. 2020, [Genome Biol](https://doi.org/10.1186/s13059-020-02015-1), PMID 32393329
- **MM treatment-specific prediction model landscape** — Jarrah et al. 2026, [Eur J Haematol](https://doi.org/10.1111/ejh.70177), PMID 41909977
- **Dynamic biomarker MM response** — Xiong et al. 2026, [J Transl Med](https://doi.org/10.1186/s12967-026-07946-0), PMID 41814396 — F1=0.75 at Cycle 4 vs R-ISS F1=0.32 (n=662 patients); the strongest current MM-clinical comparator
- **Multi-omics MM prognostic signature** — Li et al. 2026, [Ann Hematol](https://doi.org/10.1007/s00277-026-06867-8), PMID 41762247

Trajectory-method context (for the LENS SDE):
- **PRESCIENT** — Yeo et al. 2021 — potential landscape from time-series scRNA, simulates stochastic trajectories in physical time
- **scNODE** — Sha et al. 2024 — VAE + Neural ODEs for unobserved-timepoint scRNA
- **TrajectoryNet** — Tong et al. 2020 (ICML), single-cell dynamic optimal transport
- **CellRank 2** — Lange et al. 2024, multiview fate mapping

PADIMAC reference (GSE116324):
- **Chapman et al. 2018**, [Br J Haematol](https://doi.org/10.1111/bjh.15598), PMID 30181174

DepMap CRISPR external-evidence channel:
- **DepMap 23Q2 Public** — Broad Institute, figshare article 22765112

---

## License

MIT — see `LICENSE`.

## Contributing

Three project-policy rules enforced via `.claude/settings.json` hooks:

1. No `np.random.default_rng` / `np.random.seed` / `_generate_synthetic_cohort` in non-test code (`scripts/hooks/block_synthetic_data.sh`).
2. `scripts/generate_paper_figures.py:generate_all` is blocked at the Bash hook level (most panels use random data); use `scripts/run_real_figures.py` instead.
3. Any change to README / `ARCHITECTURE.md` / `RUNS.md` numeric claims must be traceable to an on-disk artifact.

---

_README v17.9 (final, 2026-05-18) — v17 added 9 commits on top of v16 head
`3cfcaf0`: `4ccf49f` (v17.1 STRING-alias→HGNC bridge), `4f120f8` (v17.2
clinical-only Cox), `72e51dc` (v17.3 model.py unification), `008d071`
(v17.4-8 longitudinal upgrade + Block C contrastive + drug-conditioned
BeatAML + regularised LENS + causal v2), and this commit (v17.9 final
gate revalidation + RUNS + README). All numbers derive from artifacts under
`checkpoints/mortfm/`, `data/processed/`, `results/mortfm/`, and
`logs/mortfm/`. The v8 ResistanceMap section is preserved as a historical
track and is still reproducible from `checkpoints/pipeline_validated.pt`._
