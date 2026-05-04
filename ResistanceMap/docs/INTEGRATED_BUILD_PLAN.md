# Integrated Build Plan — ResistanceMap (post-v8 reframing)

Date: 2026-05-03
Source artifacts (all under `docs/` unless noted):
- `TRAJECTORY_ARCHITECTURE_AND_DATA.md` — architecture decision + data audit
- `SCFOUNDATION_MODEL_PICKS.md` — frozen-encoder pick + verified SHAs
- `SOTA_BENCHMARK_PROTOCOL.md` — publishable bar, comparator numbers
- `CLINICAL_ACTIONABILITY_RUBRIC.md` — 5 resistance states, 5-point Likert, FDA SaMD guardrails
- `CAUSAL_VALIDITY_AUDIT.md` — Pearl-tier ceiling per sub-claim
- `MANUAL_DATA_PULLS.md` — dataset pull list with verified URLs
- `DRUG_STRUCTURAL_FEATURES.md` + `data/raw/drug_metadata.json` — Morgan FP integration plan
- `MODEL_LANDSCAPE_RANKING.md` — drug-response model ranking
- `MMRF_ACQUISITION.md` — MMRF CoMMpass IA22 ingest provenance (added 2026-05-03)
- `PERPATIENT_LATENT_DYNAMICS.md`, `CAUSAL_SURVIVAL_HEAD.md`,
  `FOUNDATION_MODEL_ADAPTATION.md`, `PERPATIENT_VALIDATION_PROTOCOL.md` — v10 per-patient
  track (in-flight 2026-05-03; see §3.4)
- `../resistancemap/observability/SPEC.md` — 14-metric observability/eval/optimization spec
- `../resistancemap/observability/instrumentation.py` — JSONL emitter (3 tests passing)
- `../paper/v8_artifacts/baselines/{elasticnet_results.json, mofa_results.json, mofa_factors.npy}` — baseline floors
- `../data/raw/mmrf_commpass/{clinical,treatments,samples,gene_expression,mutations,copy_number,cytogenetics}.tsv` — MMRF CoMMpass IA22 primary tables (added 2026-05-03; see finding 1.4)

This document is the single source of truth for the next sprint. It is debugging-tier;
no number here may ship to README.md or ARCHITECTURE.md without a corresponding row in
the run ledger (see release-bouncer rule).

---

## 1. The reality check (what changed in our understanding today)

Four structural findings reshape the project scope (1.4 added 2026-05-03 after the
MMRF CoMMpass pull completed mid-day):

**1.1 Cell-line drug-response prediction is at the floor.** Pooled test MSE: predict-mean
2.8106, MOFA+Ridge 2.8155 (19.7 s training), ResistanceMap 2.8364 (~30 min training),
ElasticNet 2.8916. ResistanceMap does not beat a per-drug mean baseline. A 20-factor
MOFA+Ridge model trained in 19.7 s reproduces the deep pipeline's headline number.
*Source: paper/v8_artifacts/baselines/{elasticnet_results.json, mofa_results.json}.*

**1.2 The patient scRNA data is not longitudinal.** GSE124310 (27,796 cells) and
GSE271107 (143,748 cells) are both cross-sectional, single-snapshot per patient. No
spliced/unspliced layers (kills scVelo / dynamo / cellDancer). GSE271107 was deposited
without any treatment / drug / timepoint metadata column — verified live via NCBI eutils
2026-05-03; the word "lenalidomide" does not appear anywhere in the sample annotations.
*Note: this finding is scoped to the GEO scRNA atlases only. It does NOT apply to the
MMRF bulk-omics longitudinal sub-cohort — see finding 1.4.*
*Source: TRAJECTORY_ARCHITECTURE_AND_DATA.md (scanpy audit) + CAUSAL_VALIDITY_AUDIT.md
(GEO eutils verification).*

**1.3 The "predict resistance before it happens" claim is structurally non-falsifiable
on current data.** Pearl-tier per sub-claim (verified by causal-inference-auditor):

| Sub-claim | Required | Supported | Why |
|---|---|---|---|
| (a) which resistance state | L1 | **L1** | Spearman ~0.33 on 9/11 drugs; exchangeability violated by unmeasured cytogenetics (chr1q21, del17p, t(4;14), t(11;14)) |
| (b) when | L2 | **none** | GDSC IC50 is 72-h equilibrium; no time axis exists in any on-disk training source |
| (c) via which pathway | L3 | **none** | attention weights not persisted, attention ≠ causation, no perturbation data |

Sub-claims (b) and (c) are non-falsifiable *on existing data*. Per Popper, a non-falsifiable
claim is not a scientific claim. They must be removed from any submission or downgraded to
PERCEPTION-style "predicts response" / Rade-style "associated with" framing.
*Source: CAUSAL_VALIDITY_AUDIT.md.*

**1.4 MMRF CoMMpass IA22 is now on disk (2026-05-03) and partially unblocks finding 1.3
sub-claim (b).** All seven primary tables are written and row-counted (`wc -l`):

| Table | Rows | Coverage |
|---|---|---|
| `data/raw/mmrf_commpass/clinical.tsv` | 995 patients | ISS stage 995/995, vital status 995/995, survival 995/995 (via /cases API) |
| `data/raw/mmrf_commpass/treatments.tsv` | 7,184 records | 994/995 patients with ≥1 line; 258 patients with ≥2 lines (= observable resistance event); 100 with ≥3 lines |
| `data/raw/mmrf_commpass/samples.tsv` | 4,860 aliquots | spans baseline through recurrence biopsies |
| `data/raw/mmrf_commpass/gene_expression.tsv` | 60,616 genes × 859 aliquots | merged from 859 STAR-Counts files (3.4 GB raw) under `data/raw/mmrf_commpass/rna/` |
| `data/raw/mmrf_commpass/mutations.tsv` | 154,588 rows | merged from 1,091 masked MAFs under `data/raw/mmrf_commpass/snv/`; top genes KRAS 275, NRAS 221 — matches IA-series literature |
| `data/raw/mmrf_commpass/copy_number.tsv` | 1,160,904 segment rows | merged from 1,010 segment files under `data/raw/mmrf_commpass/cnv/` |
| `data/raw/mmrf_commpass/cytogenetics.tsv` | 976 patients | del17p 13.6 %, chr1q21 33.6 %, del13q 52 %, t(4;14) 14.6 %, t(11;14) 16.8 % — all in or near published IA-series literature bands |

**Longitudinal sub-cohort audit** (counts derived from `samples.tsv` + sample-sheet visit
fields, also dated 2026-05-03):

- **51 patients with RNA-Seq at ≥2 distinct visits.** Visit-count distribution:
  736 × 1 visit, 44 × 2, 5 × 3, 2 × 4. The Tier-2-exploratory longitudinal envelope.
- **74 patients with CNV at ≥2 visits.** Slightly larger because CNV was profiled in a
  superset of timepoints.
- **319 "Recurrent Blood Derived Cancer - Bone Marrow" samples** = explicit post-treatment
  relapse biopsies. These pair against baseline biopsies for true paired pre/post
  resistance contrasts (the design Cohen 2021 PMID 33619369 used for melanoma).

What this changes: finding 1.3 sub-claim (b) — "when does resistance occur" — is now
*Tier-2 exploratory* on the N=51-74 sub-cohort, with N=995 supporting the population-marginal
survival head. It is NOT yet Pearl L2 (still observational, no perturbation), but it has
moved from "structurally non-falsifiable on existing data" to "falsifiable on existing data
with limited statistical power." Sub-claim (c) — pathway mediation — is unchanged
(MMRF has no perturbation arm).

*Source: directory listings + `wc -l` on each TSV under `data/raw/mmrf_commpass/`,
2026-05-03; `MMRF_ACQUISITION.md` for ingest provenance.*

---

## 2. The honest scope (what we can defensibly ship)

| Layer | Status today (2026-05-03) | What lands in any paper |
|---|---|---|
| Cell-line drug-response (Tier 1, L1) | At floor; ties MOFA+Ridge. **Tier-2 cross-validation against MMRF (N=994 patients with treatment records) is now runnable** once the layer-1 model is trained — PERCEPTION-style two-cohort design (cell-lines train, MMRF validate). | "Associates static omics with drug-sensitivity rankings; Spearman 0.33 on 9/11 drugs"; with MMRF cross-validation, "external validation in N=994 MM patients per PERCEPTION (PMID 38637658)" — still framed as Tier 1 association |
| Population-level disease-stage trajectory (HD→MGUS→SMM→MM) | Achievable on existing data | "Population-marginal cell-state evolution along the MGUS-to-MM axis" — never per-patient |
| Per-patient temporal forecasting | **Blocked at full-cohort scale; achievable as exploratory sub-cohort analysis in N=51 (RNA) / N=74 (CNV) paired patients, framed as such.** Of the 787 MMRF patients with any RNA-Seq, only 51 have RNA at ≥2 distinct visits (visit-count distribution: 736×1, 44×2, 5×3, 2×4); of the 908 with any CNV/WES, only 74 have profiling at ≥2 visits. Paired pre/post relapse biopsies: 319 "Recurrent Blood Derived Cancer - Bone Marrow" samples vs 2,592 baseline tumor BM samples. Statistical power is limited; report effect sizes with bootstrap CIs and pre-specify the sub-cohort split. *Verified live 2026-05-03 via `pull_mmrf_clinical.py` + direct pandas audit on `data/raw/mmrf_commpass/{samples,clinical,treatments}.tsv`; cytogenetics flags via `scripts/derive_cytogenetics.py` (logged to `logs/preprocess/cytogenetics.log`).* | Headline result is the population-marginal trajectory on the full N=994; the per-patient forecast is a *secondary, exploratory* result framed as "preliminary evidence on a 51-patient longitudinal sub-cohort, requires prospective replication" |
| Pathway-mediation causal claims | Blocked structurally | Cannot claim without perturbation data (DepMap CRISPR or Cohen-style prospective trial) |

The "predict which state, when, via which pathway, before it happens" goal is the
research **vision**, not the v9 deliverable. v9 ships layer 1 + layer 2 only, both
honestly framed; the per-patient sub-cohort forecast is a v10 prototype previewed in §3.4.

---

## 3. Architecture decision

### 3.1 Trajectory dynamics: MIOFlow primary + PRESCIENT cross-check

**First-principles math justification** (lifted from TRAJECTORY_ARCHITECTURE_AND_DATA.md):

- KL(P_t || P_{t+1}) diverges when state supports are disjoint. MM-resistant states have ≈zero density in HD samples → KL is undefined. Rules out KL-based objectives.
- MMD with bounded kernel is finite under disjoint supports but is non-identifiable on the trajectory itself (multiple flow fields produce the same boundary distributions).
- **Dynamic Wasserstein-2** (Benamou–Brenier 2000) picks a unique minimum-kinetic-energy
  geodesic between marginals: W₂² = inf ∫₀¹ E_{ρ_t}[‖v(x,t)‖²] dt subject to the continuity
  equation ∂_t ρ + ∇·(ρv) = 0. This is well-defined under disjoint supports and yields a
  unique optimal velocity field.
- **MIOFlow** (Huguet & Krishnaswamy 2022, MIT, KrishnaswamyLab/MIOFlow) extends W₂ onto
  the data manifold via a geometry-preserving autoencoder. Replace the default PHATE
  autoencoder with ResistanceMap's existing fusion-VAE so the latent space is shared.
- **PRESCIENT** (Yeo & Theis 2021, MIT, gifford-lab/prescient) yields an interpretable
  potential ψ via dx = -∇ψ dt + σ dW. The basin structure of ψ aligns directly with the
  existing `MemoryStabilityScorer`'s bistability framing — keep MemoryStabilityScorer as a
  complementary basin-depth side score.
- The 1-D pseudotime regressor in the current trajectory stage is retired to baseline-only.

**Build verdict:** adopt + extend, do not rebuild. Fork both repos.

### 3.2 Single-cell encoder: scGPT (tdc mirror)

From `SCFOUNDATION_MODEL_PICKS.md`, verified 2026-05-03:

- **Pick:** `tdc/scGPT` — verified mirror of bowang-lab/scGPT (the upstream HF repo
  returns HTTP 401 without auth; the tdc mirror is unauthenticated, MIT, 553 downloads,
  sha `acf749f3...`).
- **PMID 38409223** (Cui et al., Nat Methods 2024) — efetch-verified.
- **Why scGPT and not Geneformer:** scGPT gives a 512-d frozen embedding per cell that
  drops directly into MIOFlow's encoder slot; ~33 M-cell CELLxGENE pretraining covers
  bone-marrow lineages. Geneformer V2 (Genecorpus-104M, hidden 768 — the original draft
  said 512 was wrong) is the better fine-tuning substrate but the user is compute-constrained
  and frozen-only is the right first move.
- **Backup:** Geneformer-V2-104M_CLcancer (cancer-tuned variant, Apache-2.0 verified) for
  v10 fine-tuning experiments.
- **Avoid:** scBERT (GPL-3.0 — copyleft risk for any clinical deployment).

### 3.3 Drug encoder: Morgan fingerprints from `data/raw/drug_metadata.json`

11 drugs, all SMILES + ChEMBL ID + PubChem CID + MoA + target gene symbol curated 2026-05-03.
Wire `MultiOmicsDataset.drug_features: torch.Tensor` of shape `(11, 2048)` from RDKit
`AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)`. Concatenate per-(sample,
drug) example before the regression head in the fusion stage. Cyclophosphamide's target is
"DNA" (no HGNC symbol) — exclude it from any pathway-graph integration but keep it in the
regression target set.

This is the single highest-leverage architectural change for layer 1 — every published
cell-line drug-response model that beats per-drug means uses drug structural features.

### 3.4 Per-patient extension (v10 path)

The MMRF data landing (finding 1.4) opens a per-patient forecast track that was not
runnable when §3.1–§3.3 were drafted. The MIOFlow + PRESCIENT + scGPT pick is unchanged —
that stack handles the population-marginal trajectory. The per-patient extension is
designed in four sibling docs being written in parallel today (2026-05-03):

- **`PERPATIENT_LATENT_DYNAMICS.md`** — per-patient latent SDE / neural ODE design on
  the N=51 RNA-Seq longitudinal sub-cohort and the N=319 paired pre/post relapse biopsy
  set; shared encoder with the population MIOFlow but per-patient drift parameters.
- **`CAUSAL_SURVIVAL_HEAD.md`** — time-to-resistance / time-to-progression survival head
  using MMRF treatment records (≥2 lines = resistance event; N=258) with covariate
  adjustment for the cytogenetic confounders the causal audit named.
- **`FOUNDATION_MODEL_ADAPTATION.md`** — adapter / LoRA pathway for fine-tuning scGPT (or
  Geneformer-V2-104M_CLcancer) on the MMRF bulk RNA — bridges frozen-encoder limits when
  per-patient signal is needed.
- **`PERPATIENT_VALIDATION_PROTOCOL.md`** — pre-specified evaluation protocol for the
  exploratory per-patient claim: leave-one-patient-out on N=51, paired pre/post
  Wasserstein contrast on N=319, bootstrap CIs, calibration on the held-out timepoint.

This document does not duplicate their content — they are the source of truth for the v10
per-patient track. The relevant integration constraints back into §3.1–§3.3 are: the MIOFlow
+ scGPT latent space is shared between population and per-patient heads; per-patient drift
parameters are added on top, never replace, the population vector field.

---

## 4. Evaluation: SOTA bar + clinical rubric + observability

### 4.1 The publishable bar (from SOTA_BENCHMARK_PROTOCOL.md)

| Criterion | Threshold | Comparator (verified PMIDs) |
|---|---|---|
| Wasserstein-1 reduction on GSE271107 LOPO held-out timepoint | ≥15 % vs previous-timepoint baseline AND statistically indistinguishable from re-implemented TrajectoryNet/MIOFlow on locked split | TrajectoryNet PMID 34337419 (EB EMD 0.784 vs baselines 1.309/1.302); MIOFlow PMID 37397786 (EB W₁ at t=2 = 25.744 vs 33.415) |
| Time-to-resistance C-index | ≥0.65, lower-CI > 0.55 | No published MM baseline — we define the bar |
| Pathway accuracy@5 | ≥0.50 | (defines the bar) |
| Brier score on state predictions | ≤0.20 | calibration check |

Note: criterion 2 (C-index) is now **runnable** as of 2026-05-03 — MMRF CoMMpass labels
are on disk (`treatments.tsv` 7,184 records, N=258 with ≥2 lines as the resistance event;
`clinical.tsv` 995/995 with survival). The C-index threshold is no longer a spec; Sprint 3
will execute it.

### 4.2 Clinical actionability rubric (from CLINICAL_ACTIONABILITY_RUBRIC.md)

5 resistance states: RS-PI / RS-IMiD / RS-DUAL / RS-BCL2 / RS-EMD.

Hard rule: **ACT-SWITCH (recommending a specific regimen) is PROHIBITED** per FDA SaMD
framework. Output may say "consider clinical trial X" but never "switch to teclistamab."

5-point Likert. Realistic near-term target is **Tier 3** ("order confirmatory test") —
does not require treatment-switch correctness, only enough signal to justify a standard-of-
care diagnostic the oncologist would order anyway. Tiers 4/5 gated on MMRF CoMMpass
longitudinal lead-time validation.

OOD detector (mandatory): coverage check + range check + Mahalanobis distance in VAE latent
space. Panobinostat and Romidepsin are hard-excluded (known failure modes in v8).

Cohen's κ ≥ 0.6 across a 30-vignette calibration set, two MM specialists.

UNVERIFIED in the rubric: 5 cited NCT IDs (CARTITUDE-4, KarMMa-3, MajesTEC-3, DREAMM-8,
OCEAN) — c-trials MCP was denied; agent self-flagged. Validate via ClinicalTrials.gov v2
REST before paper use.

### 4.3 Observability: 14 metrics (from resistancemap/observability/SPEC.md)

| Axis | Live | Partial (spec divergence) | True gap |
|---|---|---|---|
| Observability (4) | O1 trace duration | O2 handoff latency, O3 cost-per-request, O4 tool latency | — |
| Evaluation (5) | — | E1 task completion, E2 guardrail violation, E5 first-pass approval | E3 factual accuracy verifier, E4 clinical Likert (now unblocked by rubric) |
| Optimization (5) | — | P1 prompt token efficiency, P2 retrieval precision@k, P4 flow step efficiency | P3 handoff success producer/consumer, P5 weekly improvement velocity |

Existing infrastructure: `resistancemap/agentops/{tracer,evaluator,optimizer,dashboard}.py`
(466 LOC, already wired in main.py, emits `logs/agentops_dashboard.json`).

New addition: `resistancemap/observability/instrumentation.py` — append-only JSONL
emitter with lock+flush+fsync, inert by default. 3 smoke tests passing in 2.52 s. Stdlib
only. Decision on whether to consolidate with existing agentops/ is deferred — the new
module is non-conflicting.

Environment gotcha: `pyproject.toml` `addopts` includes `--cov=resistancemap` but
`pytest-cov` is not installed in the venv. Either install it or remove the entry.

---

## 5. Data pull plan (from MANUAL_DATA_PULLS.md)

Ranked by leverage given the honest scope above:

1. **MMRF CoMMpass IA22** — ✅ **DONE 2026-05-03**. ~3.7 GB processed
   (4.1 GB on disk including raw STAR/MAF/CNV files) under `data/raw/mmrf_commpass/`,
   all 100 % open-access GDC tier — no dbGaP, no DAR application required. The seven
   primary tables (`clinical.tsv`, `treatments.tsv`, `samples.tsv`, `gene_expression.tsv`,
   `mutations.tsv`, `copy_number.tsv`, `cytogenetics.tsv`) are written and row-counted in
   finding 1.4. ~~start the DAR application TODAY at `mmrf.formstack.com/forms/mmrf_virtual_lab_access_request`. This is the only path to Tier 2. Application takes weeks. Until it lands, every claim in any submission is capped at Tier 1 / Pearl L1.~~ Superseded — the open-access GDC route avoided the Formstack DAR entirely. The Tier-1 cap on submissions remains in force *only* until the layer-1 model retrains and produces a runnable cross-validation against MMRF (Sprint 3, criterion 2 of §4.1).
2. **DepMap 26Q1 WES driver mutations** (`OmicsSomaticMutationsMatrixDamaging.csv`).
   Best leverage-per-effort while waiting on MMRF. Adds the unmeasured cytogenetic
   confounders that the causal audit named (chr1q21, del17p, t(4;14), t(11;14)).
3. **DepMap 26Q1 RNA-seq expression** (`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
   — note the file was renamed from the 24Q4 name). ~1,500 cell lines vs 886 with proteomics.
4. **DepMap Achilles CRISPR** (already configured at `data/raw/depmap/CRISPRGeneEffect.csv`).
   Already on disk — wire into harmonize_omics as a Tier 3-adjacent post-hoc validation
   oracle. Cohen et al. (PMID 33619369) is the published template.
5. **HMCL Keats Lab MM panel** (gated through the same MMRF Formstack DAR). MM-specific
   cell lines. Closes the "MM claim, no MM data" gap.
6. **GDSC2** — Sanger COG bucket (the cancerrxgene.org bulk URL is HTTP 410). Cleaner
   IC50 ranges than GDSC1.

Open Targets MM EFO_0001378 returned 5,639 associated targets, top-5 = CRBN 0.706,
TNFRSF17 0.678, TP53 0.667, LIG4 0.660, KRAS 0.660 — useful as a sanity-check oracle for
any pathway-attribution output.

---

## 6. Sequenced sprint plan

### Sprint 1 (this week, no blockers)
- [ ] Wire Morgan fingerprints into `MultiOmicsDataset` (Section 3.3); rerun ElasticNet
      and ResistanceMap layer-1 with drug features. Target: any drug beats predict-mean
      pooled.
- [ ] Switch drug-response target from IC50 → AUC (PRISM/GDSC both ship AUC for every
      curve, no NaN-from-non-crossing).
- [x] ✅ **Done 2026-05-03**: MMRF CoMMpass IA22 acquired via open-access GDC route
      (no DAR needed). See §5 item 1 and finding 1.4 for table-by-table inventory.
- [x] ✅ **Done 2026-05-03**: MMRF RNA-Seq (859 STAR-Counts files → `gene_expression.tsv`)
      and WES (1,091 masked MAFs → `mutations.tsv`) pulled. CNV (1,010 segments →
      `copy_number.tsv`) also done in the same pull window.
- [ ] Pull DepMap 26Q1 WES driver mutations + RNA-seq (Section 5, items 2–3) — still open;
      MMRF does not replace DepMap (different cohort, different covariate structure).
- [ ] Resolve the `pytest-cov` gotcha (Section 4.3).
- [ ] Validate the 5 cited NCT IDs in CLINICAL_ACTIONABILITY_RUBRIC.md against
      ClinicalTrials.gov v2 REST.

### Sprint 1.5 (inserted 2026-05-03 — v10 per-patient prototype, runs in parallel with Sprint 2)
- [ ] Build the longitudinal sub-cohort manifest: join `samples.tsv` × `gene_expression.tsv`
      to materialize the N=51 RNA-Seq paired-visit set and the N=319 recurrence-biopsy set
      as deterministic CSV indices under `data/processed/mmrf_longitudinal/`.
- [ ] Pre-specify the per-patient validation protocol and lock it before any model run —
      see `PERPATIENT_VALIDATION_PROTOCOL.md` (in flight today). LOPO on N=51 + paired
      pre/post Wasserstein on N=319, bootstrap CIs.
- [ ] Prototype the per-patient latent SDE head per `PERPATIENT_LATENT_DYNAMICS.md`. Train
      on the N=51 sub-cohort; treat anything beyond that as v10-paper-only, never v9.
- [ ] Wire the time-to-resistance survival head per `CAUSAL_SURVIVAL_HEAD.md` against
      `treatments.tsv` (≥2-lines event, N=258). Cytogenetic adjustment from
      `cytogenetics.tsv` (N=976) per the causal audit's named confounders.
- [ ] Decide foundation-model adaptation strategy (frozen scGPT vs LoRA-fine-tuned scGPT vs
      Geneformer-V2-104M_CLcancer) per `FOUNDATION_MODEL_ADAPTATION.md`.

### Sprint 2 (next 2 weeks)
- [ ] Fork MIOFlow (KrishnaswamyLab/MIOFlow) and PRESCIENT (gifford-lab/prescient) into
      `external/` or as git submodules.
- [ ] Replace MIOFlow's PHATE autoencoder with ResistanceMap's fusion-VAE.
- [ ] Cache scGPT frozen embeddings on the patient scRNA (GSE124310 + GSE271107) — store
      to `checkpoints/scgpt_embeddings_<gse>.pt`. Pin to `tdc/scGPT@acf749f3`.
- [ ] Train MIOFlow on the HD→MGUS→SMM→MM stage marginals; evaluate on dynamic W₁ (the
      protocol in SOTA_BENCHMARK_PROTOCOL.md, criterion (a)).
- [ ] Cross-check trajectories with PRESCIENT; report potential ψ; verify alignment with
      MemoryStabilityScorer basins.

### Sprint 3 (no longer DAR-gated — MMRF is on disk; gated only on Sprint 1 + 2 completing)
- [x] ~~Ingest MMRF CoMMpass IA22 (RNA-seq + WES + IMWG outcomes).~~ ✅ Done 2026-05-03;
      ingest covered RNA-seq + WES + CNV + clinical + treatments + cytogenetics. IMWG
      outcome derivation from `treatments.tsv` is the next step (≥2-lines event).
- [ ] Train layer-1 (drug response) with PERCEPTION-style two-cohort design: cell-lines
      train, CoMMpass (N=994 with treatments) validation. PMID 38637658 is the template.
- [ ] Compute time-to-resistance C-index on N=258 (≥2 lines) — first runnable evaluation
      of SOTA-protocol criterion (b). Target ≥0.65 with lower-CI > 0.55 per §4.1.
- [ ] Wire the clinical rubric's Tier-3 evaluation: synthetic vignettes from a
      held-out CoMMpass slice scored by 2 MM specialists, target Cohen's κ ≥ 0.6.

### Sprint 4 — paired-sample sub-cohort analysis (exploratory, gates after Sprint 3)
Scope: the MMRF longitudinal sub-cohorts identified in finding 1.4 — N=51 RNA-Seq
paired-visit patients + N=74 CNV/WES paired-visit patients + N=319 paired pre/post
relapse biopsies (verified live 2026-05-03 via `pull_mmrf_clinical.py` + direct pandas
audit on `data/raw/mmrf_commpass/{samples,clinical,treatments}.tsv`). This is **NOT a
primary forecaster** — it is a "molecular evolution at relapse" exploratory figure.

- [ ] Materialize the three paired-sample manifests as deterministic CSV indices under
      `data/processed/mmrf_longitudinal/` (already enumerated as a Sprint 1.5 task; this
      sprint *uses* the manifests).
- [ ] Compute paired pre/post Wasserstein-1 distance in the MIOFlow latent space on the
      N=319 relapse-biopsy pairs; bootstrap 95 % CI; compare to a permutation null.
- [ ] LOPO trajectory forecast on the N=51 RNA-Seq paired-visit cohort: encode visit-1,
      forecast visit-2 latent under the MIOFlow vector field, compare to observed visit-2
      embedding. Pre-specify the metric in `PERPATIENT_VALIDATION_PROTOCOL.md` *before*
      running.
- [ ] CNV evolution analysis on the N=74 paired-visit cohort: change in chr1q21 / del17p
      / del13q burden between visits. This is descriptive only — no forecasting claim.
- [ ] Frame all outputs as "exploratory molecular evolution at relapse, N=51-319 paired
      samples, requires prospective replication." Do NOT include in any headline metric;
      do NOT use to claim "predicts resistance before it happens."

### What we explicitly do NOT do
- Do not retrain the existing pipeline before Sprint 1 lands. The 30-min run is wasted
  compute given the floor finding.
- Do not claim per-patient trajectory in any figure caption. Population-marginal only.
- Do not claim pathway mediation. Use "associated with" (Rade et al. PMID 38641734).
- Do not include Panobinostat results in headline metrics — it is a documented failure
  mode that drags pooled MSE.
- Do not pull data that requires the user's credentials without explicit user action
  (MMRF DAR, dbGaP).

---

## 7. Honesty constraints that must propagate

These are not optional. They appear in every figure caption, manuscript abstract, and
release artifact:

1. **Tier 1 / Pearl L1 only**, until the layer-1 model retrains and is cross-validated
   against the on-disk MMRF cohort (Sprint 3). MMRF's arrival removed the data block but
   not the model block — until a runnable PERCEPTION-style two-cohort result lands in the
   ledger, every shipped number stays Tier 1. Frame as "associates static omics with
   drug-sensitivity rankings" or "predicts response" — never "causes" or "drives" or
   "before it happens."
2. **Population-marginal only**, until paired pre/post longitudinal data arrives. Per-
   patient claim is a Bayesian conditional prior on the population dynamics, not a
   verified forecast.
3. **No specific regimen recommendation** in any model output. ACT-SWITCH is
   FDA-prohibited.
4. **Panobinostat and Romidepsin OOD**, hard-excluded from any reported headline metric.
5. **Run-ledger discipline**: any number that lands in README.md or ARCHITECTURE.md must
   trace to a row in the run ledger with stdout.log + config.yaml + verification_chain.json
   + per_drug_metrics.csv + W&B URL. Numbers in this debugging-tier doc are NOT
   ledger-cleared.

---

## 8. Open verification debt

Items still UNVERIFIED across the 6 docs (cannot ship to paper without resolution):

| Doc | UNVERIFIED item | How to resolve |
|---|---|---|
| `MODEL_LANDSCAPE_RANKING.md` | DrugCell/MOLI/DeepCDR Spearman/AUROC numbers | Fetch full PDFs (abstracts don't contain decimals) |
| `SCFOUNDATION_MODEL_PICKS.md` | scGPT zero-shot ARI/NMI; Geneformer V2 PMID; scFoundation GDSC Pearson r | Fetch full PDFs |
| `CLINICAL_ACTIONABILITY_RUBRIC.md` | 5 NCT IDs (CARTITUDE-4, KarMMa-3, MajesTEC-3, DREAMM-8, OCEAN) | curl ClinicalTrials.gov v2 REST |
| `SOTA_BENCHMARK_PROTOCOL.md` | "ML melanoma immunotherapy signature 2024" (27 candidate PMIDs) | Disambiguate via title or abandon as comparator |
| `SOTA_BENCHMARK_PROTOCOL.md` | PRESCIENT day-4 W; MOSCOT/CellRank2 figure-bound scalars | Fetch full PDFs |

---

## 9. The single most important sentence

If only one paragraph survives this document, let it be this one (rewritten 2026-05-03 to
acknowledge that MMRF on disk has shifted the data ceiling; the Tier-1 honest framing is
unchanged for any number that ships before retraining):

*ResistanceMap is **today** a Tier 1 / Pearl L1 associative model of static cell-line
omics → drug-sensitivity rank, indistinguishable from a 19.7-second MOFA+Ridge baseline —
that is the only claim ledger-cleared as of 2026-05-03 and the only claim that may appear
in the README. **Once retrained**, the same model becomes Tier 2 / Pearl L1 via PERCEPTION-
style two-cohort cross-validation against the now-on-disk MMRF CoMMpass cohort
(N=994 patients with treatment records, N=995 with survival), bumping the data ceiling but
leaving the causal tier unchanged. **Exploratory Tier 2+** per-patient temporal forecasts
become falsifiable (with limited statistical power) on the MMRF longitudinal sub-cohort
(N=51 RNA-Seq paired-visit, N=74 CNV paired-visit, N=319 paired pre/post relapse biopsies)
via the v10 per-patient track sketched in §3.4 and detailed in PERPATIENT_LATENT_DYNAMICS.md
+ CAUSAL_SURVIVAL_HEAD.md + FOUNDATION_MODEL_ADAPTATION.md + PERPATIENT_VALIDATION_PROTOCOL.md.
The honest v9 deliverable remains a population-marginal trajectory model along the
HD→MGUS→SMM→MM disease-stage axis using MIOFlow + PRESCIENT on scGPT-frozen embeddings,
benchmarked against TrajectoryNet/MIOFlow re-implementations on dynamic Wasserstein-1, gated
on the publishable bar in §4.1, with explicit Tier-1 framing per PERCEPTION (PMID 38637658).
**Tier 2 is now reachable** on the full N=994 baseline cohort once the layer-1 model
retrains; **Tier 3 still requires DepMap CRISPR** as a perturbation oracle (already on
disk per §5 item 4 but not yet wired as a *post-hoc validation oracle, never as training
input*). Pathway-mediation (Pearl L3) claims remain blocked until that oracle is wired —
MMRF has no perturbation arm of its own.*
