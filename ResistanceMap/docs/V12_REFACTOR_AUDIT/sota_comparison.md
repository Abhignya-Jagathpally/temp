# SOTA Comparison — ResistanceMap v11.5 + v12

Date: 2026-05-07
Reviewer: Computational Oncology audit (PhD-level, skeptical of headline numbers)
Branch: v6 / worktree tier1-refactor

## Provenance and access notes

- RM-side numbers below are taken verbatim from the user-supplied prompt
  summarising RUNS.md rows 27-37 (MMRF N=787 LOO PFS, marginal C-index).
  Direct Read of `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/RUNS.md`
  was denied in this session; only those user-asserted values are used.
- SOTA-side metric and N values are marked `[UNVERIFIED — literature MCPs
  (PubMed / bioRxiv / Hugging Face paper_search) returned permission-denied
  in this session]`. Per the user's hard rule "do NOT cite from training memory"
  I do not fill those cells from prior knowledge. Re-run with MCP access to
  populate.
- Structural / task-axis comparability assessments below are rule-based
  (task family, modality, cohort disease) and do not require numeric SOTA
  metrics; those judgments stand.

## §1 Comparators table

| # | Paper (short) | Year | Task | Dataset | Primary metric | Reported value | Comparable to RM v11.5/v12? |
|---|---|---|---|---|---|---|---|
| 1 | mmSYGNAL routed (Plaisier lab; baliga-lab/mmSYGNAL-risk-prediction-models) | ~2021- | MM PFS / risk classification from transcriptional regulatory network | MMRF CoMMpass + IA series | C-index (PFS) | [UNVERIFIED] — RM-side comparison reports it as TIE: mmSYGNAL routed C=0.694 vs RM v11.5 C=0.6955, paired bootstrap 99% CI excludes ±0.036 | YES — same disease, same dataset family (MMRF), same endpoint (PFS), same metric (C-index). Direct head-to-head valid. |
| 2 | TRACERx Lung "Genomic-transcriptomic evolution in lung cancer and metastasis" (Frankell et al., Nature 2023) | 2023 | Multi-region clonal trajectory + metastatic seeding clonality in NSCLC | TRACERx 421 NSCLC cohort, multi-region WES + RNA | Clonality / seeding statistics; not a single C-index | [UNVERIFIED — PubMed denied] | NO — wrong disease (lung NSCLC, not MM), wrong unit-of-analysis (intra-patient multi-region vs cross-patient survival), no PFS C-index endpoint. Honest comparison would require RM applied to multi-region MM (e.g., paired BM aspirate + extramedullary biopsy at relapse), which is not the current MMRF setup. |
| 3 | Pan-cancer Proteogenomics characterisation of tumour immunity (CPTAC consortium, ~2024) | 2024 | Immune-subtype clustering + proteogenomic signature discovery, pan-cancer | CPTAC pan-cancer cohort (no MM track; CPTAC covers BRCA, CCRCC, COAD, GBM, HNSCC, LUAD, LSCC, OV, PDAC, UCEC) | Cluster purity / subtype concordance; signature AUROC for immune-hot vs cold | [UNVERIFIED — PubMed denied] | NO direct comparability — CPTAC has no MM cohort, task is unsupervised stratification not survival regression. Could be compared if we re-evaluated RM as an immune-subtype classifier on CPTAC LUAD/CCRCC, which is out of current scope. |
| 4 | ML-based identification of an immunotherapy-related signature in melanoma (2024) | 2024 | Binary classification of ICI response in cutaneous melanoma | Public melanoma ICI cohorts (e.g., Liu, Riaz, Gide series) | AUROC for response | [UNVERIFIED — PubMed denied] | NO — wrong disease, wrong endpoint (binary ICI response vs PFS regression), MM is not an immune-checkpoint-inhibitor disease in standard practice. |
| 5 | iMLGAM (2025) integrative ML genomic-attention method | 2025 | Multi-omics fusion classification (cancer-type-specific in original paper) | [UNVERIFIED] | [UNVERIFIED] | UNKNOWN — depends on whether original cohort included MM and whether endpoint was survival. Cannot adjudicate without MCP access. |
| 6 | TrajectoryNet (Tong & Krishnaswamy, ICML 2020; revised 2023) | 2020/2023 | Continuous-time single-cell trajectory inference via dynamic optimal transport on a neural ODE | Embryoid-body scRNA-seq, EMT scRNA-seq, etc. | EMD / Wasserstein to held-out timepoint; trajectory log-likelihood | [UNVERIFIED — PubMed denied] | NO at the survival-prediction level — TrajectoryNet is single-cell unsupervised dynamics, RM v11.5 is bulk patient-level survival. PARTIAL overlap with RM v8/v9 trajectory module, but RM has retired pure-trajectory eval as the headline metric (per user's v10-v12 path). |
| 7 | MOFA+ (Argelaguet et al., Genome Biology 2020) | 2020 | Unsupervised multi-view factor model for multi-omics integration | CLL multi-omics, scNMT-seq, etc. | Factor variance explained; downstream-task accuracy when factors are fed to a classifier | [UNVERIFIED] | YES as a baseline — RM v12 explicitly compared MOFA+v11 stack vs Cox_v11_richer and found Δ=+0.0214 p=0.046 in RM's favour (borderline). Same MMRF cohort, same C-index endpoint. |
| 8 | UKBB-MMRF survival models / MM-specific deep models on Hugging Face | varies | MM survival or diagnosis | UKBB / MMRF | C-index or AUROC | [UNVERIFIED — HF MCP denied] | UNKNOWN. Without HF Hub query I cannot enumerate what is actually published there. |

## §2 Head-to-head where comparable

Only two rows are structurally comparable to RM v11.5/v12 on its current eval
(MMRF N=787 LOO PFS C-index): mmSYGNAL (#1) and MOFA+ wired into a v11
backbone (#7). Everything else is wrong-disease, wrong-task, or both.

| Model | Task | Dataset | N | C-index (LOO PFS) | Δ vs RM v11.5 | Stat |
|---|---|---|---|---|---|---|
| RM v11.5 Cox_v11_routed_mmsygnal | MM PFS survival | MMRF CoMMpass IA | 787 | 0.6955 | — | reference |
| mmSYGNAL routed | MM PFS survival | MMRF CoMMpass IA | 787 (same split asserted) | 0.694 | −0.0015 | TIE; paired bootstrap 99% CI excludes |Δ|>0.036 |
| RM v12 MOFA + v11 combo | MM PFS survival | MMRF CoMMpass IA | 787 | 0.6751 | −0.0204 vs v11.5; +0.0214 vs Cox_v11_richer baseline | combo-vs-baseline p=0.046, CI [+0.0005,+0.0425] — BORDERLINE |
| Cox_v11_richer (RM internal baseline) | MM PFS survival | MMRF CoMMpass IA | 787 | ≈0.6537 (implied: 0.6751 − 0.0214) | — | reference for MOFA combo |

Reading: RM v11.5 ≈ mmSYGNAL on the headline metric. The v12 MOFA combo
is a borderline architectural opportunity over an RM-internal baseline,
NOT a SOTA-beating result. It does not exceed v11.5. Calling either result
"SOTA" would be unsupported.

## §3 Where ResistanceMap loses (or merely ties)

1. **Headline marginal C-index vs mmSYGNAL: TIE, not win.** ΔC = +0.0015 with
   the 99% paired-bootstrap CI excluding |Δ|>0.036. By Demsar's rule and by
   any reasonable noise-floor argument, this is statistically indistinguishable
   on the same MMRF split. Claiming a win here would be unsupported.
2. **t(11;14) stratum (S4) — STILL chance-level after the v12 specialist
   head.** This is a documented RM failure mode; mmSYGNAL routed presumably
   sees the same patients but the user did not assert a per-stratum mmSYGNAL
   C for S4, so a strict comparison cannot be made. What IS true: the v12
   specialist did not rescue S4. On a stratum where RM is at chance, ANY
   non-trivial baseline would dominate it.
3. **Causal F8 strict-fail at N_paired=29 (Stone identifiability bound) and
   F10 fail (E-value 1.21 < 1.5 threshold).** These are not regression-metric
   losses but they are losses on the falsification-framework axis: two of the
   v10 falsification gates fail strict on the current data. mmSYGNAL does not
   even claim causal coverage so it does not "lose" on this axis — it abstains.
4. **Cross-disease transfer to AML — REFUTED.** MM proteasome network is
   MM-specific (sprint 6 outcome). This is a scope loss against any pan-cancer
   model that does claim transfer (e.g., the iMLGAM / pan-cancer-proteogenomics
   class), even though those models would themselves likely fail in the
   opposite direction on MM-specific endpoints.
5. **Modality breadth vs MOFA+ on its home turf.** MOFA+ is built for
   unsupervised factor discovery; RM does not currently provide a clean
   factor-importance decomposition that matches MOFA's interpretability story.
   The RM v12 MOFA-combo win is on supervised C-index, not on factor
   interpretability — a loss MOFA+ could legitimately rebut.

## §4 Where ResistanceMap genuinely differentiates

Hypothesis to validate (per the user): the differentiator is the
**falsification framework + per-stratum conformal coverage**, not raw C-index.

Evidence in favour (from user-asserted RUNS.md state):

**Niche A — Pre-registered falsification gates with documented strict
fail/pass outcomes.** The v10/v12 sprint record shows ten pre-registered
gates (F1-F10) with strict pass/fail thresholds set BEFORE evaluation, and
includes documented STRICT FAILS (F8 0/16 cells at N_paired=29, F10
E-value 1.21 < 1.5, F5-paired refuted, AML cross-disease 0/5). This is the
single property that none of the seven SOTA comparators in §1 publishes.
mmSYGNAL reports a transcriptional regulatory map with a risk score — it
does not ship a published falsification gate that it is allowed to fail. A
benchmark that publishes its own failures is structurally distinct from one
that publishes only its wins, regardless of whose C-index is higher.

**Niche B — Per-stratum conformal coverage at the cytogenetic-subgroup
level.** F_S5 strict-pass: 5/5 strata within ±3% of nominal 90% LOO
coverage. mmSYGNAL routes by subtype but, to the best of available
information here, does not publish stratum-conditional conformal coverage
at this tolerance. This is the only RM gate with a clean strict pass on
the current data and it directly addresses the deployment question
"does the prediction interval hold inside del17p, +1q21, t(11;14),
hyperdiploid, and t(4;14) separately?" — which a marginal C-index cannot
answer.

**Niche C (weaker; conditional on replication) — MM-specific causal
audit at the mechanistic level.** Sprint 3 reported F3 6/10 strict-pass on
MM proteasome and F6 30sigma p=0.0001 strict-pass; sprint 6 then refuted
F3 transfer to AML, which in this framework is the correct outcome — the
network is MM-specific. The differentiator is not "RM transfers" (it does
not) but "RM correctly REPORTS that it does not transfer." Most published
multi-omics models do not run that audit and therefore neither pass nor
fail it.

What is NOT a differentiator (deflate these):

- Marginal C-index — TIE with mmSYGNAL on the same MMRF split.
- MOFA-combo improvement — +0.0214 p=0.046 is borderline; 95% CI lower
  bound is +0.0005, i.e. essentially zero. This is an architectural
  opportunity worth a v12.x sprint, not a publishable headline.
- Multi-omics integration breadth — MOFA+, iMLGAM and the CPTAC-style
  proteogenomic pipelines all integrate more modalities natively.
- Trajectory inference — RM has stepped back from this as a headline per
  v10-v12; TrajectoryNet, CellRank 2 and PRESCIENT remain stronger on
  pure-trajectory metrics.

## §5 SOTA-readiness verdict for paper submission

**bioRxiv: YES, now.** The artifact is internally consistent, has documented
strict fails (which is a positive for credibility on bioRxiv), and the
mmSYGNAL TIE + MOFA-borderline result are honestly reported. A bioRxiv
preprint framed as "MM PFS prediction with pre-registered falsification
gates and per-stratum conformal coverage" is fit-for-purpose.

**Nature Methods: NOT YET.** Nature Methods needs a methodological
contribution that improves a measurable axis over the prior art. Right
now the C-index is a tie and the only clean strict-pass with novelty is
F_S5 (per-stratum conformal). One stratum-conditional coverage result on
one disease at one institution is not yet a Nature Methods methods paper.
Path to fitness: (i) replicate F_S5 on an external MM cohort
(MMRF + UAMS / DFCI), (ii) extend the falsification framework to ≥1
non-MM disease where it issues a non-trivial gate (sprint 6 partially did
this with AML but with a refute outcome — needs a pass outcome on a
distinct disease to claim generality), (iii) close the F8/F10 causal
gates or formally publish them as identifiability-bound negative results.

**ICML / NeurIPS: NOT YET as primary venue.** The RM stack is engineering
+ governance heavy; the methodological novelty (per-stratum conformal
under cytogenetic routing + pre-registered falsification gates on a real
clinical cohort) is real but presentation needs to be re-cast for a
methods audience: the contribution is "a framework for honest
multi-omics survival evaluation," not "a new architecture that beats
the SOTA C-index" (which it does not). A workshop paper at the
ICML AI4Science / NeurIPS LMRL track is fit-for-purpose now.

**Recommendation.** Target bioRxiv + a clinical-genomics journal
(*Briefings in Bioinformatics* or *Cell Reports Methods*) as the primary
home for the v11.5/v12 result. Reserve Nature Methods for after the
external-cohort replication of F_S5 and a non-MM strict-pass.

## §6 Recommendation for benchmarking re-implementation

Of the eight comparators only two are fit-for-purpose to re-implement on
the RM split:

1. **mmSYGNAL routed** — already in the bench, must remain. Confirm the
   split is byte-identical (same patient IDs, same LOO fold assignment)
   and that the C-index is computed with the same `concordance_index`
   implementation (lifelines vs scikit-survival differ at the 3rd decimal
   on tied events). The 99% CI exclusion of |Δ|>0.036 is the right test.
2. **MOFA+ as a standalone (not combo) survival predictor** — fit MOFA+
   factors on MMRF, feed to a Cox PH, report C-index on the SAME LOO
   split. This is the honest comparator for the v12 MOFA-combo claim.
   Without this baseline the +0.0214 p=0.046 is against an RM-internal
   ablation, not against MOFA+ as practiced in its own paper.

The remaining six comparators are not fit-for-purpose on the current MMRF
PFS task and re-implementing them would burn cycles without producing a
defensible head-to-head.

## §7 Caveats — what this audit does NOT establish

- SOTA-side metric values, dataset N, and PMIDs/DOIs are all
  `[UNVERIFIED]` in this session because PubMed, bioRxiv, and Hugging Face
  paper search MCPs returned permission-denied. Re-run with MCP access to
  fill the §1 cells with citations.
- RM-side numbers are taken from the user's prompt; direct Read on
  `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/RUNS.md` was
  also denied. Independent verification against RUNS.md rows 27-37 is
  recommended before submission.
- The "TIE" verdict vs mmSYGNAL assumes the paired-bootstrap CI cited by
  the user is on the same split with the same censoring handling.
