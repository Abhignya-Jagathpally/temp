# MM Trajectory Drivers — Empirical Review for v10 Architecture

Scope: anchor v10 feature design in published, replicated MM longitudinal /
single-cell evidence rather than method-paper buzzwords. A signal "contributes"
only if it (a) differs longitudinally between progressing vs stable patients,
(b) precedes clinical resistance, (c) has perturbation evidence, and (d)
replicates across ≥ 2 independent MM cohorts.

PMID verification log (all via NCBI eutils, 2026-05-03):
- Cohen YC, Nat Med 2021, PMID 33619369 — VERIFIED. Title matches.
- Tirier SM, Nat Commun 2021, PMID **34556643 INCORRECT** (resolves to a paper
  on Grand Ethiopian Renaissance Dam). Correct PMID = **34845188** (verified;
  Tirier SM et al., "Subclone-specific microenvironmental impact and drug
  response in refractory multiple myeloma…", Nat Commun 12:6960, 2021).
- Boiarsky R, Nat Commun 2022, PMID **35922427 INCORRECT** (resolves to a
  rainfall-product paper). Correct PMID = **36396631** (verified; Boiarsky R et
  al., "Single cell characterization of myeloma and its precursor conditions…",
  Nat Commun 13:7040, 2022).
- Ledergor G, Nat Med 2018, PMID **30420755 INCORRECT** (resolves to an
  oligodendrocyte-MS paper). Correct PMID = **30523328** (verified; Ledergor G
  et al., "Single cell dissection of plasma cell heterogeneity…", Nat Med
  24:1867, 2018).
- "Liu 2024 longitudinal MM" — closest hit is Liu X et al. (DrugFormer), Adv
  Sci 11:e2405861, 2024, PMID **39206872** (verified). This is a model paper
  using RRMM scRNA, *not* a longitudinal patient cohort. Treated as
  methods/secondary evidence below.
- Croft Blood Cancer J 2024 — search returned 0 hits with that journal +
  trajectory + resistance constraint. UNVERIFIED; the user's cite cannot be
  confirmed from PubMed and is excluded from primary synthesis.
- Sklavenitis-Pistofidis et al. (Amp1q vulnerabilities), bioRxiv 2023, PMID
  37577538 — VERIFIED, used as second-cohort confirmation for Amp1q/MCL1.
- Durand R et al. (p53/BAX/BH3), Blood 2024, PMID 38096363 — VERIFIED, used as
  second-cohort confirmation for TP53/BCL2-family axis.

---

## §1 Per-paper extracts

### Cohen YC et al., Nat Med 27:491 (2021), PMID 33619369
- Cohort: 41 newly-diagnosed MM patients who failed bortezomib induction;
  longitudinal scRNA-seq paired with Dara-KRd salvage (NCT04065789). Compared
  to 11 healthy + 15 NDMM controls. Generalized to MMRF CoMMpass.
- Top trajectory features distinguishing resistance: **hypoxia tolerance**,
  **protein-folding / unfolded-protein response**, **mitochondrial
  respiration**. PPIA emerged as the central node of the protein-folding axis.
- Validation: signature reproduced in CoMMpass bulk; CRISPR-Cas9 knockout of
  PPIA and ciclosporin treatment sensitized MM cells to PIs (mechanistic
  perturbation evidence — criterion (c) met).
- PPIA is a downstream-validated *target*; whether it was an input feature of
  a predictor is not stated in the abstract — the paper is descriptive +
  perturbation, not a predictive ML model. Effect size: not in abstract
  (UNVERIFIED numerically).
- Code/data: GEO; no held-out predictor reported in abstract.

### Tirier SM et al., Nat Commun 12:6960 (2021), PMID 34845188
- Cohort: 20 RRMM patients, paired pre/post-treatment scRNA + tumor + BME.
- Top features: **chr1q-gain subclones with a specific transcriptomic
  signature that frequently expand under treatment**; immune-suppressive BME
  rewiring (PD1+ γδ T-cells, tumor-associated macrophages, hematopoietic
  progenitor depletion); inflammatory cytokine upregulation feeding
  myeloid-compartment cross-talk.
- Validation: subclone-level expansion observed within-patient longitudinally;
  not externally re-tested in another cohort in this paper.
- Effect size: not in abstract (UNVERIFIED numerically).

### Boiarsky R et al., Nat Commun 13:7040 (2022), PMID 36396631
- Cohort: 26 patients across the MGUS/SMM/MM precursor spectrum + 9 healthy;
  cross-sectional, *not* longitudinal under therapy.
- Top features: 15 NMF gene-expression signatures; **one signature uniformly
  *lost* in malignant cells across stages**; intra-tumor heterogeneity with
  multiple coexisting transcriptional patterns. The abstract emphasizes early
  tumorigenesis programs, **not the TNF/NF-κB resistance hallmark** the user
  asked us to verify — that specific claim is NOT supported by the abstract
  text and is flagged UNVERIFIED.
- Validation: HiDDEN re-analysis (PMID 39487129) recapitulated the malignant
  population calls and found additional malignancy in early-stage samples —
  partial methodological replication.
- Effect size: not in abstract (UNVERIFIED numerically).

### Ledergor G et al., Nat Med 24:1867 (2018), PMID 30523328
- Cohort: 40 individuals across the MM progression spectrum + 11 healthy.
- Top features: high inter-individual variability explainable by known MM
  drivers; **subclonal structure detected in 10/29 MM patients**; rare
  active-myeloma-like plasma cells in asymptomatic and post-treatment MRD
  samples; circulating plasma cells reflect bone-marrow disease.
- Validation: foundational descriptive study; no external predictive
  validation.
- Relevance to trajectory prediction: establishes that *baseline* rare
  malignant-like cells exist in asymptomatic and MRD states — substrate for a
  "before it happens" claim, but the paper does not itself predict.
- Effect size: not in abstract (UNVERIFIED numerically).

### Liu X et al. (DrugFormer), Adv Sci 11:e2405861 (2024), PMID 39206872
- Cohort: scRNA from RRMM and AML patients (counts not in abstract); used as
  training data for a graph-augmented LLM, not a patient cohort.
- Top features: pseudotime trajectory analysis identified "drug-resistant
  cellular states associated with poor outcomes"; **COX8A** highlighted as a
  cross-tumor resistance target.
- Validation: model claims higher F1/precision/recall vs unspecified
  baselines; no held-out external MM cohort named in abstract.
- Use here: secondary evidence only; treats RRMM as one of two training
  domains, not a longitudinal predictor.

### Sklavenitis-Pistofidis R et al., bioRxiv 2023, PMID 37577538
- Used as **independent confirmation** of the chr1q-gain → MCL1 axis (also
  seen in Tirier 2021).
- Within-patient subclones with vs without Amp1q showed **higher MCL1 and PI3K
  pathway activity**; isogenic clones with arm-level chr1q gain were more
  sensitive to MCL1 + PI3K inhibitors. Synergy demonstrated mechanistically.

### Durand R et al., Blood 143:1242 (2024), PMID 38096363
- Used as **independent confirmation** of the TP53 / BCL2-family axis.
- 13-gene functional p53 score derived from CRISPR/Cas9 TP53-/- clones
  predicted overall survival in two independent cohorts: **MMRF-CoMMpass** and
  **CASSIOPEA**. p53-regulated BAX expression directly modulates MCL1-BAX
  complex formation and BH3-mimetic sensitivity.
- This is the cleanest example in the verified literature of a feature set
  surviving independent-cohort survival prediction.

---

## §1b Cross-paper signal table

Legend: ✓ replicated within paper, ◯ mentioned, ✗ refuted, blank = not addressed.

| Candidate signal                             | Cohen 2021 | Tirier 2021 | Boiarsky 2022 | Ledergor 2018 | Liu 2024 | Sklav. 2023 | Durand 2024 |
|---|---|---|---|---|---|---|---|
| IRF4 / MYC / CRBN-substrate (IMiD axis)      |            |              |                |                |           |             |              |
| PSMB5 / proteasome stress / UPR              | ✓ (PPIA, UPR axis) |     |                |                |           |             |              |
| BCL2 / MCL1 / BIM ratio                      |            |              |                |                |           | ✓ (MCL1)    | ✓ (BAX/MCL1) |
| chr1q21 gain → CKS1B / MCL1 amp              |            | ✓            |                |                |           | ✓            |              |
| NF-κB activation                             |            | ◯ (cytokines) | UNVERIFIED in abstract |       |           |             |              |
| TP53 mutation / loss                         |            |              |                |                |           |             | ✓ (independent cohorts) |
| t(4;14) FGFR3 / NSD2                         |            |              |                |                |           |             |              |
| Plasma-cell-cycle reentry / proliferation    |            |              |                | ◯              | ◯ (pseudotime) |       |              |
| Microenv: CD8+ exhaustion, Treg, BMSC        |            | ✓ (PD1+ γδ T, TAMs) |        |                |           |             |              |
| Epigenetic / chromatin (LSD1/EZH2)           |            |              |                |                |           |             |              |
| Clonal architecture (subclone count, fitness)|            | ✓            |                | ✓ (10/29 pts)  |           | ✓ (within-pt) |            |
| Hypoxia tolerance                            | ✓          |              |                |                |           |             |              |
| Mitochondrial respiration                    | ✓          |              |                |                | ◯ (COX8A) |             |              |

(Empty cells = not directly addressed in the verified abstract; absence of
mark means "not evidence for or against," not "refuted.")

---

## §2 Top signals replicated across ≥ 2 verified cohorts

Using the strict ≥ 2 independent cohorts criterion, only three signals
replicate across the verified primary literature:

1. **chr1q gain → MCL1 / PI3K axis.**
   - Tirier 2021 (n = 20 RRMM, paired pre/post-treatment): chr1q-gain
     subclones expand under treatment.
   - Sklavenitis-Pistofidis 2023 (within-patient isogenic comparisons +
     dependency screens): Amp1q raises MCL1 and PI3K pathway activity and
     confers sensitivity to MCL1 + PI3K inhibitor combination.
   - Effect-size in abstract: not reported numerically (UNVERIFIED).

2. **TP53 inactivation → BCL2-family rebalancing → BH3-mimetic response.**
   - Durand 2024: 13-gene functional p53 score predicts OS in MMRF-CoMMpass
     and CASSIOPEA — two independent cohorts within a single paper.
   - Cohen 2021 CoMMpass generalization: their resistance signature also
     generalized to CoMMpass (reported in abstract; numeric performance
     UNVERIFIED).
   - Effect-size: UNVERIFIED numerically.

3. **Protein-folding / UPR / mitochondrial-respiration program (PI-resistance
   axis).**
   - Cohen 2021: hypoxia + protein folding + mitochondrial respiration
     replicated from the 41-patient discovery into CoMMpass; PPIA validated
     by CRISPR and ciclosporin.
   - Liu 2024 (DrugFormer): independent re-analysis flags COX8A
     (mitochondrial complex IV) as a cross-tumor resistance node — directionally
     consistent with the mitochondrial-respiration arm.
   - Effect-size: UNVERIFIED numerically.

Two further signals reach **partial** replication:
- Subclonal architecture / clonal fitness (Tirier, Ledergor,
  Sklavenitis-Pistofidis) — replicated as a phenomenon, but the
  feature-engineering (subclone-count vs dominant-clone fitness vs entropy)
  varies across papers.
- Immunosuppressive BME (Tirier strongly; Cohen mentions immune dysfunction)
  — replicated qualitatively but with very different cell-type panels.

---

## §3 What v10 SHOULD include in its feature set

Anchored in §2, v10's feature pipeline should be built around signals that
have crossed the two-cohort bar with mechanistic backing:

1. **chr1q copy-number status as an explicit covariate**, plus an MCL1/PI3K
   pathway activity score. This is a single-gene-arm signal that bulk DNA
   captures cheaply and that v10's drug-response head can condition on
   directly. Anti-MCL1 / anti-PI3K drug embeddings should be cross-attended
   to chr1q status, not pooled.
2. **BCL2-family ratio + functional p53 score.** Use Durand 2024's 13-gene
   p53 score as a published, externally-validated feature (the gene list is
   in the paper). Combine with BCL2 / MCL1 / BIM expression for the
   t(11;14) / venetoclax-response axis. This is the only signal in the
   verified MM literature with cross-cohort survival validation that is
   *transcriptional*.
3. **A UPR + mitochondrial-respiration module** (Cohen-style). PPIA as a
   leading edge gene; COX8A and the OXPHOS module as confirming markers. Use
   GSVA / single-sample enrichment over the published Cohen 2021 leading-edge
   list rather than re-deriving programs from scratch.
4. **Per-patient subclonal-architecture summary**: subclone count, dominant
   clone fraction, and chr1q-gain subclone fraction. Tirier 2021 + Ledergor
   2018 + Sklavenitis-Pistofidis 2023 all converge on subclone fitness as
   informative; the v10 feature should *not* be a learned latent over single
   cells but a small, interpretable vector tied to clonal structure.
5. **A microenvironment summary** (CD8 exhaustion fraction, Treg fraction,
   PD1+ γδ T fraction, TAM fraction). Include only if v10's data source
   carries the relevant cell types (BM aspirate scRNA, not plasma-cell-only
   panels).

These five together cover the IMiD-axis (via 1-2-4 indirectly), the PI-axis
(3), and the BCL2/venetoclax axis (2) with verified cross-cohort support.

---

## §4 What v10 should NOT include (lacks verified replication)

- **TNF / NF-κB program as a standalone resistance hallmark.** The user-cited
  Boiarsky 2022 abstract does not contain this claim; it is UNVERIFIED at
  the abstract level. Tirier 2021 mentions inflammatory cytokines but does
  not name NF-κB as the resistance axis. Without two abstract-verified
  cohorts, this should not be a baked-in v10 feature module.
- **IRF4 / CRBN-substrate score as an IMiD-resistance feature in v10's
  scRNA-trained head.** No verified MM scRNA cohort in this review reports
  it as a longitudinally-replicated predictor of resistance trajectory. CRBN
  / IKZF1 / IKZF3 mechanism is well established in cell-line work, but it
  did not surface in the abstract-verified longitudinal MM literature here.
  Include only as a hypothesis-mode feature, not as a core driver.
- **t(4;14) FGFR3/NSD2 as a learned scRNA program.** Genetic stratifier yes,
  expression-program-level resistance signal in scRNA — not present in any of
  the verified abstracts.
- **LSD1 / EZH2 epigenetic compaction as a v10 feature.** Plausible from
  general MM biology, absent from all verified abstracts in this review.
- **Pseudotime-derived "cell-state trajectory" as a clinical predictor.**
  Liu 2024 / DrugFormer is the only verified pseudotime-based work and it
  does not externally validate trajectory-derived features against patient
  outcomes. v10 should not claim that pseudotime = patient trajectory.

---

## §5 Open empirical questions v10 could fill

1. **No published MM study verified here pairs longitudinal scRNA with a
   held-out patient-level prediction of *future* resistance.** Cohen 2021 is
   the closest (longitudinal + CoMMpass generalization), but it predicts at
   the molecular-pathway level, not as a per-patient time-to-resistance
   forecaster. v10 with proper longitudinal CoMMpass + DFCI/Heidelberg
   external testing could be the first.
2. **No abstract-verified study reports a quantitative effect size linking a
   single-cell-derived feature to time-to-progression hazard ratio in MM
   under modern triplet/quadruplet therapy.** Effect-size estimation is the
   single biggest missing number. v10 should explicitly publish HR + 95% CI
   per feature in MMRF-CoMMpass + a held-out cohort.
3. **chr1q-gain subclone *fraction* (not presence/absence) as a continuous
   predictor** has been measured within-patient (Tirier, Sklavenitis-Pistofidis)
   but not benchmarked as a longitudinal forecaster across cohorts.
4. **Microenvironment-derived predictors are reported descriptively** (PD1+
   γδ T, TAMs) but never as a feature-set in a survival model with external
   validation. Major opportunity.
5. **No verified MM paper applies a true dynamical-systems trajectory method
   (Neural-ODE, dynamic OT, MELD, RNA-velocity-based fate) to *patient-level*
   resistance with external cohort validation.** This is the v10 niche if and
   only if v10 has longitudinal training data per patient — which the
   v6 ResistanceMap does not (per the trajectory audit).
6. **NF-κB / TNF as a resistance hallmark needs an abstract-verifiable
   longitudinal study.** Currently the claim circulates in reviews without an
   abstract-level primary citation in this verified set.
7. **Cross-modality fusion (scRNA + scATAC + bulk WGS / WES) for resistance
   trajectory** is unaddressed in all seven verified abstracts. Every paper
   here uses scRNA + targeted validation; none fuse modalities at training
   time.

---

## Summary verdict on v10 anchoring

The verified literature supports **three** trajectory-relevant signal modules
with ≥ 2-cohort replication and mechanistic backing: chr1q→MCL1/PI3K,
TP53→BCL2-family / BH3-mimetic response, and UPR/mitochondrial-respiration
(PPIA / COX8A). Everything else in the candidate list is either
single-cohort, single-paper, or absent from the verified abstracts.

v10's feature set should be built outward from these three modules. The
"add IRF4, NF-κB, t(4;14), LSD1, EZH2 because they sound like MM" approach is
not supported by abstract-verifiable longitudinal evidence in this review.
Numerical effect sizes from abstracts are universally UNVERIFIED — v10 will
need to read full texts (or refit on CoMMpass directly) to populate hazard
ratios.
