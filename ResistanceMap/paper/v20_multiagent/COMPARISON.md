# ResistanceMap v20 vs Nearest SOTA — Honest Head-to-Head

**Scope:** This document compares the *proposed* ResistanceMap v20 "lab-first" multiple-myeloma
progression/relapse forecaster against the nearest published methods. It is deliberately conservative.

> **Empirical status, stated plainly:** The proposed v20 model has **NO trained end-to-end result yet.**
> There is no ResistanceMap v20 C-index, AUROC, or hitting-time calibration number to report. The only
> ResistanceMap numbers that exist and may be cited are the *baseline* survival results below. We do **not**
> fabricate a ResistanceMap model C-index, and we do **not** claim a win. The comparison is therefore on
> **conceptual axes**, with the published comparators' numbers attributed to their sources.

---

## 1. The thesis under test

ResistanceMap v20 is **not a detector**. The claim is that longitudinal **routine labs are the primary,
interpretable signal**, and omics are an *optional* layer. The architectural commitments:

1. Labs are a **half-life-filtered, assay-noisy projection** of a latent resistance state.
2. A **PK-constrained observation decoder** *inverts* the measurement (de-convolves assay noise + drug
   pharmacokinetics) before any hazard is computed.
3. A **Waddington-SDE** casts relapse as **basin-escape**, with **hitting-time as the hazard**.
4. A **mechanism classifier** decomposes risk over the trajectory.
5. A **falsification / claim-gate harness** refuses unearned claims.

**Expectation: parity with strong baselines, not a win.** The intended contribution is
**measurement-inversion + legibility + falsification discipline**, not a leaderboard delta.

---

## 2. Real ResistanceMap metrics (the only numbers we own)

These are **baseline** survival results from the GDC open-tier MMRF-CoMMpass loader. They are **not** the
proposed v20 model — they are the floor it must clear, and they are sobering.

| Model (baseline) | Endpoint | N / events | Features | C-index [95% CI] |
|---|---|---|---|---|
| ElasticNet-Cox | OS, censoring-aware | 994 / 191 | age + gender (2) | **0.664 [0.590, 0.739]** |
| CoxPH | OS, censoring-aware | 994 / 191 | age + gender (2) | 0.646 [0.578, 0.710] |
| Random Survival Forest | OS, censoring-aware | 994 / 191 | age + gender (2) | 0.557 |
| Gradient-Boosted Survival | OS, censoring-aware | 994 / 191 | age + gender (2) | 0.555 |

**Reading:** With only age + gender, ElasticNet-Cox reaches 0.664 — **age dominates**. This is the honest
clinical-only floor. The expanded longitudinal lab panel (`clinical_features.py` /
`encode_longitudinal_labs()`) is **not yet wired into the default model path** (canonical model still defaults
to the original 5 clinical features), so the lab-first thesis is **not yet empirically tested**.

Additional substrate on disk: **BeatAML AWS FPKM-UQ** = 510 specimens × 60,483 genes (no ex-vivo
drug-response AUC in the bucket — an AML cohort, used for transfer/pretraining only, not an MM endpoint).

---

## 3. Nearest comparators (attributed published numbers)

| Method | Substrate | Endpoint / transition | N | Headline metric | Verified |
|---|---|---|---|---|---|
| **PANGEA-SMM** | Routine labs (M-protein, sFLC ratio, creatinine, Hb), evolving | SMM → MM precursor | 2,344 (7 centers) | **C ≈ 0.79** (0.78 w/o biomarker history or recent marrow) | ✅ PubMed |
| **mmSYGNAL** | Transcriptional program activity (SYGNAL GRN), subtype-routed | PFS across full trajectory | 881 train / 1,367 val | Beats ISS / cytogenetics / gene panels; *interpretable per-subtype* | ✅ PubMed |
| **Ferle 2025** | Longitudinal clinical sequences (LSTM + conditional-RBM) | Progression @3mo | ~875 | **AUROC ≈ 0.78 @3mo** | 🟡 **NOT verified via MCP** |
| **SCOPE / Sontag** | Transformer over longitudinal EHR/clinical trajectories | MM progression | — | AUROC-optimized discriminative | 🟡 **NOT verified via MCP** |
| **Maura 2025 (genomic-MM)** | Genomics + 2/20/20 (static) | Malignant transformation (MGUS/SMM) | 374 | Genomics improves prediction | ✅ PubMed (positioning foil) |

**Attribution / citations:**
- PANGEA-SMM — Chabrun F, et al. *Enhanced dynamic risk stratification of smoldering multiple myeloma.*
  Nat Med (2026). PMID 41876650. DOI 10.1038/s41591-026-04304-x.
- mmSYGNAL — Murie C, Turkarslan S, … Baliga NS. *Individualized dynamic risk assessment and treatment
  selection for multiple myeloma.* Br J Cancer 132(10):922–936 (2025). PMID 40169765.
  DOI 10.1038/s41416-025-02987-6.
- Maura F, et al. *Genomics Define Malignant Transformation in Myeloma Precursor Conditions.*
  J Clin Oncol 44(3):188–199 (2025). PMID 41061199. DOI 10.1200/JCO-25-01733.
- Hevroni G, et al. *From MGUS to multiple myeloma: Unraveling the unknown of precursor states.*
  Blood Rev 68:101242 (2024). PMID 39389906. DOI 10.1016/j.blre.2024.101242. (Motivation citation.)

> **Honesty flag (no fabrication):** **Ferle 2025** and **SCOPE/Sontag** did **NOT resolve via the
> PubMed or bioRxiv/medRxiv MCP** under author-, title-, and journal-targeted queries. Their numbers
> (AUROC ≈ 0.78 @3mo; transformer/discriminative) are carried from the project brief and are marked
> **unverified**. Do **not** cite them with invented PMIDs/DOIs — pull real identifiers from npj Digital
> Medicine before they enter the manuscript.

---

## 4. Conceptual head-to-head (the real comparison)

Because v20 has no trained result, the defensible comparison is axis-by-axis, not metric-by-metric.

| Axis | ResistanceMap v20 (proposed) | PANGEA-SMM | mmSYGNAL | Ferle 2025 🟡 | SCOPE 🟡 |
|---|---|---|---|---|---|
| **Substrate** | Routine labs primary; omics optional | Routine labs | Transcriptional programs (omics-required) | Clinical sequences | EHR trajectories |
| **Temporality** | Longitudinal latent-state dynamics (SDE rollout) | Longitudinal, evolving biomarkers | Per-timepoint static program activity | Longitudinal (LSTM) | Longitudinal (transformer) |
| **Labs treatment** | **Measurement-inversion**: labs = PK/half-life-filtered noisy emission; decoder de-convolves before hazard | **Face-value** trajectory covariates in hazard regression | n/a (omics, not labs) | Raw sequence into LSTM | Raw sequence into attention |
| **Survival rigor** | Hitting-time hazard from basin-escape; censoring-aware (IMWG-PFS, observed-event gate) | C-statistic, landmark/Cox-style, large multi-center | PFS prediction, multi-cohort validation | AUROC @3mo (discriminative, less survival-native) | AUROC (discriminative) |
| **Interpretability / mechanism** | Mechanism classifier over the *trajectory of basin escape*; legible in lab+latent space | Interpretable biomarker covariates | **Strong**: per-subtype transcriptional-program decomposition | Black-box CRBM latent | Opaque attention map |
| **Causal discipline** | Causal generative chain (latent → PK emission → observed lab) + claim-gate refusal | Statistical hazard model | GRN-grounded (mechanistic but static) | None explicit | None explicit |
| **Empirical status** | **No trained result yet** (baselines only: EN-Cox 0.664) | **C ≈ 0.79, N=2,344** | Validated N=1,367 | AUROC ≈ 0.78 🟡 | — 🟡 |

### Sharpest single contrast per comparator
- **vs PANGEA-SMM** (best modality twin): same "labs are primary" philosophy, but PANGEA reads labs at
  **face value** as hazard covariates; v20 **inverts the measurement** (PK + assay-noise de-convolution)
  first. Different transition too — PANGEA targets **SMM→MM precursor**; v20 targets **active-disease
  PFS/relapse** on GDC MMRF. This is the cleanest legibility/parity benchmark.
- **vs mmSYGNAL** (mechanism twin): mmSYGNAL's decomposition lives in **omics space and is static per
  timepoint**; v20's lives in **lab + latent-dynamics space and is longitudinal**. v20's thesis is that
  labs alone carry the primary interpretable signal — the "interpretability without omics" contrast.
  Memory note: v11.5 already **ties** mmSYGNAL on PFS C-index, which reinforces *parity, not win*.
- **vs Ferle 2025** 🟡 (architectural twin): both are sequence-to-hazard with a generative latent, but
  Ferle's CRBM is a **black-box** latent; v20's latent is **physically constrained** (PK operator +
  Waddington potential whose basin-escape *is* the hazard).
- **vs SCOPE/Sontag** 🟡 (transformer counterpoint): transformer learns opaque attention over visits;
  v20 imposes a **causal generative chain** + explicit hitting-time hazard + falsification harness, trading
  raw discrimination for interpretability and honest claim-gating.

---

## 5. Where the niche is genuinely unoccupied

No indexed MM method found combines **(a)** a PK-constrained observation decoder, **(b)** a
basin-escape / hitting-time SDE hazard, and **(c)** a falsification / claim-gate harness on routine labs.
No direct "M-protein/sFLC PK-kinetics → relapse hazard" forecasting model surfaced for active MM (the
kinetics literature is descriptive, not a hazard model). The **PK-inversion + Waddington-hitting-time +
honesty-harness** combination appears to be the defensible novelty axis — *even under a parity-not-win
expectation.*

---

## 6. Honest verdict

- **Expected outcome: parity** with strong baselines (PANGEA ~0.79; mmSYGNAL ties at v11.5; Ferle ~0.78 🟡),
  **not** a discrimination win. Age already buys EN-Cox 0.664 on 2 features; the lab-first decoder must
  justify itself on **legibility and honesty**, not leaderboard delta.
- **Contribution, restated:** (1) **measurement-inversion** — labs treated as a PK/assay-noise emission of a
  latent state rather than face-value covariates; (2) **legibility** — mechanism decomposition over a basin-
  escape trajectory; (3) **falsification discipline** — a claim-gate harness that refuses unearned claims.
- **What must happen before any metric claim:** the expanded longitudinal lab panel must be **wired into the
  default model path** (currently additive-only, not in the canonical 5-feature path); the DAG dependency
  inversion (ISSUE 3) and the harmonize `tier` KeyError (ISSUE 6) must be fixed; and the hard blocks
  (scvi-tools + scikit-misc absent; no MMRF outcome data on disk; Spark confirmed-dataset + split manifest)
  must be cleared so `10_run_baselines.py` can produce a real v20 result on the MMRF PFS endpoint.

---

*No ResistanceMap C-index is asserted for the proposed model anywhere in this document. All comparator
numbers are attributed; PANGEA-SMM, mmSYGNAL, and Maura 2025 are PubMed-verified; Ferle 2025 and
SCOPE/Sontag are explicitly marked unverified-via-MCP pending manual DOI retrieval.*
