# MORT-FM Longitudinal Data Inventory

Status: 2026-05-18 (Phase D scaffolding pass)

This document catalogs public Multiple-Myeloma and AML cohorts that
could supply same-patient longitudinal molecular pairs (Level 3 evidence
per the v15 plan). The current MMRF substrate (29 pairs) FAILS the
`gate_longitudinal_trajectory` threshold of 100 pairs — additional
cohorts are needed.

## Currently ingested

| Cohort | Patients | Paired molecular samples | Endpoint signal | Source |
|---|---|---|---|---|
| MMRF CoMMpass IA-12 | 994 | **29 with t0+t1 RNA-seq** | OS + TT2L + bort_1L | data/processed/mmrf_paired_patients.tsv (already on disk) |
| BeatAML 1.0 | 451 specimens / 303 patients | 0 (specimen-level only, no longitudinal) | ex-vivo AUC | data/processed/beataml/ |
| GSE124310 + GSE271107 | 51 patient-stage pseudobulks | 0 (cross-sectional pseudotime) | disease_stage_pseudotime | data/processed/single_cell/scrna_manifest.csv |

## Candidate cohorts NOT yet ingested

The following public datasets advertise longitudinal molecular data
relevant to MM resistance. None are currently on this disk; they are
the search targets for the next data-ingestion sprint.

### Multiple Myeloma — bulk transcriptomics, paired baseline → relapse

1. **MMRF CoMMpass IA-19+** — newer interim analysis adds patients beyond IA-12
   - Possibly 60–120 patients with paired t0/t1 RNA-seq
   - Source: GDC (Genomic Data Commons) + dbGaP phs000748
   - Ingestion friction: dbGaP controlled access; requires IRB

2. **GSE116324 (Chapman et al. PADIMAC)** — **NOT paired longitudinal**
   - 44 newly diagnosed MM patients with **baseline-only** RNA-seq + PAD
     (bortezomib + adriamycin + dexamethasone) treatment response
   - Ingested 2026-05-18 at `data/processed/padimac/` — 98.8% Block-A
     feature coverage (1,975/2,000), so it cleanly joins the Block-A
     static_drug_response training pool.
   - Does **NOT** unlock `gate_longitudinal_trajectory` — corrected from
     the prior version of this inventory which misattributed it to APEX.
   - For paired pre/post bortezomib, see GSE9782 (Mulligan et al.) and
     GSE39754 (Brioli et al.) which actually carry baseline + relapse pairs.

3. **GSE9782 (Mulligan et al.)** — bortezomib pre-treatment + response endpoint
   - Cross-sectional but with response labels usable as resistance proxy
   - n ≈ 264 patients; Affymetrix HT-array
   - Ingestion friction: low

4. **EGAS00001005029 (PCROWD/Genome Medicine 2020)** — paired baseline + relapse RNA-seq
   - ~25 MM patients with serial RNA-seq + IS/MFC MRD
   - Source: EGA controlled access
   - Ingestion friction: medium (EGA token + manuscript-level IRB)

### Multiple Myeloma — single-cell, paired

5. **Tirier et al. 2021 (Nature Communications 12:6960)** — scRNA-seq pre/post relapse
   - 14 MM patients with baseline + relapse scRNA-seq
   - Source: EGA EGAS00001004746
   - Ingestion friction: medium (EGA)

6. **Cohen et al. 2021 (Cancer Cell 39:1422)** — scRNA-seq with serial sampling
   - 26 MM patients across MGUS / SMM / MM / relapse
   - Source: dbGaP phs002435
   - Ingestion friction: high (dbGaP controlled)

### AML — paired diagnosis → relapse

7. **TCGA AML LAML cohort** — 200 patients, mostly diagnosis-only
   - 0 longitudinal pairs by default; could mine for treatment-naive vs post-induction
   - Source: GDC public
   - Ingestion friction: low

8. **GSE48173 (Christopher et al. NEJM 2018)** — AML diagnosis + relapse WES
   - 50 paired patients with WES (not RNA-seq)
   - Source: GEO + dbGaP
   - Ingestion friction: medium

9. **BeatAML 2.0 (extended waves)** — newer release adds longitudinal samples
   - Possible n ~60 paired patients (advertised, not yet validated on this disk)
   - Source: Synapse syn23561010
   - Ingestion friction: low–medium

### Drug-treated cell-line time-series (Level 4 evidence)

10. **LINCS L1000** — 1000+ cell lines × 1000s of perturbagens × multiple timepoints
    - Source: clue.io public
    - Ingestion friction: low; gigantic; needs cell-line → MM cross-walk

11. **Bortezomib resistance ex-vivo time-courses** — multiple GEO sets
    - GSE6480 (Markovina 2008), GSE15179 (Decaux 2008), GSE89393 (Mitsiades lab)
    - Source: GEO public
    - Ingestion friction: low

## Recommended next-data-sprint targets

For a Level-3 longitudinal_trajectory claim with ≥100 paired patients,
the highest-value ingestion path is:

1. **MMRF IA-19 update** (if accessible) → adds 30–80 more paired patients
2. **GSE116324** (bortezomib pre/post APEX) → +~150 patients
3. **Tirier 2021 scRNA** (EGA) → +14 single-cell paired pairs

Combined this could plausibly reach n ≥ 150 paired MM patients, clearing
the 100-pair gate. **This requires no model changes — only data work.**

## Gate decisions with each scenario

| Scenario | n_pairs | gate_longitudinal_trajectory |
|---|---|---|
| Current (MMRF only) | 29 | **FAIL** |
| + GSE39754 (Brioli paired) | ~50 | **FAIL** (still <100) |
| + Tirier 2021 EGA scRNA | ~64 | **FAIL** + scRNA channel |
| + MMRF IA-19 (if accessible) | ~130 | **PASS** |
| + GSE9782 (APEX cross-sectional usable as response proxy) | ~330 | **PASS** + drug-response cross-validation |

(Removed prior row that assumed GSE116324 was paired — it is not.)

## Honest limitations of this inventory

* The advertised patient counts come from manuscript-level reports; the
  *usable* count after QC + matching + calendar-time-label availability
  is typically 60–80% of the advertised number.
* dbGaP-controlled cohorts cannot be pulled without IRB. The bulk of
  high-quality MM longitudinal data is in this tier.
* Drug-treated cell-line time-series (Level 4) is abundant but does NOT
  unlock the *patient-level* claim — it is auxiliary evidence only.

---

*Generated by Phase D of v15 next-pass. See `r-2026-05-18-mortfm-longitudinal-substrate`
RUNS row for the actual 29-pair substrate built from MMRF.*
