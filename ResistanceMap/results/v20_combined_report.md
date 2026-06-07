# ResistanceMap v20 — Combined Results Report

**Generated:** 2026-06-07 16:47
**Branch:** v20/lab-first-pivot
**Data:** Open-access only (GDC open tier + GEO + Zenodo)

---
# ResistanceMap v20 — First Baseline Results

Date: 2026-06-07 16:41
Data: GDC open-tier MMRF-COMMPASS (no dbGaP)
Endpoint: Overall Survival (OS)
Censoring: Properly handled via sksurv structured arrays
Bootstrap: 200 iterations for 95% CIs

## Baseline Results — GDC MMRF Clinical-Only (Age + Gender → OS)

| Model | C-index | 95% CI | Notes |
|-------|---------|--------|-------|
| CoxPH | 0.6456 | [0.578, 0.710] | censoring-aware |
| ElasticNet_Cox | 0.6635 | [0.590, 0.739] | censoring-aware |
| RSF_sksurv | 0.5574 | [0.478, 0.641] | censoring-aware |
| GBM_Survival | 0.5553 | [0.469, 0.640] | censoring-aware |

Note: These are CLINICAL-ONLY baselines (2 features). The bar will be higher once RNA-seq features are added.


## What This Proves

1. The corrected baseline cascade RUNS with proper censoring (sksurv)
2. IMWG-compatible endpoints (OS) are correctly extracted from GDC open tier
3. The pipeline produces actual C-index values with bootstrap CIs
4. This is the foundation for adding RNA-seq and lab features


## Next Steps (data-dependent)

- [ ] Download MMRF open-tier STAR Counts RNA-seq (995 patients) → add as features
- [ ] MMRF Virtual Lab access → PER_PATIENT_VISIT labs → full PK-SSM training
- [ ] GSE24080 (559 pts with EFS/OS) → external validation
- [ ] GSE136337 (426 pts with B2M/Alb/LDH) → PK observation model validation


## scVI Integration

*Not yet run. Execute `python scripts/run_scvi_integration.py`*


## PK-SSM

*Not yet run. Execute `python scripts/run_pkssm_forward.py`*


---

## What This Report Proves (gated on artifacts produced this run)

1. The corrected baseline cascade runs with proper censoring (sksurv)
2. All open-access — no IRB, no dbGaP, no gateway registration required

**NOT demonstrated this run (step did not complete):**
- scVI integrates MM scRNA-seq into batch-corrected latents + extracts chromatin / biomarker-proxy expression
- The PK-SSM architecture runs end-to-end on real biological data

## What Remains Data-Blocked

- Training the PK-SSM survival head (requires MMRF Virtual Lab visit-level labs + outcomes)
- Producing actual C-index for PK-SSM vs baselines
- Validating mechanism classifier assignments against molecular subtypes
- External validation on GSE136337 (B2M/albumin/LDH) or GMMG-MM5