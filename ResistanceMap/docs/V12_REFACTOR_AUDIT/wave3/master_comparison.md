# Wave 3 — Master Comparison Table (W3.4)

**Author:** metrics-auditor (synthesis re-persisted by main thread)
**Date:** 2026-05-07
**Source artifacts (read-only):**
- `paper/v8_artifacts/v12_refactor_audit/integration_benchmark.json` — run `r-2026-05-07-v12-refactor-data-integration-audit`
- `paper/v8_artifacts/v12_refactor_audit/metrics_recomputed.json`
- `paper/v8_artifacts/v11_sprint5/cox_per_patient_log_hazards.npz` (sha-256 prefix `3317ab00f1e3fbc2`, n=787, 224 events)
- `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_per_patient_log_hazards.npz`

**Task derivations (LOO Cox PH, penalizer=0.1; B=10000 stratified-by-cyto bootstrap; DeLong via Pencina-D'Agostino paired-bootstrap SE B=200; Bonferroni family m=8 → α_bonf=0.05/8=0.00625).**

---

## §1 Method × metric matrix

| Method | C-index | S1 del17p | S2 t(4;14) | S3 +1q21 | S4 t(11;14) | S5 other | AUROC@12 | AUROC@24 | Δ̄ vs richer | 95% CI | p_boot | DeLong z | p_DL | Bonf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Cox_v11_richer (baseline) | 0.6538 | 0.698 | 0.586 | 0.688 | 0.521 | 0.621 | 0.7123 | 0.6650 | ref | ref | ref | ref | ref | ref |
| Cox_v11_routed_mmSYGNAL (v11.5) | 0.6955 | 0.730 | 0.667 | 0.694 | 0.643 | 0.659 | 0.7505 | 0.7110 | ref | ref | ref | ref | ref | ref |
| mmSYGNAL routed | 0.6938 | 0.705 | 0.722 | 0.664 | 0.693 | 0.682 | 0.7153 | 0.7099 | ref | ref | ref | ref | ref | ref |
| **PCA(K=29) + v11** | 0.6754 | 0.720 | 0.629 | 0.702 | 0.546 | 0.647 | n/a | n/a | +0.0217 | [+0.0005, +0.0433] | 0.0450 | 2.040 | 0.0413 | **No** |
| **ICA(K=29) + v11** | 0.6754 | 0.721 | 0.628 | 0.701 | 0.545 | 0.647 | n/a | n/a | +0.0217 | [+0.0003, +0.0434] | 0.0474 | 2.035 | 0.0419 | **No** |
| **MOFA+(K=29) + v11** | 0.6751 | 0.712 | 0.626 | 0.681 | 0.578 | 0.656 | n/a | n/a | +0.0214 | [+0.0005, +0.0425] | 0.0456 | 2.036 | 0.0417 | **No** |
| **JIVE-joint(K=29) + v11** | 0.6751 | 0.724 | 0.619 | 0.707 | 0.564 | 0.641 | n/a | n/a | +0.0214 | [+0.0017, +0.0418] | 0.0344 | 2.173 | 0.0298 | **No** |
| BlockCCA + v11 | 0.6737 | 0.721 | 0.605 | 0.714 | 0.568 | 0.640 | n/a | n/a | +0.0200 | [+0.0017, +0.0388] | 0.0348 | 2.154 | 0.0313 | **No** |
| KernelPCA(rbf) + v11 | 0.6738 | 0.721 | 0.623 | 0.696 | 0.538 | 0.651 | n/a | n/a | +0.0201 | [−0.0022, +0.0427] | 0.0766 | 1.824 | 0.0681 | **No** |
| NMF(K=29) + v11 | 0.6711 | 0.709 | 0.624 | 0.699 | 0.548 | 0.651 | n/a | n/a | +0.0174 | [−0.0039, +0.0391] | 0.1122 | 1.628 | 0.1034 | **No** |
| SparsePCA + v11 | 0.6726 | 0.719 | 0.607 | 0.691 | 0.546 | 0.649 | n/a | n/a | +0.0189 | [−0.0018, +0.0399] | 0.0742 | 1.871 | 0.0614 | **No** |

**n/a notes:** AUROC@12mo and @24mo were computed only for the three reference rows in `metrics_recomputed.json`; not produced for the 8-method grid. Brier@12mo and per-stratum 90% jackknife+ coverage exist only for the v10 stack (`r-2026-05-03-v11s5-conformal`), not for these 8 factor constructions. No values can be honestly reported here without fabrication.

## §2 Per-stratum lift heatmap (data)

Δ_C per stratum vs Cox_v11_richer:

| Method | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|
| PCA + v11 | +0.022 | +0.043 | +0.014 | +0.025 | +0.026 |
| ICA + v11 | +0.023 | +0.042 | +0.013 | +0.024 | +0.026 |
| MOFA+ + v11 | +0.014 | +0.040 | −0.007 | +0.057 | +0.035 |
| JIVE + v11 | +0.026 | +0.033 | +0.019 | +0.043 | +0.020 |
| BlockCCA + v11 | +0.023 | +0.019 | +0.026 | +0.047 | +0.019 |
| KernelPCA + v11 | +0.023 | +0.037 | +0.008 | +0.017 | +0.030 |
| NMF + v11 | +0.011 | +0.038 | +0.011 | +0.027 | +0.030 |
| SparsePCA + v11 | +0.021 | +0.021 | +0.003 | +0.025 | +0.028 |

S4 t(11;14) is persistently the weakest absolute stratum across all 8 (raw C 0.538–0.578, never exceeding 0.58). MOFA+'s apparent S4 lead (+0.057) is the largest in that column but co-located with the v12 BCL2 specialist STRICT FAIL (Δ_S4=+0.0135, p=0.446) — chair verdict §3(iii) framed this as a safety flag, not a positive result.

## §3 Multiple-comparison verdict

**Bonferroni family-wise threshold:** α_bonf = 0.05 / 8 = **0.00625**.

DeLong p across the 8 methods: **0.0298 to 0.1034**. Bootstrap two-sided p: **0.0344 to 0.1122**.

**Zero of 8 methods survive Bonferroni at α=0.00625.** The most significant — JIVE-joint + v11 — has DeLong p=0.0298, falling 4.8× short of the family-wise threshold. Five methods (PCA, ICA, MOFA+, JIVE, BlockCCA) clear the unadjusted α=0.05 bootstrap threshold but fail Bonferroni. Three (KernelPCA, NMF, SparsePCA) fail even unadjusted.

## §4 Headline insight

**The +0.021 marginal C-index "lift" is method-agnostic.** Effect sizes cluster within 0.0043 across all 8 methods (NMF 0.6711 → PCA/ICA 0.6754); CIs overlap nearly completely. Under family-wise α=0.05 with m=8, the entire cluster is consistent with the null hypothesis that any 29-dim unsupervised factorization of the MMRF expression matrix produces equivalent rank-order signal when concatenated to v11_richer.

Stated positively: there is a real, reproducible, modest improvement over Cox_v11_richer when the feature set is expanded from 23 to 52 features by ANY reasonable unsupervised decomposition. Stated honestly: this is a feature-expansion effect, not an architectural one.

## §5 Honesty check — is MOFA+ architecturally special?

**No.** Five lines of evidence:

1. **C-index spread across 8 methods is 0.0043** (NMF 0.6711 ↔ PCA/ICA 0.6754). MOFA+ (0.6751) is the third-best of eight by C-index.
2. **Bootstrap Δ:** MOFA+ +0.0214 sits between PCA/ICA (+0.0217) and JIVE/BlockCCA (+0.0214/+0.0200). Differences are in the fourth decimal — bootstrap noise.
3. **MOFA+'s apparent S4 advantage** (Δ_S4=+0.057) coincides with the failed v12 BCL2 specialist (RUNS.md row 36) — this is a regularization artifact that triggers the safety flag, not a MOFA-specific biological signal.
4. **JIVE-joint achieves an identical marginal C** to MOFA+ (both 0.6751) at zero seconds extraction time vs MOFA+'s 77 s.
5. **No method survives Bonferroni** — none of the eight, including MOFA+, can claim a multiple-comparison-robust improvement.

The honest framing for the v12 manuscript is the chair verdict §3(i) language verbatim: *"29 unsupervised expression factors of any reasonable kind close the v11_richer gap by Δ=+0.021, p=0.045 (single comparison only); none survive Bonferroni at α=0.05/8."*

Recommended live-pipeline implementation (still): replace mofapy2 with `sklearn.PCA(K=29)` per `data_integration_validation.md` §5 step 4 — same lift, faster, simpler dependency surface.

## §6 What's next to actually move the needle

The +0.021 ceiling under any of the 8 factor decompositions suggests the bottleneck is not the integrator but the underlying baseline expression matrix. Real options to break the ceiling, in cost order:

1. **External cohort F_S5 replication** — the only direction the chair verdict §6 gap-1 names that is a cohort-bound limit, not a method-bound limit.
2. **Paired modalities (CNV + proteomics + RNA)** — currently single-view; multi-view methods cannot prove their worth on a single-view input.
3. **Touzeau t(11;14) feature** (W3.3, in flight) — directly attacks the S4 weakness that all 8 factor methods fail to fix.
4. **NUTS HMC re-inference** (Sprint B) — closes the v12 PPC 5/10 fail caveat; doesn't change C-index but lifts the bioRxiv → Cell Reports Methods conditional.
