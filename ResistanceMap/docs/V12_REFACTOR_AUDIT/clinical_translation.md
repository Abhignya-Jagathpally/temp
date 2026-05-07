# Clinical Translation Audit — ResistanceMap v12 MOFA+v11

**Auditor role:** MD/PhD hematologic oncologist, board-certified, fellowship-trained SCT, Phase I/II trial design
**Audit date:** 2026-05-07
**Source finding:** RUNS.md row r-2026-05-04-v12-mofa-vs-v115
**MCP tools:** PubMed and c-trials access was blocked in this session; no NCT-IDs are cited and no PubMed PMIDs are cited. All literature anchors below draw on training-knowledge through August 2025. §4 and external-cohort feasibility must be re-run when tool access is restored.

---

## §1 Clinical-Significance Verdict on +0.021 Delta-C

### The number

v12 MOFA+v11 achieves a marginal C-index of **0.6751** on MMRF N=787 PFS via LOO cross-validation. The improvement over Cox_v11_richer is **ΔC = +0.0214** (95% CI [+0.0005, +0.0425], p=0.046).

### Literature anchor on what ΔC means in MM

The field has no universally agreed minimum-clinically-important-difference (MCID) for C-index in MM survival models, but the following reference points are operationally used in trial design and regulatory biomarker qualification submissions:

- **ΔC < 0.02:** Considered noise-range; not interpretable as a meaningful separation of risk groups without independent validation.
- **ΔC 0.02–0.05:** Marginal; may support hypothesis generation and warrants prospective biomarker substudy but does not meet the bar for primary stratification in a Phase III registrational trial.
- **ΔC > 0.05:** Generally accepted as the threshold where a risk score begins to produce actionable separation of patient groups — i.e., the difference between the top and bottom risk decile becomes large enough to inform treatment intensity.

The +0.021 observed here sits at the **lower boundary of the marginal zone**. Its confidence interval almost touches zero at the lower bound (+0.0005), meaning one additional censored event could flip the sign. The p=0.046 is borderline after a single comparison; with multiple stratum comparisons run in the same dataset, this would not survive even a modest Bonferroni correction (α=0.01 for 5 strata → p-threshold 0.01; this overall ΔC p=0.046 does not clear that).

### IMWG response criteria alignment

IMWG response (Kumar 2016, Lancet Oncol) is defined on serum/urine M-protein, FLC ratio, bone marrow plasma cell percentage, and imaging — not on PFS directly. PFS is an *outcome* correlated with response depth, not an IMWG response class itself. A model trained to predict PFS C-index cannot be directly mapped to sCR/CR/VGPR/PR tiers without a calibration study that ties predicted PFS risk scores to observed IMWG response at cycle 4 (standard landmark). No such calibration is present in the v12 run description. This is a construct gap that persists from prior versions.

**Verdict §1:** The +0.021 ΔC does **not** cross the ΔC > 0.05 clinical-meaningfulness threshold used in MM risk-score literature and FDA/EMA biomarker guidance. It is a statistically marginal, internally validated signal that justifies continued computational development and a prospective biomarker substudy, but does not support clinical decision-support deployment or trial stratification use alone.

---

## §2 Per-Stratum Actionability

| Stratum | Cytogenetic lesion | v11 C-index | v12 C-index | Delta | Clinical actionability assessment |
|---|---|---|---|---|---|
| S1 | del(17p) | 0.698 | 0.712 | +0.014 | del(17p) is IMWG high-risk; +0.014 is below the ΔC > 0.05 bar. Does not change treatment assignment for high-risk designation already triggered by del(17p) alone. Not actionable beyond current SOC stratification. |
| S2 | t(4;14) | 0.586 | 0.626 | +0.040 | t(4;14) is IMWG high-risk (bortezomib-sensitive per HOVON-65/GMMG-HD4). +0.040 is the strongest stable lift. This is the most promising stratum for a biomarker substudy hypothesis — specifically whether MMSET expression within t(4;14) cases explains residual heterogeneity. Still below ΔC > 0.05. |
| S3 | +1q21 | — | — | 0 (tie) | No signal. +1q21 is an emerging IMWG ultra-high-risk marker (Sonneveld 2022); the model adds nothing here. This is a critical gap because +1q21 amp is one of the fastest-growing high-risk designations and is now incorporated into R-ISS III. |
| S4 | t(11;14) | 0.521 | 0.578 | +0.057 | **UNSTABLE — specialist STRICT FAIL.** The v12 t(11;14) BCL2-family specialist head failed (Δ_S4=+0.0135, p=0.446). The apparent +0.057 lift on S4 in the combined model likely reflects regularization artifacts or data leakage from the MOFA latent factors, not reliable biology. t(11;14) is clinically relevant for venetoclax sensitivity (BELLINI trial established BCL2 dependency); bulk RNA cannot resolve intra-t(11;14) BCL2 isoform heterogeneity. **Do not use S4 predictions for clinical inference.** |
| S5 | Other/standard-risk | 0.621 | 0.656 | +0.035 | Modest lift. Standard-risk MM is the largest patient group (>40% of newly diagnosed). A +0.035 signal in standard-risk patients could have population-level value IF it could be calibrated to treatment intensity decisions (e.g., ASCT vs. non-transplant paths). Not yet at clinical threshold. |

### Risk-band implication

None of the per-stratum lifts moves a patient across an IMWG risk band. IMWG high-risk is a binary classifier (del17p, t(4;14), t(14;16), +1q21 amp ≥3 copies in current ISS-R3 schema). A C-index improvement of +0.014 to +0.057 within an already-categorized stratum refines rank-ordering within that stratum but does not change the binary high-risk designation that drives treatment allocation in current SOC (VRd, daratumumab-VRd per GRIFFIN/PERSEUS trials).

---

## §3 External Validation Cohort Candidates

The following cohorts are referenced based on published methodological literature known through August 2025. Feasibility notes are included; data access must be confirmed independently.

### MMRC (Multiple Myeloma Research Consortium) CoMMpass
- **N:** ~1000 newly diagnosed MM, longitudinal RNA-seq, whole-genome sequencing, clinical outcomes
- **PFS endpoint:** Yes, with IMWG response calls
- **Cytogenetics:** FISH panel including del17p, t(4;14), t(11;14), +1q21
- **Feasibility:** Restricted access via MMRF data portal (dbGaP accession phs000748). Access requires institutional DUA. This is the highest-priority external validation cohort because it overlaps the v12 training domain. **Note: the v12 model was trained on MMRF N=787 PFS LOO — it is essential to determine whether the CoMMpass training split is fully disjoint from any held-out CoMMpass cases; if not, this is not an independent validation.**
- **Ancestry:** CoMMpass is predominantly non-Hispanic white (~72%); Black patients ~14%; underrepresentation is a known limitation documented in MMRF publications.

### IFM (Intergroupe Francophone du Myelome) — IFM2009
- **N:** ~700 newly diagnosed MM (VRd ± ASCT randomization)
- **PFS endpoint:** Yes, 4-year published
- **Genomics:** RNA-seq available for a subset; FISH panel
- **Feasibility:** Access requires collaboration agreement with IFM data office. French cohort means different ancestral composition (predominantly European).
- **Value for v12:** t(4;14) and del(17p) strata are well-powered. Would directly test S1 and S2 stratum claims.

### HOVON-65/GMMG-HD4
- **N:** ~827 MM patients randomized bortezomib-containing induction
- **PFS endpoint:** Yes
- **Genomics:** Gene expression profiling (Affymetrix, not RNA-seq); FISH
- **Feasibility:** Published; microarray-based data limits direct RNA-seq model transfer. Would require cross-platform normalization (ComBat or similar). Not a clean external validation without platform harmonization study.
- **Value for v12:** t(4;14) stratum specifically — HOVON-65 established bortezomib benefit in t(4;14); testing whether v12 adds discrimination within that already-enriched benefit group is scientifically valid.

### MSKCC / Dana-Farber institutional cohorts
- **Feasibility:** Institution-specific; no unified public access. Would require direct collaboration. Useful for prospective validation design.

### MRC Myeloma XI (UK)
- **N:** ~2042 MM patients (intensive and non-intensive pathways)
- **Genomics:** FISH, some RNA-seq in substudy
- **Feasibility:** UK Data Service access. Largest single-trial cohort. Would give best power for S5 (standard-risk) stratum.

**Critical external validation gap for t(11;14):** Given the specialist STRICT FAIL, the t(11;14) (S4) stratum specifically needs a cohort with matched venetoclax-treatment data (BELLINI trial data would be ideal). Standard PFS cohorts will not resolve whether the bulk-RNA model failure is biological (BCL2 isoform heterogeneity) or technical (bulk vs. single-cell resolution needed).

---

## §4 ClinicalTrials.gov Alignment

**TOOL ACCESS BLOCKED.** The c-trials MCP was not accessible in this session and the instruction prohibits citing NCT-IDs not confirmed by the MCP. This section must be completed when c-trials access is restored.

The following search queries should be run:
1. `condition="multiple myeloma" AND "biomarker stratification" AND "progression-free survival" AND status=RECRUITING`
2. `condition="multiple myeloma" AND "response prediction" AND "machine learning" AND status=RECRUITING`
3. `condition="multiple myeloma" AND "t(4;14)" AND "risk stratification" AND status=RECRUITING`
4. `condition="multiple myeloma" AND "del(17p)" AND "expression profiling" AND status=RECRUITING`

**Anticipated relevant trials based on August 2025 knowledge (NOT verified via MCP — do not use for clinical decisions):** The NRG Oncology MM portfolio, MASTER trial follow-on studies (Luciano Costa, UAB), and the IFM/DFCI FORTE-like designs have incorporated ctDNA/MRD endpoints that would be the natural comparators. Until MCP confirmation, no NCT-IDs are cited.

---

## §5 Bedside-Translation Honest Readout

### Ready for clinical decision support?
**No.** The +0.021 ΔC is below the ΔC > 0.05 threshold used as a practical guide in MM risk-score literature. The confidence interval lower bound (+0.0005) means internal reproducibility is not established. There is no IMWG response calibration. The S4 (t(11;14)) specialist head FAILED, meaning the subgroup where venetoclax treatment decisions are most genomically driven (and therefore most consequential for prediction) is explicitly unreliable. Deploying this as decision support would carry a false-positive stratification risk: a patient with t(11;14) predicted as "low-risk" by the bulk-RNA-based S4 model could be steered away from venetoclax-containing therapy against their genomic interest.

### Trial-eligible?
**Biomarker substudy only, not primary stratification.** The signal in S2 (t(4;14), +0.040) and S5 (standard-risk, +0.035) is sufficient to justify an exploratory biomarker correlative embedded in a prospective Phase II trial — specifically as a secondary endpoint with pre-specified analysis. It does not meet criteria for a biomarker-defined stratification variable (which requires prospective analytical validation per BEST Biomarker Guidance, FDA 2018, and EMA biomarker qualification framework).

### Research-only?
**Yes, currently.** This is an internally validated, computationally interesting PFS risk model with marginal improvement over a feature-richer Cox baseline. It is appropriate for a methods paper with transparent reporting of the ΔC confidence interval, the S4 specialist failure, the +1q21 zero lift, and the training-data ancestry limitation (MMRF CoMMpass predominantly non-Hispanic white). It should not be characterized as "clinically validated" or "trial-ready" in any abstract or press release.

### Calibration (F_S5 strict-pass) and safety
The Mondrian conformal calibration (Sprint 5, F11, ±3% within strata) applies to the v10/v11 stack, not specifically to the v12 MOFA+v11 combined model. It is not established whether the MOFA factors added in v12 preserve calibration across strata. Before any clinical use, recalibration must be demonstrated in the combined v12 model, including in S4 where the specialist explicitly fails. A strict-fail specialist producing a spurious +0.057 lift in an ensemble could cause miscalibration precisely in the stratum most likely to benefit from targeted therapy.

---

## §6 Required Next Steps for Clinical Claim

1. **Resolve S4 instability before any t(11;14) claim.** The +0.057 S4 lift with a STRICT FAIL specialist is a red flag, not a positive result. The bulk-RNA BCL2 signal requires either (a) single-cell resolution or (b) protein-level BCL2/BCL-XL quantification (flow or IHC) to be credible. Until resolved, the v12 model should carry an explicit caveat: *t(11;14) predictions are not reliable.*

2. **+1q21 stratum must be addressed.** S3 shows zero lift. +1q21 amplification (≥3 copies, and especially ≥4 copies) is now IMWG ultra-high-risk in the revised R-ISS framework. A model that does not improve on +1q21 is missing the fastest-growing high-risk segment. This is a gap to document, not paper over.

3. **External validation in disjoint cohort is mandatory before any clinical claim.** LOO-CV on MMRF N=787 is internal validation only. The minimum publication standard for a clinical claim is one external cohort with pre-specified primary endpoint. IFM2009 or MRC Myeloma XI are the most accessible starting points.

4. **IMWG response calibration study.** If the goal is eventual clinical decision support, a bridge study is needed: take a cohort with both RNA-seq and IMWG response calls, fit a calibration curve from predicted PFS risk score to observed depth-of-response, and report the calibration error. Without this, the endpoint-validity gap flagged in the v6 audit remains open.

5. **Ancestry audit of training data.** Confirm the exact ancestry breakdown of the 787 MMRF patients used in v12. MMRF CoMMpass is ~72% non-Hispanic white; if this is the training set, prospective validation in a diverse cohort (e.g., SWOG S1211 which enrolled more minority patients, or the ongoing ECOG-ACRIN EA4151 which has explicit diversity requirements) is needed before generalizing claims.

6. **Re-run §4 with c-trials MCP.** Current trial landscape (NRG-MM, daratumumab+VRd combinations with MRD endpoints, venetoclax-combination trials in BCL2-high patients) must be surveyed to establish whether v12 adds anything beyond already-enrolled biomarker arms.

7. **p-value context in abstract.** If this work is submitted for publication, the abstract should report: ΔC = +0.021 (95% CI [+0.001, +0.043], p=0.046, internally validated, N=787 LOO). It should not report the S4 stratum lift (+0.057) without the specialist failure caveat in the same sentence.

---

## Summary

| Dimension | Assessment |
|---|---|
| Overall ΔC clinical significance | Below ΔC > 0.05 threshold; marginal, not transformative |
| Confidence interval | Lower bound near zero; not robustly positive |
| S2 t(4;14) | Most credible per-stratum signal; still below MCID; substudy-eligible |
| S4 t(11;14) | Specialist FAIL — S4 lift unstable and potentially misleading |
| S3 +1q21 | Zero lift — critical gap for R-ISS ultra-high-risk |
| IMWG calibration | Absent — fundamental construct gap persists |
| Clinical decision support | Not ready |
| Trial use | Exploratory biomarker substudy only |
| Publication classification | Research/methods paper, NOT clinical validation paper |
| External validation | Required before any clinical claim |
