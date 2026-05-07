# V12 Refactor Audit — WAVE-3 VERDICT

**Author:** integration-chair (synthesis, Wave 3; persisted by main thread)
**Date:** 2026-05-07
**Branch:** v6 / worktree `tier1-refactor`
**Status:** EXTENDS `docs/V12_REFACTOR_AUDIT/CHAIR_VERDICT.md` (Wave 1). Does NOT replace it.
**Inputs reconciled:** all 8 Wave-3 reports under `docs/V12_REFACTOR_AUDIT/wave3/` plus JSON under `paper/v8_artifacts/v12_refactor_audit/wave3/`. Live PubMed / bioRxiv / HF MCP access was restored for Wave 3 (chair §6 gap 1 partially closed); Open Targets MCP still deferred.

---

## §1 What changed since Wave 1

**Wave-1 §4 changes now IMPLEMENTED.**
- §4 item 6 (PCA substitution): DONE in `scripts/v12_refactor/wave3/s_v12_pca_vs_v11_5_pfs.py`. Original v12 mofapy2 script unmodified per hard rule. Lift preserved within ±0.001 on Δ and ±0.001 on p (PCA Δ=+0.0217 p_boot=0.0450 vs MOFA Δ=+0.0214 p_boot=0.0456); V12_ARCHITECTURAL_OPPORTUNITY verdict gate fires identically; mofapy2 retained behind `RM_INCLUDE_MOFA_FACTORS=1` env-gate (`pca_substitution.md` §S1, §S4). Factor extraction 77.0 s → 0.038 s (-2026×).
- §4 item 7 (parallelism): DESIGNED but NOT applied. Three diff patches with bit-identicality proof at `RM_N_JOBS=1` written; `verify_parallelism.py` must run on a 16-core host before `git apply` (`parallelism_patches.md` §5). Expected speedups 8–24× per target.

**Negative results that LANDED in Wave 3.**
- Sprint C (Touzeau t(11;14) 5-gene panel): PRE-COX REFUTE. `F_S4_PANEL_PRESENT` fired at 1/5 (threshold ≥3/5); only `BCL2L1` cleared the |Cohen-d|>0.5 bar; BAX moves in the wrong direction (`sprint_c_touzeau.md` §2). Published negative result against the bulk-RNA-as-proxy-for-BH3-function hypothesis on MMRF, NOT against Touzeau itself (Touzeau measured mitochondrial cytochrome-c release, not transcript abundance). Reinforces chair §3(iii) S4 safety flag.
- PMID correction: chair-cited PMID 24305167 does not resolve. Actual paper is PMID 23860449 (Leukemia 2014, DOI 10.1038/leu.2013.216), which contains no discrete 5-gene panel — only BCL2/MCL1 ratio + qualitative BCL2-family expression (`sprint_c_touzeau.md` §1). Sprint plan must be amended.
- Master comparison: 0/8 methods survive Bonferroni at α=0.05/8=0.00625 (or α=0.05/16=0.003125 per W3.5 viz-curator's stricter framing); JIVE-joint alone reaches DeLong p=0.0298, falling 4.8× short of m=8 family-wise threshold. The +0.021 lift is method-agnostic — five methods cluster within 0.0043 C-index (`master_comparison.md` §3, §4).

**Novelty claims SHARPENED.**
- Niche-A is now QUANTIFIED: 0/11 audited papers satisfy Q1 (numeric falsification threshold), Q2 (published failure), or Q3 (pre-registered before evaluation). Targeted PubMed query returns 0 hits. RM ships 7 distinct published gate-failure records (`niche_a_novelty.md` §1, §2). The "to our knowledge" hedge is necessary; the claim is defensible.
- Interpretability (PRELIMINARY — full SHAP runs blocked, biological-prior predictions only): v11.5's entire +0.04 lift comes from ONE feature (`mmsygnal_routed`); stripped of mmSYGNAL routing v11.5 collapses to v11_richer (`interpretability_demo.md` §A.1). v12's +0.021 lift is ~70% concentrated in S2 (+0.040) + S4 (+0.057); in S1/S3/S5 v12 adds ≤+0.014 (`interpretability_demo.md` §A.2). The v11.5 "ties mmSYGNAL" framing now sharpens to "v11.5 IS mmSYGNAL once you remove the routed score."

**New comparator surfaced.**
- Hussain & Sontag 2024, *npj Digital Medicine* PMID 39075240, joint NDMM/RRMM transformer N=703 + N=720 external. Direct MM-PFS competitor at trial scale; was NOT in the Wave-1 SOTA list (`literature_sweep_v3.md` §2). Must be benched against in any submission. SurvBoard 2025 (PMID 41031875) provides a leaderboard route to test the v12 deep stack vs simple statistical baselines.

---

## §2 Updated submission-readiness verdict

**bioRxiv: YES, now** — every chair §4 text correction has a Wave-3 artifact backing it; PCA substitution removes the multi-omics-framing risk; Touzeau negative result strengthens, rather than weakens, the falsification-framework story; figures are real-data (no `np.random`).

**Briefings in Bioinformatics / Cell Reports Methods: CONDITIONAL PASS** — sunset 2026-09-30. Re-evaluation criterion: (i) Hussain/Sontag head-to-head submitted at minimum as supplement; (ii) `master_comparison.md` Bonferroni framing (m=8 or m=16) carried verbatim into Methods; (iii) Sprint B (NUTS HMC PPC) lands or is documented as deferred negative result.

**Nature Methods: NOT YET — UNCHANGED from Wave 1.** v12 still does not beat v11.5 (combo C=0.6751 < v11.5 C=0.6955). 0/8 methods survive Bonferroni. F_S5 5/5 conformal still single-cohort. Wave 3 ADDED: Hussain/Sontag is the field's external benchmark — Nature Methods candidacy is gated on at least matching their ISS-delta on TOURMALINE-or-equivalent.

**ICML / NeurIPS as primary venue: NOT YET — UNCHANGED.** AI4Science / LMRL workshop track remains fit-for-purpose.

**NEW recommendation — SurvBoard 2025 leaderboard submission (PMID 41031875).** Low cost (~2-3 person-days), directly tests the v12 deep stack vs RSF/BlockForest under standardized preprocessing. Outcome is publishable in either direction.

---

## §3 Wave-3 material findings to add to the v12 manuscript (priority order)

**(i) PCA = MOFA on this slice — no dependency cost to fix the framing.** The 0.038-s `sklearn.PCA(K=29)` substitution preserves Δ within ±0.001 and verdict gate fires identically. Adopt the substitution as the live pipeline; keep mofapy2 behind env gate. Manuscript text becomes the chair §3(i) verbatim language: *"29 unsupervised expression factors of any reasonable kind close the v11_richer gap by Δ≈+0.021."*

**(ii) Bonferroni context for the headline p-value.** The 8-method panel must accompany the Δ=+0.0217 figure. Sentence template: *"PCA(K=29)+v11_richer Δ=+0.0217, p_boot=0.045 (single comparison; 0/8 methods clear α=0.05/8 Bonferroni in the 8-method panel of `master_comparison.md` §1)."*

**(iii) Sprint C result is a featured negative result.** Add Methods § "Touzeau-panel pre-Cox sanity gate (F_S4_PANEL_PRESENT)" with 1/5 result, BCL-XL down-regulation observation (Cohen-d=−0.583), and explicit statement that Touzeau itself is NOT refuted (BH3-profiling ≠ bulk-TPM). Cite PMID 23860449 (NOT 24305167). Promote `F_S4_PANEL_PRESENT` to permanent F-gate.

**(iv) Hussain/Sontag must appear in Related Work.** *npj Digital Medicine* PMID 39075240 is the direct MM-PFS comparator at trial scale; absence from a v12 submission is a credibility risk. Frame v12 as orthogonal (multi-omics LOO + falsification gates on N=787 MMRF) vs Hussain/Sontag (joint event-prediction transformer on N=703 + N=720 trial cohorts). Head-to-head benchmark is rank-1 priority for v12.5+.

**(v) Niche-A claim text is now defensible at full strength.** Insert `niche_a_novelty.md` §5 stronger Methods variant verbatim. Keep abstract claim with "to our knowledge" hedge — not optional.

**(vi) Interpretability §A findings stand on grounded JSONs.** v11.5-collapses-to-v11_richer-without-mmsygnal_routed and the v12 S2+S4 ≈70% concentration are firm. The 3-patient SHAP rankings are biological-prior PREDICTIONS, not measured values; do NOT include patient-level SHAP plots in the manuscript until `interpretability_demo.py` is re-run. Aggregate-level Patient C safety-flag framing IS publication-ready as a Limitations vignette.

**(vii) FHRMM falsifier (PMID 42057483).** Functional-HR MM cases lose baseline-cytogenetics + SKY92 prognostic value. Any v12 manuscript claiming MM-PFS prediction must address whether the model retains discrimination inside the FHRMM subset, or explicitly excludes that population.

---

## §4 REVISED required pre-submission changes (Wave-1 §4 rolled forward)

| # | Source | Status | Action remaining |
|---|---|---|---|
| 1 | Chair §4.1 (downgrade "MOFA+ shared factors" framing) | NEEDED — text-only | Manuscript edit; `pca_substitution.md` §S4 supplies replacement language |
| 2 | Chair §4.2 (drop "beats v11" / "improves over SOTA") | NEEDED — text-only | Manuscript edit |
| 3 | Chair §4.3 (S4 specialist STRICT FAIL co-located with +0.057) | REINFORCED by Sprint C | Add `sprint_c_touzeau.md` §2 result alongside RUNS.md row 36 in same paragraph |
| 4 | Chair §4.4 (honest-caveats block) | UPGRADED — Bonferroni m=8/16 result available | Use `master_comparison.md` §3 verbatim |
| 5 | Chair §4.5 (causal hedges / RC-2, RC-4) | NEEDED — text-only | Unchanged from Wave 1 |
| 6 | Chair §4.6 (PCA substitution) | DONE | Adopt PCA path as default in next sprint; current new file is side-by-side, not replacement |
| 7 | Chair §4.7 (parallelism) | DESIGNED, NOT APPLIED | Run `verify_parallelism.py` on 16-core host; if bit-identical at `RM_N_JOBS=1`, `git apply` the patches |
| 8 | Chair §4.8 (figures replace any np.random panels) | DONE for 3 of 4 W3 figures | Calibration ladder skipped — ship existing v10 calibration ladder instead |

**NEW Wave-3-only required changes.**

| # | Source | Action |
|---|---|---|
| W1 | `sprint_c_touzeau.md` §1 | **PMID correction in `v12_5_sprint_plans.md` Sprint C: 24305167 → 23860449.** Documentation defect, not science defect. |
| W2 | `sprint_c_touzeau.md` §5 item 3 | Add `F_S4_PANEL_PRESENT` to permanent F-gate registry. Definition: ≥3/5 panel genes show |Cohen-d|>0.5 between target stratum and rest, evaluated BEFORE any Cox lift. |
| W3 | `master_comparison.md` §3 + Fig W3.1 caption | Add Bonferroni m=16 row to manuscript Table 1. Label: "no method clears family-wise α=0.003125." |
| W4 | `literature_sweep_v3.md` §2 row 1 | Hussain & Sontag (PMID 39075240) added to Related Work. Frame v12 as orthogonal, NOT competitive (until head-to-head runs). |
| W5 | `interpretability_demo.md` §A.1 | New manuscript sentence: "v11.5's entire +0.04 marginal lift over v11_richer is attributable to a single engineered feature (mmsygnal_routed); v11.5 stripped of this feature is identical to v11_richer." |
| W6 | `parallelism_patches.md` §5 | Before any production run with `RM_N_JOBS=-1`, gate on the verifier passing on real production host. |

---

## §5 REVISED v12.5 sprint priority

Sprint C is now DONE with a published negative result. Re-rank:

**Rank 1: Sprint D — Hussain/Sontag head-to-head on MMRF CoMMpass.** Newly required. Reproduce their joint event-prediction transformer; report C-index, IBS, calibrated PFS@12/24/36mo. MEDIUM cost (~3-5 person-days, <100 GPU-hr). Outcome publishable either way under niche A.

**Rank 2: Sprint B — NUTS HMC + per-stratum heteroscedastic σ_obs.** Closes the 5/10 PPC failure caveat. ~45 min CPU; gate F1_PPC ≤ 2/10. If PPC stays > 2/10, publishable as HBayes class-misspecification.

**Rank 3: SurvBoard 2025 leaderboard submission.** Low-medium cost; directly tests floor-evidence concern; standardized preprocessing answers "is the deep stack justified" in either direction.

**Rank 4: Sprint A — scFoundation / ComBat-corrected encoder.** Demoted: literature confirms NO public bulk-RNA + ComBat-corrected protocol exists for any scFM; running this is high-risk-of-F2-fail by field-wide gap, not RM-specific implementation. v12.5's near-term bottleneck is external-cohort replication, not encoder upgrade. Defer to v13 unless H100 is idle.

**Sprint C closure:** keep `F_S4_PANEL_PRESENT` as the lasting deliverable. Do NOT re-run with the existing 5-gene panel. Re-open conditional on (a) Touzeau 2014 supplement furnishing a different discrete panel, or (b) entirely new t(11;14) panel passing F_S4_PANEL_PRESENT pre-Cox.

---

## §6 Publication-readiness CHECKLIST

### 6A — bioRxiv submission (target: now)

| # | Gate | Status | Artifact pointer | Owner-action |
|---|---|---|---|---|
| 6A.1 | All 13 RUNS.md numeric claims reproduce within ±0.0001 (modulo 1 SVI non-determinism) | DONE | `metrics_audit.md` §6 + `ground_truth_run.md` | None |
| 6A.2 | "MOFA+ shared factors" framing replaced cohort-wide | NOT STARTED | Chair §4.1; W1, §3(i) | Manuscript edit |
| 6A.3 | "v12 beats v11.5" / "beats SOTA" claims removed | NOT STARTED | Chair §4.2 | Manuscript edit |
| 6A.4 | S4 t(11;14) +0.057 always co-located with specialist STRICT FAIL + Sprint C refute | NOT STARTED | Chair §4.3; W3 `sprint_c_touzeau.md` §2 | Manuscript edit |
| 6A.5 | Bonferroni m=8 (or m=16) caveat in headline | NOT STARTED | `master_comparison.md` §3; W3 W3 | Manuscript edit + Table 1 update |
| 6A.6 | Causal hedges (RC-2, RC-4) applied | NOT STARTED | Chair §4.5 | Manuscript edit |
| 6A.7 | Niche-A "to our knowledge" hedge in abstract; stronger variant in Methods | NOT STARTED | `niche_a_novelty.md` §5 | Manuscript edit |
| 6A.8 | Figures use real-data Wave-3 panels (3/4 ready; calibration ladder = Wave-1 v10 fig) | DONE | `wave3_figures.md` Fig W3.1/W3.2/W3.4 | Verify caption-md provenance copied into manuscript |
| 6A.9 | PMID 23860449 (NOT 24305167) cited for Touzeau anywhere it appears | NOT STARTED | `sprint_c_touzeau.md` §1 | Manuscript + sprint plan edit |
| 6A.10 | Hussain/Sontag PMID 39075240 in Related Work | NOT STARTED | `literature_sweep_v3.md` §2 | Manuscript edit |

### 6B — Nature Methods candidacy (target: 2027-Q1 earliest)

| # | Gate | Status | Artifact pointer | Owner-action |
|---|---|---|---|---|
| 6B.1 | External MM cohort replicates F_S5 5/5 within ±5% | NOT STARTED | Chair §6 gap 2 | DUA initiation: IFM2009, MRC Myeloma XI, or HOVON-65 |
| 6B.2 | At least one method survives Bonferroni m≥8 on a primary metric | NOT STARTED | `master_comparison.md` §3 | Either (a) larger paired-modality cohort, or (b) tighter primary metric |
| 6B.3 | v12 (or successor) matches-or-beats Hussain/Sontag ISS-delta on TOURMALINE-equivalent cohort | NOT STARTED | `literature_sweep_v3.md` §6 row 1 | Sprint D |
| 6B.4 | Paired modalities (CNV + proteomics + RNA) at N≥100 | NOT STARTED | Chair §6 gap 4 | Cohort acquisition (DUA) |
| 6B.5 | Either F8-DPS strict pass at higher N_paired OR architectural acknowledgement that Pearl ceiling stays at L1 | PARTIAL | `causal_audit.md` §3 | Either acquire larger paired set, or commit to L1 framing in cover letter |
| 6B.6 | F_S4_PANEL_PRESENT promoted to permanent F-gate in repo + sprint plan | NOT STARTED | `sprint_c_touzeau.md` §5 item 3; W3 §4 W2 | Sprint plan edit |
| 6B.7 | Parallelism patches applied + bit-identicality verified on production host | NOT STARTED | `parallelism_patches.md` §5; W3 §4 W6 | Run verifier; apply patches |

---

## §7 Memory updates needed

Extend `project_v12_refactor_audit_outcome.md` with Wave-3 deltas:

> **Wave-3 deltas (2026-05-07).** All four Wave-1 conclusions HOLD. PCA(K=29) substitution preserves the v12 lift within ±0.001 on Δ and ±0.001 on p; adopt as live pipeline. Sprint C Touzeau test REFUTED at the pre-Cox sanity gate (`F_S4_PANEL_PRESENT` 1/5) — published negative result, NOT a sprint failure; Touzeau itself is not refuted (BH3-profiling ≠ bulk-TPM). PMID corrected: 24305167 → 23860449. 0/8 methods survive Bonferroni at α=0.05/8 in the master comparison; +0.021 lift is method-agnostic. Niche A quantified: 0/11 audited papers satisfy any of {pre-set numeric falsification threshold, published gate failure, pre-registration} — RM is 1-of-1 in the audited set. New direct MM-PFS comparator surfaced: Hussain/Sontag 2024 *npj Digital Medicine* PMID 39075240 — must be benched against in any submission. v11.5 single-feature dependency (entire +0.04 lift on `mmsygnal_routed` alone) sharpens the v11.5/mmSYGNAL TIE into "v11.5 IS mmSYGNAL once you remove the routed score." `F_S4_PANEL_PRESENT` is the new permanent pre-Cox biological-sanity gate.

Also append to `project_resistancemap_floor_evidence.md` referencing master_comparison §1 and Fig W3.4 — the per-drug Spearman wins (Bortezomib, Romidepsin, Etoposide, Lenalidomide) survive Wave-3 audit; the Panobinostat-driven aggregate floor failure is unchanged.

---

## §8 Closing falsification

This Wave-3 verdict (CONDITIONAL_PASS for bioRxiv now; NOT_YET for Nature Methods, with sunset 2026-09-30 for Briefings/CRM conditional pass) flips to FAIL if:

1. **The PCA-MOFA equivalence breaks under paired modalities.** Chair closing falsifier 1 stands: if a re-run with paired CNV / proteomics shows mofapy2 exceeds PCA / ICA / JIVE by Δ ≥ +0.01 with p_boot < 0.01, multi-omics framing is restored and §3(i) is retracted. Wave-3 reinforces this falsifier — JIVE-joint already ties MOFA at Δ=+0.0214 with extraction time 0 s vs MOFA's 77 s on single-view.

2. **A re-audit of the niche-A comparator set surfaces a paper with Q1+Q2+Q3 all = Y.** `niche_a_novelty.md` §4 names selection-bias risk; if a wider TRIPOD-AI sweep finds even ONE comparator that publishes a pre-set numeric falsification threshold AND a documented gate failure AND pre-registration, the niche-A claim drops from "1-of-1" to "rare," and Methods must explicitly cite the counterexample.

3. **Sprint D fails to even match Hussain/Sontag's ISS-delta on a TOURMALINE-equivalent cohort, AND v12 cannot demonstrate a methodological niche orthogonal to their joint-prediction transformer.** Currently v12 framed as orthogonal (multi-omics + falsification gates + per-stratum conformal). If head-to-head shows v12 is BOTH worse on discrimination AND not differentiable on calibration / governance, the bioRxiv→Briefings conditional pass collapses and v12 needs new framing or new contribution.
