# V12 Refactor Audit — CHAIR VERDICT

**Author:** integration-chair (synthesis)
**Date:** 2026-05-07
**Branch:** v6 / worktree `tier1-refactor`
**Inputs reconciled:** all 8 Wave-1 reports under `docs/V12_REFACTOR_AUDIT/`
plus JSON under `paper/v8_artifacts/v12_refactor_audit/`. The `ground_truth_run`
deliverable referenced in the brief was rolled into `metrics_audit.md` (12/13
RUNS.md claims reproduce; 1 SVI-non-determinism discrepancy on PPC T2
t(11;14) dispersion noted there). The literature sweep (Tier 2.C) is BLOCKED:
PubMed, bioRxiv, ClinicalTrials.gov, and Hugging Face MCP access were denied
in this session — see `sota_comparison.md` §Provenance and `clinical_translation.md`
§4. No SOTA PMID/DOI or NCT-ID is asserted in this verdict.

---

## §1 Executive verdict

**bioRxiv: YES, now** — the artifact is internally consistent, 12/13 RUNS.md
claims reproduce numerically (`metrics_audit.md` §6), the falsification record
is honest, and the per-stratum and per-method comparator tables in
`data_integration_validation.md` §3 are publishable as-is. **Briefings in
Bioinformatics / Cell Reports Methods: CONDITIONAL** on (a) downgrading the
"MOFA+ shared-factor" framing to "29 unsupervised expression factors of any
kind close the v11_richer gap" and (b) carrying the S4 t(11;14) specialist
STRICT FAIL caveat anywhere the +0.057 S4 lift appears
(`clinical_translation.md` §2, §6, §7). **Nature Methods: NOT YET** — v12
does not beat v11.5 (combo C=0.6751 < v11.5 C=0.6955; `metrics_audit.md` §2,
`sota_comparison.md` §2), v11.5 only TIES mmSYGNAL on the headline metric
(`sota_comparison.md` §2 §3.1), and the only clean strict-pass with novelty
(F_S5 5/5 within ±3%) needs external-cohort replication
(`sota_comparison.md` §5). **ICML / NeurIPS as primary venue: NOT YET** — the
contribution is governance + falsification framework, not a SOTA-beating
architecture; AI4Science / LMRL workshop track is fit-for-purpose now
(`sota_comparison.md` §5).

---

## §2 Uniqueness brief — pick the strongest niche

I evaluated the three candidates that `sota_comparison.md` §4 surfaced and
pick **(A) Pre-registered F-gates with published strict fails** as the
strongest defensible niche. Reasoning, with explicit comparison against (B)
and (C):

**Why (A) wins.** The v10/v11/v12 sprint record ships ten pre-registered
gates (F1–F10) with thresholds set BEFORE evaluation, plus documented
**STRICT FAILS** that the project allowed itself to publish: F8 0/16 cells
at N_paired=29 (Sprint 7), F10 E-value 1.21 < 1.5 (Sprint 4,
`causal_audit.md` §3 confirms STRUCTURAL FAIL), F5-paired refuted (Sprint
2), AML-native F3 0/5 (Sprint 6), v12 t(11;14) specialist Δ_S4=+0.0135
p=0.446 (RUNS.md row 36, surfaced in `clinical_translation.md` §2).
`sota_comparison.md` §4 makes the structural argument: none of the seven
SOTA comparators it could enumerate publishes a falsification gate it is
allowed to fail. mmSYGNAL ships a transcriptional regulatory map and a risk
score; it does not ship a pre-registered gate it has publicly failed. This
is a category difference, not a degree difference — and it is robust to the
two big losses (no v12 SOTA-beating C-index; literature sweep blocked).

**Why not (B) per-stratum F_S5 5/5 within ±3% as primary.** It is the only
clean strict-pass with novelty (`visualizations.md` Figure 6:
Ridge_v11features all 5 strata within ±0.006 of 0.90), but it is a result on
ONE cohort at ONE institution, with the v12 MOFA recalibration explicitly
NOT yet demonstrated (`clinical_translation.md` §5 "Calibration"). Until
external replication lands, F_S5 is the **strongest evidence** for niche (A),
not a niche on its own. Promote it from "primary niche" to "lead piece of
evidence inside niche (A)."

**Why not (C) honest cross-disease non-transfer reporting as primary.**
Sprint 6 AML 0/5 F3 is the right outcome (MM proteasome is MM-specific,
biologically expected — `causal_audit.md` §1 supports this), but reporting
a refute outcome is a *consequence* of niche (A), not a separate niche. Fold
into (A) as a second example of "the framework that names its own failures."

**Defensible one-line claim for the abstract.** "ResistanceMap is, to our
knowledge, the only multi-omics MM survival model that pre-registers
falsification gates with numerical thresholds AND publishes the gates it
fails." This is defensible because (i) the gates exist on disk before each
sprint runs (`v12_5_sprint_plans.md` §F-gates sections show the format), (ii)
strict fails are documented in RUNS.md and reproduced in `metrics_audit.md`
§5, and (iii) `sota_comparison.md` §4 found no comparator that publishes
this. The "to our knowledge" hedge is mandatory because the literature sweep
was blocked.

---

## §3 Material findings that must be reflected in any v12 manuscript

In priority order:

**(i) MOFA isn't MOFA on this cohort.** The v12 mofapy2 call is single-view
PPCA-with-ARD on log1p(TPM), not multi-view shared-factor inference
(`data_integration_validation.md` §1, citing
`scripts/v12/s_v12_mofa_vs_v11_5_pfs.py:217-224` — single view, single
group, Gaussian likelihood, ARD-on-weights). Empirically: PCA(K=29)+v11_richer
reaches Δ=+0.0217 p_boot=0.045, ICA+v11_richer Δ=+0.0217 p=0.047,
JIVE-joint+v11 Δ=+0.0214 p_boot=0.034, and MOFA+v11 (v12) Δ=+0.0214 p=0.046
— all indistinguishable (`data_integration_validation.md` §3 head-to-head
table). JIVE-joint *alone* reaches C=0.6763 > MOFA+v11 combo C=0.6751
(+0.0226 single-model lift, 95% CI [-0.001, +0.048]). Any manuscript text
calling this "multi-omics MOFA+" must be replaced with "29 unsupervised
expression factors of any reasonable kind." Recommended live-pipeline fix:
replace mofapy2 with `sklearn.PCA(K=29)` (zero new dependencies, identical
performance, `data_integration_validation.md` §5 step 4).

**(ii) v12 does not beat v11.5.** v12 MOFA+v11 combo marginal C=0.6751 sits
**below** v11.5 Cox_v11_routed_mmsygnal C=0.6955 (Δ=−0.0204; both reproduce
to 0.0000 against RUNS.md, `metrics_audit.md` §2). The +0.0214 lift cited
for v12 is against `Cox_v11_richer` (C=0.6538, RM-internal baseline), not
against v11.5. v11.5 itself is a TIE with mmSYGNAL routed (C=0.694, paired-
bootstrap 99% CI excludes |Δ|>0.036; `sota_comparison.md` §2). Therefore the
v12 manuscript headline cannot be "beats SOTA," "beats mmSYGNAL," or "beats
v11.5." It can be: "an architectural opportunity at the L1 discrimination
layer over an internal feature-richer baseline, borderline-significant
(p=0.046), CI lower bound +0.0005 — fragile to one bootstrap reseed."

**(iii) S4 t(11;14) +0.057 is a SAFETY FLAG, not a positive result.** The
v12 BCL2-family specialist head FAILED its own pre-registered gate
(Δ_S4=+0.0135, p=0.446; RUNS.md row 36). The +0.057 S4 lift in the combo
model therefore likely reflects regularization / leakage from MOFA latent
factors rather than reliable t(11;14) biology
(`clinical_translation.md` §2 row S4). t(11;14) is the venetoclax-decision
stratum (BELLINI-context); a spuriously positive prediction here could
mis-route a patient away from BCL2-targeted therapy
(`clinical_translation.md` §5). Required text: anywhere the S4 number
appears, the specialist STRICT FAIL must appear in the same sentence
(`clinical_translation.md` §6 step 7).

**(iv) Pearl-tier ceiling holds at L1-with-structural-prior.** F8 + F9 PASS,
F10 STRUCTURAL FAIL persists under v12 (`causal_audit.md` §3): MOFA factors
add ~3% incremental variance explained, insufficient to move the NIE CI
lower bound from HR=1.029 to the HR≈1.13 needed for E-value ≥ 1.5. F8-DPS
STRICT FAIL is operator-independent and N_paired=29 driven; v12 adds no new
paired (z0,z1) data. Any manuscript phrase implying "treatment selection,"
"counterfactual," "predicts trajectory before it happens," or "causal
identification of new pathways" requires a hedge — `causal_audit.md` §6
provides the safe-claim list.

---

## §4 Required code/text changes BEFORE submission (NOT applied — list)

1. **(text)** Anywhere the v12 result is described as "MOFA+ shared
   factors," replace with "29 unsupervised expression factors (PCA / MOFA
   single-view PPCA equivalent on this slice)." Source:
   `data_integration_validation.md` §1, §6 risk 1.
2. **(text)** Anywhere v12 is described as "beats v11" or "improves over
   SOTA," replace with "is an L1 architectural opportunity over the
   feature-richer internal baseline; v12 does NOT exceed v11.5 (ΔC=−0.020)
   and v11.5 ties mmSYGNAL." Source: `metrics_audit.md` §2,
   `sota_comparison.md` §2.
3. **(text)** Anywhere the S4 t(11;14) +0.057 lift is reported, the v12
   specialist STRICT FAIL (Δ_S4=+0.0135, p=0.446, RUNS.md row 36) must
   appear in the same sentence and the bedside-warning ("not safe for
   clinical inference") in the same paragraph. Source:
   `clinical_translation.md` §2, §5, §6.
4. **(text)** Honest-caveats block must contain: (i) p=0.046 with single
   comparison only, no Bonferroni survival under the 16-method panel
   (`data_integration_validation.md` §6 risk 3); (ii) CI lower bound +0.0005
   means one censored event flip changes the sign; (iii) K=64 mofapy2 has
   not converged (ELBO still moving 1.57% per 5 iters at iter=20,
   `data_integration_validation.md` §2); (iv) PHATE → UMAP fallback
   disclosed in figure captions (`visualizations.md` header).
5. **(text)** Apply RC-2 hedge from `causal_audit.md` §5: do NOT describe
   mmSYGNAL as a "treatment selection" benchmark. Apply RC-4 hedge: NIE is
   "supporting directional evidence," not identification. Add the safe-claim
   list from `causal_audit.md` §6 to the Methods or Limitations section
   verbatim.
6. **(code, optional but strongly recommended)** Replace
   `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py` mofapy2 call with `sklearn.PCA(K=29)`
   on the same HVG-5000 log1p TPM matrix, with mofapy2 retained behind
   `RM_INCLUDE_MOFA_FACTORS=1` env-gate
   (`data_integration_validation.md` §5 steps 1–4). Zero dependency cost,
   identical statistical conclusion, removes the false multi-omics framing.
7. **(code, defer to v12.5)** Apply the §3 parallelism wins from
   `tier1_dead_code.md` (5–20× expected on `s4d_nie_estimation.py`,
   `s5_mondrian_jackknife_plus.py`, `s3fix_v3.py`); archive 18.5 kLOC dead
   code per `tier1_dead_code.md` §1, §2 — none touches an F-gate script.
8. **(figures)** Confirm all 6 real-data figures from `visualizations.md`
   (PHATE→UMAP fallback noted; F8 strict-fail caveat in Fig 2; rank-based
   residuals in Fig 5) replace any prior `generate_paper_figures.py` panels
   that used `np.random` (per the user's standing memory on figure
   fabrication risk).

---

## §5 v12.5 sprint priority order

Per `v12_5_sprint_plans.md` §Cross-sprint notes (≤1 GPU-hr H100 + ~2 CPU-hr
total, all three independently parallelizable):

1. **Sprint C — Touzeau 2014 5-gene t(11;14) classifier** [HIGHEST
   PRIORITY]. Directly attacks the §3(iii) safety flag; same pre-registered
   threshold (Δ_S4 > +0.05, 95% CI excludes 0) the v12 specialist already
   failed at, so the test is honest. ~40 min CPU. If F_S4_PASS triggers, the
   manuscript can carry a credible t(11;14) story; if it fails with
   F_S4_PANEL_PRESENT passing, that is a publishable negative result that
   refutes bulk-RNA transferability of Touzeau without refuting Touzeau
   itself (`v12_5_sprint_plans.md` Sprint C negative-result protocol).
2. **Sprint B — NUTS HMC + per-stratum heteroscedastic σ_obs** [SECOND
   PRIORITY]. Closes the 5/10 PPC failure surfaced as a v12 caveat; ~45 min
   CPU, no GPU. Formalizes a new gate F1_PPC ≤ 2/10 (currently 5/10). If
   PPC stays > 2/10 after NUTS, that is an honest negative result naming
   HBayes class-misspecification (not inference) — still fits niche (A).
3. **Sprint A — scFoundation / ComBat-corrected encoder** [THIRD PRIORITY,
   GATED]. ≤ 1.0 GPU-hr H100. Highest risk of failing F2 (Geneformer raw
   already failed F2 zero-trust per `r-2026-05-03-v11s1-encoder-reject`),
   which is the right outcome for niche (A) — a publishable encoder-rejection
   record. Run in parallel with B and C only if H100 is free; otherwise
   defer.

---

## §6 Known gaps

1. **Literature sweep BLOCKED.** PubMed, bioRxiv, ClinicalTrials.gov, and
   Hugging Face MCPs returned permission-denied in this session
   (`sota_comparison.md` §Provenance, `clinical_translation.md` §4 header).
   Six of the eight comparators in `sota_comparison.md` §1 have unverified
   metric / dataset cells and must be re-run with MCP access before
   submission. No NCT-IDs, no PMIDs cited in this verdict.
2. **External MM cohort missing.** All v11.5 / v12 / F_S5 results are LOO on
   MMRF N=787 only. `clinical_translation.md` §3 enumerates IFM2009,
   HOVON-65/GMMG-HD4, MRC Myeloma XI, MMRC CoMMpass-disjoint as the
   highest-priority external cohorts; access requires DUAs not yet
   initiated. Until at least one external cohort lands, the per-stratum
   conformal coverage claim is single-institution evidence, not a
   methodological generalization (the Nature Methods bar from
   `sota_comparison.md` §5).
3. **K=64 MOFA convergence not closed.** `data_integration_validation.md`
   §2 shows K=64 still moving 1.57 % per 5 iters at iter=20; projected
   50.4 min to convergence is borderline against the 1-hour CPU budget. v12
   ran K=32 (K_eff=29) and the K=64 deviation from the pre-registered plan
   is documented but not resolved. Decision: since the recommended
   replacement is PCA(K=29), this gap becomes moot if §4 change 6 is
   accepted; otherwise a 1-hour mofapy2 K=64 run must close it.
4. **Paired modalities (CNV / proteomics) absent.** Single-view regime is
   the worst case for any multi-view method (`data_integration_validation.md`
   §6 risk 5); the +0.0214 lift attributed to MOFA cannot be defended as
   multi-omics value until paired data lands. F2 N_paired=27 and F8 N_paired=29
   limits remain Stone-bound (`causal_audit.md` §4).
5. **F10 STRUCTURAL FAIL persists** under v12 (`causal_audit.md` §3 verdict).
   This is a known feature, not a regression, but any future text that
   implies otherwise must be caught.

---

## §7 Memory updates needed

The standing memory `project_resistancemap_floor_evidence.md` ("RM ties
MOFA+Ridge (19.7s) and loses to predict-mean on pooled MSE") needs nuance,
not retraction. Recommended replacement text:

> ResistanceMap drug-MSE story: predict-mean floor (2.8106) < MOFA+Ridge
> (2.8155) < RM 10-agent DAG (2.8364) on CCLE 11-drug task — ranking
> reproduced 2026-05-07 in `metrics_audit.md` §4 against
> `pipeline_validated.pt`. Panobinostat (MSE=22.6) dominates pooled MSE
> across all models; per-drug RM beats baseline on Bortezomib, Romidepsin,
> Etoposide, Lenalidomide by Spearman (`metrics_audit.md` §7 note 2).
> Survival story: v11.5 Cox_v11_routed_mmsygnal C=0.6955 TIES mmSYGNAL
> 6-submodel C=0.6957 (paired-bootstrap 99% CI excludes |Δ|>0.036). v12
> MOFA+v11 combo C=0.6751 is **below** v11.5 by 0.020; the v12 +0.0214
> lift is over the v11_richer internal baseline only, not over v11.5 or
> mmSYGNAL. v12 does not change either of the two TIE outcomes. Defensible
> niche is the falsification framework + per-stratum conformal coverage,
> NOT raw discrimination.

Additionally, add a new memory pointer:
`project_v12_refactor_audit_outcome.md` capturing (a) the four §3 material
findings, (b) the §2 uniqueness brief verdict, (c) the §5 sprint priority
ordering. Cross-link to all 8 Wave-1 reports.

---

## Closing falsification

This verdict would flip from CONDITIONAL_PASS to FAIL if any of the
following becomes true:

1. The §3(i) "MOFA isn't MOFA" finding is rebutted by a re-run of
   `data_integration_validation.md` §3 with paired CNV / proteomics in
   which mofapy2 demonstrably exceeds PCA / ICA / JIVE by Δ ≥ +0.01 with
   p_boot < 0.01. (Outcome: niche brief stays; multi-omics framing is
   restored.)
2. The §3(iii) Sprint C Touzeau panel **passes** F_S4_PASS (Δ_S4 > +0.05,
   95% CI excludes 0) with F_S4_PANEL_PRESENT also passing. (Outcome: S4
   safety flag downgrades; t(11;14) story becomes a positive contribution
   to the v12.5 manuscript.)
3. External cohort (IFM2009 or MRC Myeloma XI) replicates F_S5 5/5 within
   ±5%. (Outcome: Nature Methods candidacy becomes defensible; current
   "NOT YET" lifts.)

Word count: ~2 380.
