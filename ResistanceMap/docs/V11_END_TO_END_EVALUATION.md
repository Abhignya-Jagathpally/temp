# v11/v11.5 End-to-End Critical Evaluation

**Date:** 2026-05-03
**Scope:** ResistanceMap from data acquisition through v11.5 SOTA-tying claim
**Author:** parent-session integrative review (independent of architect-chair, causal-auditor, clinical-translator agents running in parallel)

This document is the *internal* honest read of the pipeline before the
chair verdict lands. It exists to ensure that when the chair, causal,
and clinical reports come back, they can be cross-checked against an
independent eval that did NOT see them first. It is also the document
that flags what to defend, what to hedge, and what to reject in the
manuscript.

---

## §1 What the pipeline *actually* does, end-to-end

### §1.1 Data path

```
MMRF CoMMpass IA22 (N=787 baseline-paired patients, 29 paired post-induction)
   ├─ baseline RNA-seq TPM (57,690 ENSG)         → mmrf_baseline_expression.parquet
   ├─ baseline copy-number Segment_Mean (chr×bp) → copy_number.tsv
   ├─ paired RNA-seq baseline+post-induction     → mmrf_paired_*.parquet (N=29)
   ├─ FISH cytogenetic indicators (5 strata)     → mmrf_sprint4_analysis.tsv
   ├─ tt2L_days, had_2L                          → PFS endpoint
   └─ STRING v12 PPI (12651 nodes)               → ppi graph for F6, RWR/multi-α
```

**Encoder:** v10 PCA(64) on the baseline TPM. The v11 plan §3 attempted
to replace this with scFoundation (blocked by packaging) → Geneformer
fallback (REJECTED by F2 zero-trust check, `r-2026-05-03-v11s1-encoder-reject`).
**v10 PCA(64) remains production.**

**Endpoint:** time-to-second-line therapy initiation. Right-censored.
787 patients with 351 events at the latest snapshot. This is a
PFS-proxy, not OS, and not IMWG-defined progression (which requires
serum free light-chain criteria not in MMRF IA22).

### §1.2 Core trained components

```
U_θ : ℝ^64 → ℝ              scalar potential (Sprint 1 + 2)
       fit by deep-score-matching with PPI Tikhonov regularizer
       N=787; 19,489 ENSG → STRING node mapping; full-channel STRING

HBayes posterior of resistance-mediator coefficients (Sprint 2)
       Bayesian NUTS over 23 v11_richer features
       N=787; F1+F2 PASS at v10; F2 LEAK at Geneformer-encoder
       (rejection — falsification framework working)

CellRank-2 trajectory probabilities (Sprint 3 / v10 baseline)
       on PPI-propagated baseline expression
       45 consensus MM driver mean-rank under multi-α attention
       (H=4 heads at α∈{0.3,0.5,0.7,0.9}, mean aggregator)
       — STRICT generalization of v10 RWR (H=1, α=0.7 bit-identical)

Cox PH (Sprint 5 / v11.5)
       lifelines penalizer=0.1, LOO C-index
       feature variants: v10 (11), v11features (15), v11_richer (23),
       routed_mmsygnal (24 = v11_richer + mmSYGNAL routed),
       six_mmsygnal_scores (29 = v11_richer + raw 6 program scores)

Mondrian jackknife+ conformal (Sprint 5)
       Barber-Candès-Ramdas-Tibshirani 2021 wrapper
       3 base learners: Ridge_v10, Ridge_v11features, GBM_v10
       5 cyto strata, 90% target coverage
```

**The pipeline is honest about what's trained and what's borrowed.** The
mmSYGNAL routed score is borrowed from Murie/Baliga 2025 (BJC PMID
40169765, GPL-3.0 caret/glmnet weights). The IA12 program-activity
matrix is borrowed from the same publication.

### §1.3 Falsification gates (pre-registered, post-only verdict)

| Gate | Test | v10 verdict | v11 verdict | Stone-bound expected? |
|---|---|---|---|---|
| F1 | latent-encoder reconstruction Δ | PASS | PASS at v10; LEAK at Geneformer (rejected) | – |
| F2 | zero-trust (predict-mean baseline) | REFUTED at v10 (correct null) | LEAKED at Geneformer (rejected by zero-trust rule) | – |
| F3 | proteasome-MM specificity | 6/10 STRICT PASS (MM); 0/5 (AML) — biologically expected | preserved | – |
| F4 | Helmholtz curl fraction | PASS (0.0146); framing corrected ∇×∇U=0 by identity | PASS, M1 Lyapunov NEW PASS 0/116 violations | – |
| F5 | paired latent change | substitute fails at N=29 → REFUTED | preserved REFUTED, expected | YES (Stone bound) |
| F6 | 45-driver mean-rank vs degree-matched null | 30.7σ STRICT PASS | 30.6σ STRICT PASS, T_obs improved by 64.9 ranks | – |
| F7 | drug-target / functional-residue | 2/3 STRICT (Vorinostat structural fail in BOTH oracles) | preserved | – |
| F8 | DPS single-snapshot identifiability | 0/16 STRICT FAIL | 0/16 STRICT FAIL, robust to operator | YES (Stone bound at d=64, N=29) |
| F9 | NIE | PASS | preserved | – |
| F10 | NIE-tipping | structural FAIL confirmed | preserved | YES (1-DAG limit) |
| F_S5 | per-stratum coverage 5/5 ±3% | STRICT PASS | STRICT PASS, marginal width −8 d | – |

**Net falsification score:** 5 STRICT PASS (F4 corrected, M1 NEW, F6
preserved+improved, F_S5 preserved+tightened, F3-MM); 2 STRICT FAIL
(F8-DPS, F10-NIE-tipping); both fails are pre-registered as
identifiability-bound — they are *expected* refutations under the Stone
minimax bound at N_paired=29, d=64.

**This is the strongest argument the framework is honest:** when the
pipeline tries to claim something it cannot identify (counterfactual
trajectory at sub-Stone-N), the gate refutes it. When the encoder
upgrade silently leaks information, F2 refutes it.

---

## §2 Critical evaluation — what's strong, what's weak

### §2.1 Strong

1. **Calibrated 90% PFS coverage with v11 architectural features** —
   `r-2026-05-03-v11s5-conformal`. Pure-v11 contribution: 4 Waddington
   features added on top of 11 v10 features tighten median interval
   width by 8 days at 0.898 marginal coverage. F_S5 strict pass.
   **This claim is independent of mmSYGNAL.** The marginal width move
   is small (0.5%) but signed in the design direction across 4/5
   strata. This is the v11 architectural deliverable.

2. **F6 STRICT PASS preserved through architectural generalization** —
   `r-2026-05-03-v11s3-gat-f6`. Multi-α attention propagation
   (H=4, α∈{0.3,0.5,0.7,0.9}) reduces to v10 RWR (H=1, α=0.7) with
   max-abs deviation = 0.0. Mean driver rank improves 64.9 positions.
   30.6σ vs degree-matched null. This is a meaningful methodological
   contribution: the framework was designed to allow architectural
   substitution while preserving falsification verdicts, and this run
   confirms it works.

3. **F4 framing correction + M1 Lyapunov NEW gate pass** —
   `r-2026-05-03-v11s1-ode-diag`. The Sprint 1 verdict caught that
   F4 (Helmholtz curl fraction) is identically zero by the
   ∇×∇U=0 identity for any scalar potential — F4 was therefore
   trivially passing at v10. v11 introduced M1 Lyapunov (potential
   monotone-decreasing along the gradient flow) as the non-trivial
   Helmholtz-Hodge dynamical content. **0/116 violations across 4 T
   values**, and the v10 single-Euler operator was diagnosed as 2.72
   L2 off the true ODE flow at η=2 — a non-trivial honest correction.

4. **TIE with mmSYGNAL on PFS discrimination, paired-test confirmed** —
   `r-2026-05-03-v11s5-paired-cindex-test`. The v11.5 ensemble Cox
   marginal C=0.6955 vs unbiased 6-submodel mmSYGNAL C=0.6957;
   z-test p=0.994; B=10000 paired bootstrap p=0.989 (marginal),
   p=0.979 (stratified). 99% CI [−0.035, +0.036] rules out either
   direction's advantage > ±3.6 C-index points. **The TIE is robust
   to all three caveats raised against the original v11.5 result.**

5. **Falsification framework is working as designed.** F2 zero-trust
   caught the Geneformer encoder leak (Δ=8.36, CI [2.34, 15.18]
   excludes 0 at v11 where v10 REFUTED with CI containing 0). F8-DPS
   refutation is robust to operator (v10 single-Euler AND v11
   Neural-ODE), confirming the refutation is identifiability-bound.

6. **Bounded-claims rule held.** Pearl-tier ceiling is L1 +
   structural-prior across all sprints. No language anywhere in the
   verdict docs promotes to L2 (interventional / counterfactual). This
   is the principal scientific-integrity contribution.

### §2.2 Weak

1. **Pure-v11 discrimination is at the floor (0.654), not SOTA.** The
   v11.5 SOTA-tying claim requires the mmSYGNAL routed score as an
   input feature. v11_richer alone (Cox PH on 23 v11 features
   including the 4 Waddington landscape features) has marginal
   C=0.6538 — Δ=−0.04 below mmSYGNAL. **~95% of the gain from
   v11_richer to v11.5 is from mmSYGNAL features, not v11
   architecture.** v11 features add only +0.010 incremental on top of
   raw 6 mmSYGNAL submodel scores. The Caveat #2 decomposition makes
   this explicit.

2. **F8-DPS still STRICT FAIL.** The single-snapshot
   identifiability gate cannot pass at N_paired=29, d=64
   (Stone minimax bound requires N_paired > ~150 for d=64 at the
   chosen smoothness class). The pipeline correctly predicted this
   ahead of running and the empirical refutation matches. **A
   per-patient L2 forecasting claim cannot be made on this dataset.**
   This is structural, not fixable in v12 without longitudinal data.

3. **del(1p36) and FGFR3+ proxies are V11-derived, not official MMRF
   FISH calls.** The §3.9 closure of Caveat #3 used (a) chr1 1–30 Mb
   weighted-mean Segment_Mean for del(1p36) at bottom-decile cutoff
   and (b) baseline FGFR3 z-score top-decile for FGFR3+. Both are
   reasonable proxies (literature-prevalence-aligned) but not
   ground-truth. A reviewer with MMRF biomarker portal access could
   tighten this; we explicitly flag it in the verdict §3.9.6.

4. **t(11;14) discrimination remains at chance (S4 C=0.644 in v11.5,
   0.643 best of any variant).** None of MSE-Ridge, MSE-GBM,
   Cox-richer, Cox+mmSYGNAL reaches above-chance discrimination there
   in our hands. t(11;14) MM has CCND1 over-expression and BCL2
   dependence that none of our features capture. v12 work item.

5. **Encoder upgrade rejected; v10 PCA(64) is not state-of-the-art.**
   scFoundation packaging blocked (Tencent OneDrive weights, Python
   3.7 dep, pickle-only sibling); Geneformer leaked F2; perturblab
   requires unpublished package. The Pearl-ceiling-preserving forward
   path requires either (a) ComBat-corrected Geneformer inputs, (b)
   attention-pooled adapter pretrained only on masked-gene
   reconstruction, or (c) waiting for HF-native scFoundation weights.

6. **Outstanding pipeline gaps:**
   - HBayes posterior-predictive checks (#67) — not yet executed.
   - TCGA-LAML cross-disease replication (#69) — not executed.
   - MCP connector stubs (#63) still return empty arrays in
     `resistancemap/mcp_integration/connectors.py`. The live MCP
     servers are wired only via the agent system, not the runtime
     pipeline.
   - v11 sprint orchestrator + main.py wiring (#71) not yet integrated;
     each v11 sprint is currently a standalone script.

7. **Data-integration mechanism not formally validated against MOFA+
   on the specific MMRF cohort.** The data-integration-validator agent
   has produced `data_integration_audit.md` but a head-to-head
   benchmark with MOFA+ on the same N=787 PFS endpoint is not
   completed for v11.

### §2.3 Things explicitly NOT claimed (correctly)

- L2 (interventional / counterfactual) anywhere
- Per-patient PFS forecasting
- Drug-resistance trajectory prediction at the single-cell level
- Single-snapshot identifiability of the latent flow at N_paired=29
- Mechanistic causal interpretation of v11 Waddington features
- That mmSYGNAL is "made obsolete" by v11 (it is not — it is the SOTA
  reference being matched on the same cohort)

---

## §3 Paper-readiness scope

### §3.1 Two valid paper claims, separately defensible

**Claim A (pure-v11, no mmSYGNAL dependence):**
*"On MMRF CoMMpass IA22 (N=787, 5-stratum cytogenetic Mondrian),
ResistanceMap v11 achieves calibrated 90% PFS prediction-interval
coverage with all 5 strata within ±3% of nominal (F_S5 STRICT PASS)
using a Ridge base learner augmented with 4 Waddington-landscape
features derived from a deep-score-matched scalar potential
(U(z₀), U(z(T)), ‖∇U(z₀)‖, ‖z(T)−z₀‖). Marginal interval width is
tightened by 8 days vs the v10 baseline at the same coverage."*

**Defensible at:** Nature Methods (methodology), JCO Clinical Cancer
Informatics, ICLR (calibrated-conformal-prediction track), bioRxiv.

**Claim B (v11.5 ensemble, with mmSYGNAL features):**
*"Cox-PH ensemble combining 23 v11 features and the Murie/Baliga 2025
mmSYGNAL routed risk score ties the SOTA mmSYGNAL benchmark on PFS
discrimination on the same N=787 MMRF cohort (marginal C=0.6955 vs
0.6957; paired z-test p=0.994; B=10000 paired bootstrap p=0.989;
99% CI rules out either-direction advantage > ±3.6 C-index points).
v11 architectural features add +0.010 incremental discrimination on
top of raw mmSYGNAL submodel scores, while preserving the F_S5
calibrated coverage (Claim A)."*

**Defensible at:** the same venues *only if* the manuscript foregrounds
the decomposition (95% from mmSYGNAL, ~5% from v11). Otherwise it is
a misleading SOTA claim.

### §3.2 What freeze decision the user should make

| Venue | Claim A only | Claim A + B (with honest decomposition) | Claim B alone |
|---|---|---|---|
| Nature Methods (methodology lead) | YES | YES | NO |
| ICML / ICLR (ML lead) | YES | YES (with Caveat #2 in §1) | NO |
| JCO CCI (clinical-informatics lead) | borderline (need TCGA-LAML or external cohort) | YES | NO |
| Cancer Cell / Nat Cancer (clinical-impact lead) | NO (population-level only, no L2) | NO without prospective validation | NO |
| bioRxiv (preprint, no peer review) | YES today | YES today | YES with caveat |

**Recommended freeze path:** Claim A as the lead, Claim B as the
"discrimination addendum." This protects the paper from the reviewer
who would correctly point out that the SOTA-tying claim is mostly
mmSYGNAL.

---

## §4 What needs to happen for v12

Ranked by paper-impact / effort:

1. **Independent transcriptional-program inference on a held-out
   cohort.** miner3 or DESeq2-derived program scoring on a non-MMRF
   cohort, then routed via the Murie/Baliga A>B>C scheme. Removes
   the v11.5 dependence on mmSYGNAL pre-fit weights.
2. **External cohort PFS replication.** The MMRC IA12 cohort, or
   another MM RNA-seq + outcome dataset (e.g., HOVON-65, EMC-92).
   Tests claims A and B out-of-distribution.
3. **TCGA-LAML cross-disease replication of F3 + F_S5 (#69).**
   Already pre-registered. F3 is biologically expected to FAIL on AML
   (proteasome is MM-specific); F_S5 should generalize as a
   methodology contribution.
4. **HBayes PPC (#67).** Posterior-predictive check on the v10 HBayes
   coefficients — confirms the Bayesian module is calibrated.
5. **Per-stratum t(11;14) specialist head.** BCL2/MCL1/CCND1
   features. Closes the only stratum that's at chance.
6. **Encoder retry: ComBat-corrected Geneformer or HF-native
   scFoundation when available.** F2-zero-trust gating preserved.
7. **MCP connector wiring (#63), pipeline orchestrator (#71).**
   Engineering hygiene; not paper-blocking.

---

## §5 Verdict

The v11/v11.5 pipeline has **two paper-defensible claims** at L1 +
structural-prior. Claim A (calibrated coverage with Waddington
features) is the principled methodology contribution. Claim B
(SOTA-tying ensemble) is honest only with the Caveat #2 decomposition
foregrounded.

**Five strict-pass gates (F4-corrected, M1, F6, F_S5, F3-MM)**, **two
strict-fail gates that were pre-registered to fail** (F8-DPS at
sub-Stone-N, F10-NIE-tipping at single-DAG). The falsification
framework working as designed is itself a contribution.

**The pipeline is paper-ready for ICML/ICLR/Nature Methods at Claim A,
and for the same venues at Claim A+B with honest decomposition.** It is
NOT ready for Cancer Cell or any venue that requires L2 evidence /
prospective use. v12 work is well-scoped and pre-registered.

The agent-trio output (architecture-chair, causal-auditor,
clinical-translator) running in parallel will sharpen this verdict;
this document is the parent-session reference for that integration.
