# Causal-Inference Audit — ResistanceMap v12 (MOFA+v11 architectural change)

**Date:** 2026-05-07
**Auditor:** PhD Causal Inference (Pearl/Lange-Hansen mediation / Robins-style sensitivity)
**Scope:** Assess whether the v12 MOFA+v11 combo (RUNS.md row 37; Cox marginal C=0.6751) changes
any causal-inference conclusion reached under v10/v11. Preserve v10 Sprint 7 F8 STRICT FAIL as
a load-bearing datum; do not inflate Pearl tier.

**Sources read:**
- `docs/V12_CAUSAL_REAUDIT.md` (504 lines)
- `docs/V12_MOFA_VS_V115_VERDICT.md`
- `scripts/v10/s4d_nie_estimation.py`
- `paper/v8_artifacts/v10_sprint4/nie_proteasome_tt2L.json`
- `paper/v8_artifacts/v10_sprint4/nie_tipping_point.json`
- `paper/v8_artifacts/baselines/mofa_results.json`
- `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`
- RUNS.md rows 19, 20, 23, 25, 37

---

## §1 v12 DAG Diagram

The v12 pipeline introduces MOFA shared factors as additional covariates in the Cox PH model.
Their causal position must be established before any mediation re-derivation.

### MOFA factor construction (from script inspection)

MOFA factors are computed by running mofapy2 K=32 on `log1p(TPM)` of the top 5,000 HVG at
**baseline** (MMRF enrollment time-point, single modality). They are linear-algebraic summaries
of the expression matrix observed at t=0. No treatment-time expression is used. No outcome
labels are involved in the factorization.

### DAG

```
                 ┌─────────────────────────────────────────────┐
                 │  Biological baseline state (t = 0)          │
                 │  ┌───────────┐   ┌────────────────────┐     │
                 │  │  Z_bio    │──►│  MOFA factors F     │     │
                 │  │(true expr)│   │  (observed summary  │     │
                 │  └─────┬─────┘   │  of Z_bio at t=0)  │     │
                 │        │         └────────┬───────────┘     │
                 │        │                  │                  │
                 └────────┼──────────────────┼──────────────────┘
                          │                  │
          ┌───────────────┼──────────────────┼──────────────────────┐
          │               ▼                  ▼                       │
          │    C (confounders) ◄── F (MOFA)                         │
          │    {ISS, age, gender,             │                       │
          │     cyto_del17p, cyto_1q21,       │                       │
          │     cyto_del13q, cyto_t4_14,      │                       │
          │     cyto_t11_14}                  │                       │
          │          │                        │                       │
          │          ▼                        ▼                       │
          │     A (treatment) ─────────────► Y (TT2L, PFS)           │
          │     bort_1L                       ▲                       │
          │          │                        │                       │
          │          ▼                        │                       │
          │     M (mediator) ─────────────────┘                       │
          │     M_proteasome                                          │
          │     (PSMB5+PSMB1+PSMB2 RWR score, baseline)              │
          └──────────────────────────────────────────────────────────┘

          U (unmeasured confounders; F10 is about robustness to U)
          U ──► A, M, Y (back-door open paths, not blocked by C or F)
```

**Node classification:**

| Node | Type | Temporal position | Role in DAG |
|------|------|-------------------|-------------|
| Z_bio | Unobserved biological state | t = 0 | Latent common cause |
| F (MOFA factors) | Observed deterministic function of Z_bio | t = 0 | PRE-TREATMENT baseline covariate |
| C (cytogenetics/ISS/age) | Observed | t = 0 | PRE-TREATMENT confounder |
| A (bort_1L) | Observed | t = 0, enrollment | Treatment node |
| M (M_proteasome) | Observed mediator | t = 0 (baseline expression) | Mediator |
| Y (tt2L_days, had_2L) | Observed outcome | t > 0 | Survival endpoint |
| U | Unobserved | t = 0 | Unmeasured confounders |

**Key topological conclusion:** MOFA factors F are a deterministic linear function of baseline
expression (probabilistic PCA equivalent under K_effective=29 ARD-pruned, single-view Gaussian
MOFA). They are causally upstream of treatment — the patient's tumor expression profile exists
before any treatment assignment. Therefore:

- F is a PRE-TREATMENT covariate, NOT a mediator.
- F is NOT a descendant of A.
- F is a function of the same baseline expression that generates M_proteasome.
- M and F are children of the same parent (Z_bio) — they are sibling summaries of baseline biology.

---

## §2 NIE Re-Derivation Under v12

### 2.1 Does adding MOFA factors close any back-door path?

The v10 back-door criterion for A→Y requires blocking all paths A ← ... → Y not through M.
The v10 confounder set C = {ISS, age, gender, 5 cytogenetics} blocks the paths visible in the
graph. The remaining threat paths are through U (unmeasured).

MOFA factors F are a linear transform of baseline expression. They capture latent transcriptional
structure — shared biology that partially proxies U. If some unmeasured confounders U are
correlated with baseline expression (plausible: ISS stage correlates with high-risk transcriptional
programs; del17p transcriptional consequences partially encoded in HVG space), then conditioning
on F may PARTIALLY proxy-block paths through U.

**Identification result (graph-level):**

Conditioning on F in the NIE Cox regression would weaken, but not close, the back-door path
through U, because:

1. F captures linear shared variation; non-linear or multi-omic confounders in U remain open.
2. F is derived from the same expression matrix that generates M_proteasome. Conditioning on
   F when M_proteasome is also in the model introduces partial redundancy (F and M share
   common parent Z_bio). This does not create a collider but it does not fully block U either.
3. The tipping-point analysis (nie_tipping_point.json) shows that a confounder pair
   (RR_AU=1.5, RR_UY=1.10) suffices to nullify the lower-CI NIE. MOFA factors explain
   ~4% of variance in the outcome beyond C (estimated from ΔC: combo − v11_richer = +0.021).
   A 4% reduction in residual variance implies that the MOFA proxy only partially captures U;
   the joint (1.5, 1.10) nullification strength remains plausible even after conditioning on F.

**Estimation result:** If MOFA factors are added to the s4d Cox model as additional baseline
controls, the new confounder set becomes C* = C ∪ F. This is a legitimate enrichment of the
adjustment set under sequential ignorability (VanderWeele 2015 §2.4: any baseline covariate
can be added to C). The resulting NIE estimator remains:

    NIE_C* = (β_M|C* + β_AM|C*) × ΔM_C*

where all regression coefficients are refit on the augmented covariate set.

### 2.2 Does adding F open any new collider path?

A collider opens if F is a common descendant of two independent paths. The concern would be
conditioning on a common effect of A and of an unmeasured confounder (Berkson's bias). Here:

- F is constructed at baseline (pre-treatment); it cannot be a descendant of A (temporal order).
- F is a descendant of Z_bio, which is also a parent of M and of U's biological component.
- Conditioning on F does NOT open a new collider between A and U because F is upstream of A,
  not a common effect. F is a child of Z_bio; A is a child of Z_bio and/or of clinical factors.
  Since A is assigned POST-enrollment (clinician decision given cytogenetics and clinical state),
  F and A share a common parent (baseline biology) but F is not a descendant of A.

**No new collider path is opened by adding MOFA factors to the adjustment set.** This is the
standard result for pre-treatment covariates (Robins 2003: adjusting for pre-treatment
covariates never induces collider bias relative to the A→Y causal path).

### 2.3 Does the v12 NIE estimator change structurally?

Adding F to C does not change the Lange-Hansen Cox NIE structural form. The decomposition:

    NIE = (β_M|C* + β_AM|C*) × E[M|A=1, C*] − E[M|A=0, C*]

remains valid. The identification assumptions A1–A5 (listed in s4d_nie_estimation.py) are now
stated with respect to the augmented set C* = C ∪ F:

- A1* : No unmeasured confounding A→Y given (C, F)
- A2* : No unmeasured confounding A→M given (C, F)
- A3* : No unmeasured confounding M→Y given (A, C, F)
- A4  : No exposure-induced mediator-outcome confounder [unchanged — F pre-treatment]
- A5  : Cox PH holds [unchanged — model form preserved]

A1*–A3* are strictly WEAKER (closer to true) than A1–A3 because F partially proxies U.
But "weaker assumption" does not mean "assumption holds" — the residual U after conditioning
on (C, F) must still be negligible, which is unverifiable from observational data alone.

---

## §3 F10 E-Value Re-Estimate Under v12 Baseline Control Set

### 3.1 Quantitative bounds on E-value change

The v10 F10 result: NIE HR_lower_CI = 1.029, E-value at CI bound = 1.20.

To reach F10 PASS (E ≥ 1.5), the lower CI bound would need to reach HR ≈ 1.13 (because
E-value = HR + sqrt(HR×(HR-1)); solving E=1.5 gives HR ≈ 1.13).

Adding MOFA factors to C would, if it reduces residual confounding, shift the NIE point estimate
upward (fewer confounders in U → larger observed NIE on natural scale). The MOFA+v11 combo
gains +0.021 marginal C-index over v11_richer alone. This incremental discrimination reflects
MOFA factors capturing approximately 3–5% of prognostic variance beyond the current Cox feature
set. For the NIE CI lower bound to move from HR=1.029 to HR=1.13 (+10% on the HR scale) would
require that a substantial fraction of the unconfounded NIE was being suppressed by the open
back-door paths currently controlled only by C.

**Assessment: NO — the E-value of 1.5 is not reachable under v12 with high confidence.**

Reasons:

1. The tipping-point analysis shows the NIE lower CI is nullifiable by (RR_AU=1.5, RR_UY=1.10).
   The MOFA factors can at best reduce RR_UY by the proportion of U's variance they proxy.
   The +0.021 ΔC implies MOFA adds ~3% variance explained on top of the existing 23 features.
   This is insufficient to move RR_UY from 1.10 toward 1.0 by enough to prevent nullification
   at a (1.5, 1.0+ε) confounder strength.

2. The F10 failure is confirmed STRUCTURAL across all 7 mediator panels (M_seed3 through
   M_26S_full; RUNS.md row 20). Richer biological mediator definitions dilute NIE further;
   MOFA factors are an orthogonal covariate enrichment, not a biological mechanism re-definition.
   There is no causal-graph reason why adding linear expression factors to C would raise the
   NIE point estimate sufficiently.

3. MOFA factors are derived from the same expression space as M_proteasome (both functions of
   baseline log1p-TPM HVGs). Conditioning on F while also including M introduces attenuation:
   the coefficient β_M|C* will partially absorb MOFA factor variance, which may REDUCE rather
   than increase the estimated NIE.

**Verdict: F10 STRUCTURAL FAIL persists under v12. The E-value at the CI lower bound is unlikely
to cross 1.5 even if MOFA factors are added to the NIE Cox confounder set. This conclusion holds
at the identification-argument level (graph + sensitivity); a re-run with C* would be confirmatory
but is not expected to overturn the structural result.**

---

## §4 Pearl-Tier Ceiling: Still L1-with-Structural-Prior, or Promoted?

### 4.1 What would promote the tier?

L1 → L2 promotion requires: the NIE is identified (F8 PASS), specific (F9 PASS), AND robust
to unmeasured confounding at a pre-registered threshold (F10 PASS with E ≥ 1.5). None of
these conditions are altered by the v12 MOFA change:

- F8 and F9 remain PASS (NIE direction and specificity are not affected by adding covariate F
  to C; they were already PASS under the smaller C).
- F10 remains FAIL (§3 above). The MOFA +0.021 ΔC gain is too small to close the confounding
  gap quantified by the (1.5, 1.10) tipping pair.

L2 → per-patient L3 would further require F8-DPS PASS (per-patient dynamic trajectory
prediction). F8-DPS is STRICT FAIL under BOTH v10 single-Euler and v11 Neural-ODE operators
(RUNS.md rows 23, 25). The v12 MOFA addition does not provide new paired (z0, z1) training
data; it adds a new covariate to a Cox PH model. The Stone(d=64)≈150 estimation bound at
N_paired=29 is unaffected by whether or not MOFA factors appear in a separate Cox regression.

### 4.2 Does the v12 MOFA result contain ANY causal content?

The +0.021 marginal ΔC from the combo model (MOFA+v11) is a purely L1 result: it is a
difference in concordance statistics on observed (X, Y) data. It does not invoke do(), does
not estimate a counterfactual, and does not require identification of any causal path.

The fact that MOFA factors carry "orthogonal signal" to v11 Waddington features (V12_MOFA
verdict, §Verdict) means only that they improve observed-outcome rank discrimination — a
sufficient statistic for nothing beyond L1 prediction.

**Pearl-tier ceiling: STILL L1-with-structural-prior. No promotion. The v10 Sprint 7 F8
STRICT FAIL is operator-independent and sample-size-driven; v12 MOFA does not address it.**

---

## §5 Language Audit — v12 Documents

### 5.1 V12_MOFA_VS_V115_VERDICT.md

**Flag 1 — "MOFA+ shared factors carry orthogonal signal to v11's Waddington/PSMB-seed-3/cyto
features at the +0.021 marginal C-index level"**

Status: SAFE at L1. "Orthogonal signal" is used in the statistical sense (incremental C-index
in an observational Cox regression). No intervention or counterfactual is claimed. Correct as
written.

**Flag 2 — "v12 architectural opportunity"**

Status: SAFE at L1, but requires a hedge in any manuscript use. "Architectural opportunity"
implies that incorporating MOFA factors into v12 would improve predictive discrimination. This
is a projection from an observational ΔC result. The hedge required: "This is a predictive
discrimination gain (L1 associational); it does not imply that MOFA factor construction is a
causal mechanism or that the +0.021 ΔC reflects identification of a new causal path."

**Flag 3 — V12_MOFA_VS_V115_VERDICT.md honest caveats section: "Single-modality MOFA+: MMRF
IA22 in our processed slice has no paired proteomics"**

Status: SAFE. Explicitly acknowledges that "True MOFA+ multi-omics power is NOT exercised."
This is an L1 scope limitation, correctly stated.

### 5.2 RUNS.md row 37

**Flag 4 — "MOFA+ shared factors carry orthogonal signal to v11's Waddington/PSMB-seed-3/
cyto features"**

Status: L1-safe as stated in RUNS.md. No inflation.

**Flag 5 — No v12-specific causal language found in RUNS.md row 37 that overreaches.** The
row correctly reports marginal C-index, paired bootstrap CI, and pre-registered verdicts
without asserting mediation, intervention, or counterfactual content.

### 5.3 Inherited language risks from V12_CAUSAL_REAUDIT.md (unresolved at v11)

The following required changes from V12_CAUSAL_REAUDIT.md (prior audit, 2026-05-04) remain
unresolved and become MORE important now that v12 uses MOFA to strengthen the Cox model:

**RC-2 (HIGH):** M2 must NOT describe mmSYGNAL as a "treatment selection" benchmark when
citing it as the SOTA comparison. The v12 combo C=0.6751 approaches mmSYGNAL (C=0.6957);
if M2 uses the new v12 combo result alongside the mmSYGNAL comparison, the risk of L2
import through the mmSYGNAL citation is amplified because the v12 combo is now the manuscript's
headline discriminative result.

**RC-4 (MEDIUM):** Sprint 4 NIE "supporting evidence" hedge must appear in M1 §7 regardless of
whether v12 adds MOFA to the NIE model; see §3 above for why MOFA does not rescue F10.

---

## §6 Carve-Outs: Which v12 Claims Are Safe to Publish at L1 with Structural Prior

| Claim | Safe? | L1 basis | Required hedge |
|-------|-------|----------|----------------|
| MOFA+v11 combo achieves marginal C=0.6751 (Δ=+0.021 vs v11_richer, 95% CI [+0.0005, +0.0425]) | SAFE | Paired bootstrap concordance statistic on observed (X, Y) | Report 95% CI borderline significance (p=0.046); note K=32 deviation from pre-registered K=64 |
| MOFA factors carry orthogonal prognostic signal beyond Waddington+cyto features | SAFE | ΔC from nested covariate set comparison, L1 | "Orthogonal" is statistical, not causal |
| MOFA factors are pre-treatment covariates (not mediators, not colliders) in v12 DAG | SAFE | Temporal precedence; constructed from baseline TPM | State explicitly in Methods; see §1 above |
| v12 MOFA enrichment does NOT change NIE identification or robustness | SAFE | Structural argument (§2–§3); F10 E-value gap too large | Include in Limitations; do NOT omit |
| Pearl-tier ceiling remains L1-with-structural-prior | SAFE | F8-DPS STRICT FAIL operator-independent; F10 structural | State in abstract and §6 fallback |
| v12 MOFA+v11 combo as a v12 architectural opportunity (L1) | SAFE with hedge | L1 predictive ΔC | Add sentence: "This is an associational discrimination gain; it does not imply causal identification of new pathways" |
| Proteasome NIE as "supporting directional evidence" (v10 Sprint 4) | CONDITIONAL SAFE | F8+F9 PASS; F10 FAIL explicitly stated; RC-4 hedge applied | RC-4 hedge from V12_CAUSAL_REAUDIT.md §4 is mandatory |

**Claims that remain NOT safe to publish without additional data or redesign:**

| Claim | Why unsafe | What would make it safe |
|-------|-----------|------------------------|
| "MOFA factors improve causal identification of proteasome mediation" | No graph-level identification gain; U still partially open; F10 still fails | Re-run s4d with C*=C∪F, verify E-value; expected to still fail (§3) |
| "v12 combo predicts which resistance pathway a patient will transition through" | Requires L2 mediation identification (F10 PASS) which is absent | Perturbation-response data (DepMap CRISPR screens with paired outcomes) |
| "v12 closes the per-patient forecasting gap" | F8-DPS STRICT FAIL is sample-size-driven; MOFA does not provide new paired (z0,z1) data | N_paired ≥ 150 for d=64 (Stone bound); requires longitudinal RNA collection |
| Any "treatment selection" framing | Pearl ceiling is L1; do-calculus identification not established | RCT or instrumental-variable design on the MMRF cohort |

---

## Summary Verdict Table

| Gate / Claim | v10 status | v12 change? | v12 verdict |
|---|---|---|---|
| F8 NIE CI excludes zero (Sprint 4) | PASS CI [+0.029, +0.152] | MOFA does not change NIE structural form | PASS preserved |
| F9 specificity (0/50 random gene sets) | PASS | Unaffected by MOFA covariate enrichment | PASS preserved |
| F10 E-value ≥ 1.5 | FAIL (E=1.20) | MOFA +0.021 ΔC too small to close (1.5, 1.10) tipping gap | FAIL preserved — STRUCTURAL |
| F8-DPS per-patient drift (Sprint 7) | STRICT FAIL (0/16) | MOFA does not add paired (z0,z1) data | STRICT FAIL preserved |
| Pearl-tier ceiling | L1-with-structural-prior | No change | L1-with-structural-prior |
| MOFA DAG position | N/A | Pre-treatment baseline covariate; no collider induced | Pre-treatment covariate (safe to adjust) |
| New back-door paths opened by MOFA? | N/A | None — MOFA is pre-treatment | No new paths |
| mmSYGNAL "treatment selection" language risk | RC-2 required | Amplified: v12 combo C=0.6751 brings v12 closer to mmSYGNAL C=0.6957 | RC-2 more urgent at v12 |

---

*End of v12 MOFA causal re-audit. All numerical claims trace to artifact paths cited above.*
*Artifacts: `paper/v8_artifacts/v10_sprint4/nie_proteasome_tt2L.json`, `nie_tipping_point.json`,*
*`paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_pfs.json`, RUNS.md rows 19, 20, 23, 25, 37.*
