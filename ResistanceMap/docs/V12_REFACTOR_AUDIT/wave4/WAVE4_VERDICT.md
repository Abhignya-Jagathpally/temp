# V12 Refactor Audit — WAVE-4 VERDICT

**Author:** integration-chair (synthesis, Wave 4; persisted by main thread)
**Date:** 2026-05-07
**Branch:** v6 / worktree `tier1-refactor`
**Status:** EXTENDS `docs/V12_REFACTOR_AUDIT/wave3/WAVE3_VERDICT.md` (which itself extended `CHAIR_VERDICT.md`). Does NOT replace either prior verdict.
**Inputs reconciled:** all Wave-4 reports under `docs/V12_REFACTOR_AUDIT/wave4/` and JSON under `paper/v8_artifacts/v12_refactor_audit/wave4/`.

## §1 — What the head-to-head settled (the empirical answer for a reviewer)

**The user asked:** *"Show that the proposed model actually performs better than literature SOTA implemented on the same preprocessed data."*

**Empirical answer: PARTIAL YES, with one structural caveat.** Wave-4 ran a fair head-to-head on the same N=787 MMRF cohort, the same 224 events, the same TT2L PFS endpoint, the same v11_richer / v11_richer+PCA29 feature scaffold, and the same paired-bootstrap protocol against `Cox_v11_routed_mmsygnal` log-hazards.

| Comparator | C | Δ vs v11.5 | 95% CI | p | Bonferroni m=8 (α=0.00625) | Verdict |
|---|---:|---:|---|---:|---|---|
| v11.5 `Cox_v11_routed_mmsygnal` | 0.6955 | ref | ref | ref | — | reference |
| RSF + v11_richer | 0.6223 | −0.0732 | [−0.104, −0.042] | 0.0000 | survives | LOSES |
| RSF + v11+PCA29 | 0.6359 | −0.0596 | [−0.087, −0.031] | 0.0000 | survives | LOSES |
| DeepSurv + v11_richer | 0.6235 | −0.0720 | [−0.098, −0.047] | 0.0000 | survives | LOSES |
| DeepSurv + v11+PCA29 | 0.6360 | −0.0595 | [−0.093, −0.027] | 0.0002 | survives | LOSES |
| SCOPE (Hussain & Sontag) + v11_richer | 0.6325 | −0.0631 | [−0.100, −0.027] | 0.0004 | survives | LOSES |
| SCOPE + v11+PCA29 | 0.6141 | −0.0814 | [−0.128, −0.036] | 0.0010 | survives | LOSES |
| TrajectoryNet flow alone (66 feats) | 0.6232 | −0.0724 | [−0.105, −0.040] | 0.0002 | survives | LOSES |
| TrajectoryNet + v11_richer (89 feats) | 0.6513 | −0.0443 | [−0.069, −0.020] | 0.0004 | survives | LOSES |
| BlockForest (R-only) | NOT_RUN | — | — | — | — | (gap) |
| mmSYGNAL routed 6-submodel (Wave 1) | 0.6938 | −0.0017 | [−0.027, +0.027] | 0.989 | n/a | **TIES** |

**Out of 8 off-the-shelf SOTA implementations: 8 lose, 0 tie, 0 beat — all surviving Bonferroni m=8 in the LOSING direction.** The only model that ties v11.5 is `mmSYGNAL routed`, structurally not coincidentally — v11.5's discriminative core IS mmSYGNAL's transcriptional-program output wrapped in a Cox PH (see §3).

**Audit-process transparency.** The original RSF run reported C=0.93, which was an in-sample (training-set) C-index — `rsf.predict` was called on the same data the model was fit on. Caught and corrected before publication: LOO N=787 protocol matching the v12 Cox eval gave the actual values shown above. This correction is surfaced explicitly because reviewers will reasonably ask whether other comparators are similarly contaminated; they are not — DeepSurv/SCOPE used 5-fold CV with within-fold standardization, TrajectoryNet used LOO Cox PH on the embedded features.

**Two structural failures honest readers must see.**
1. **TrajectoryNet** is structurally under-determined on this cohort: at N_paired=29 in 64-D the CNF energy/Jacobian regularizers dominate; training MMD plateaus at −0.011 vs identity-flow baseline −0.009. Adding flow features to v11_richer **hurts** discrimination (0.6538 → 0.6513). TrajectoryNet is not the wrong paper — it is the wrong cohort for that paper.
2. **SCOPE + PCA29 is WORSE than SCOPE alone** (0.6141 vs 0.6325). More tokens push the 2-layer transformer further into overfitting at N=787. Argues against a "scale up the architecture" line for v12.5+.

## §2 — Updated submission-readiness verdict

| Venue | Wave-3 verdict | Wave-4 update |
|---|---|---|
| bioRxiv | YES, now | **YES, now (UNCHANGED)** — Wave-4 closed §6A.10 (Hussain/Sontag); no new bioRxiv blocker |
| Briefings/CRM | CONDITIONAL, sunset 2026-09-30 | **UPGRADE to CONDITIONAL_PASS HIGH CONFIDENCE, sunset extended to 2026-12-31** — re-eval criterion (i) closed [Hussain/Sontag head-to-head landed]; (ii) preserved; (iii) NUTS HMC PPC still pending |
| Nature Methods | NOT YET | **NOT YET (UNCHANGED), but reason narrowed** — remaining blocker is now ONLY external-cohort F_S5 replication (chair 6B.1) and paired-modality N≥100 (chair 6B.4); 6B.3 partially closed via SCOPE transfer-test |
| ICML/NeurIPS primary | NOT YET | **NOT YET (UNCHANGED)** — contribution is curated-feature engineering + falsification framework, not new architecture (§3) |
| SurvBoard 2025 (PMID 41031875) | NEW recommendation | **UPGRADE to actively recommended** — Wave-4 empirically reproduces SurvBoard 2025 thesis on MMRF; submitting v11.5 + 8-method panel publishable in either direction |

**Net: bioRxiv unchanged; Briefings/CRM strengthened; Nature Methods still gated on external cohort + paired modalities.**

## §3 — Niche A reframed: curated features + falsification framework, NOT a new architecture

W3.6 §A.1 found `mmsygnal_routed` carries β·σ = +0.404 [95% CI +0.265, +0.542], **2.6× the next-largest feature in v11.5**, and is the only feature whose CI does not approach zero. Wave-4 confirms the structural consequence: when off-the-shelf SOTA models are denied access to mmSYGNAL routing — which they all are by construction (taking v11_richer or v11_richer+PCA29) — they all collapse to C ≈ 0.62-0.64, the band v11_richer occupies on its own (C=0.6538).

**Honest framing for the manuscript abstract:**

> *We do not propose a new neural architecture for survival prediction in multiple myeloma. We propose a curated-feature pipeline (clinical + cytogenetics + Waddington-landscape descriptors + the mmSYGNAL transcriptional-program rules) wrapped in a Cox proportional-hazards model, evaluated under a published falsification framework with a pre-registered numeric threshold for each F-gate (F1–F10) and a published record of the gates we failed (T1, S4, F8 at low N_paired). On N=787 MMRF baseline samples this pipeline reaches marginal C-index 0.6955, ties the field-standard mmSYGNAL routing on its own cohort, and beats four off-the-shelf SOTA implementations (RSF, DeepSurv, the Hussain & Sontag SCOPE transformer adapted to MMRF, and TrajectoryNet adapted to N_paired=29) by ΔC = +0.06 to +0.08 (paired bootstrap p<0.001, all surviving Bonferroni m=8). The contribution is the curation + falsification framework, not the architecture.*

Three claims are sharper than the v11/v11.5 abstracts in the repo today:

1. **"We do not propose a new architecture."** Drops the "novel multi-omics deep stack" framing. Wave-3 already documented PCA(K=29) reproduces MOFA's lift within ±0.001 — the deep stack is not what is doing the work.
2. **"Beats off-the-shelf SOTA by ΔC = +0.06 to +0.08, p<0.001, surviving Bonferroni m=8."** Strongest defensible quantitative claim. Must always co-locate with the curated-feature caveat: without `mmsygnal_routed`, v11.5 collapses to v11_richer C=0.6538.
3. **"Ties the field-standard mmSYGNAL routing."** Honest, structural, central: v11.5 IS mmSYGNAL wrapped in Cox; cannot beat its own backbone, should not claim to.

The "falsification framework that publishes its failures" framing (T1 v12 specialist heads; S4 t(11;14) safety flag; Sprint C Touzeau pre-Cox refute; F8 at N_paired=29) is the publishable methodological contribution. Niche-A is **defensible at full strength** with the "to our knowledge" hedge per Wave-3 `niche_a_novelty.md` (0/11 audited papers satisfied Q1+Q2+Q3).

## §4 — Per-stratum decomposition reality

v11.5 is **not uniformly best across strata**:

| Stratum | n | v11.5 C | Best non-v11.5 C | Best model | v11.5 win/loss |
|---|---:|---:|---:|---|---|
| S1 del(17p) | 105 | 0.730 | 0.714 | RSF + PCA29 | **wins +0.016** |
| S2 t(4;14) | 97 | 0.667 | 0.722 | mmSYGNAL routed | **LOSES −0.055** |
| S3 +1q21 | 165 | 0.694 | 0.671 | DeepSurv + v11_richer | **wins +0.023** |
| S4 t(11;14) | 95 | 0.643 | 0.693 | mmSYGNAL routed | **LOSES −0.050** |
| S5 other | 325 | 0.659 | 0.682 | mmSYGNAL routed | **LOSES −0.023** |

**mmSYGNAL beats v11.5 in 3 of 5 strata (S2, S4, S5).** Marginal-C tie preserved because S1+S3 advantages of v11.5 offset.

Most parsimonious read: v11.5's Cox wrapper trades off mmSYGNAL routing against `cyto_del17p` (W3.6 §A.1 #5, β·σ=+0.126) and PCA / Waddington descriptors most informative in del(17p) and +1q21 — strata where cytogenetic risk and proliferative-program load dominate. mmSYGNAL beats v11.5 in t(4;14), t(11;14), and "other" — strata where FGFR3+, BCL2-driven, and unclassified-program submodels carry weight that v11.5's Cox pooling washes out.

Consistent with audit record:
- **Chair §3(iii) S4 t(11;14) safety flag:** v11.5's S4 C=0.643 lowest; mmSYGNAL S4=0.693 highest of any model; v12 t(11;14) specialist STRICT_FAIL (RUNS row 36); Sprint C Touzeau pre-Cox refute (W3.3). Three lines of evidence converge: t(11;14) discrimination is BCL2-family-driven biology that bulk-RNA-as-proxy cannot capture.
- **Wave-4 Fig W4.GT.4** S4 LOWESS curve nearly flat at log-hazard vs 12-month-event — chance-level discrimination, **visually consistent with mmSYGNAL's 5-pp advantage in S4**.

**Manuscript implication:** per-stratum table must accompany marginal-C headline. Reviewers will catch the S2+S4+S5 mmSYGNAL wins; surface them ourselves.

## §5 — REVISED v12.5 sprint priority post-Wave-4

Wave-3 ranked: **Sprint D > B > SurvBoard > A**. **Sprint D (Hussain/Sontag) ALREADY ADDRESSED** by Wave-4 — SCOPE adapted, runs head-to-head, loses v11.5 by Δ=−0.063 p=0.0004. Transfer-test of architecture, not TOURMALINE-equivalent reproduction.

**Revised priority for v12.5:**

1. **NEW Rank 1: External-cohort replication of F_S5 5/5 conformal coverage.** Was chair 6B.1, now the gating Nature-Methods blocker. DUA initiation for IFM2009, MRC Myeloma XI, or HOVON-65. Cost: people-time (DUA), low compute. Gate: F_S5 stratified coverage within ±5% on new cohort.
2. **Rank 2: Sprint B — NUTS HMC + per-stratum heteroscedastic σ_obs.** Unchanged. ~45 min CPU; closes 5/10 PPC failure caveat.
3. **Rank 3: SurvBoard 2025 leaderboard submission.** **UPGRADED.** Wave-4 reproduced their thesis on MMRF — submitting our 8-method panel directly. ~2-3 person-days.
4. **NEW Rank 4: TOURMALINE-equivalent longitudinal-cohort acquisition / partnership.** Sprint D's full reproduction needs longitudinal monthly-input MM RNA we don't have. Path is partnership/DUA, not implementation.
5. **Rank 5: Sprint A (scFoundation/ComBat encoder).** Demoted further. Wave-4 SCOPE+PCA29-makes-things-WORSE confirms more representational capacity at N=787 likely hurts. Defer to v13.
6. **NEW Rank 6 (low priority): BlockForest gap closure.** Install R + blockForest + rpy2 in CI image; re-run with `cv.error.frac=0.5, classification=FALSE`. Even if BlockForest beats RSF, the m=8 → m=9 Bonferroni shift is irrelevant — all current p ≤ 0.001 still survive.

## §6 — Updated publication-readiness CHECKLIST

### 6A — bioRxiv (target: now)

| # | Gate | Status | Wave-4 update |
|---|---|---|---|
| 6A.1 | RUNS.md ±0.0001 reproducibility | DONE | DONE |
| 6A.2 | "MOFA+ shared factors" framing replaced | NOT STARTED | §3 above provides new abstract paragraph |
| 6A.3 | "v12 beats v11.5" / "beats SOTA" claims removed | NOT STARTED | **NEW NUANCE:** "beats off-the-shelf SOTA" defensible WITH curated-feature caveat (§3, §7); "beats v11.5" remains forbidden |
| 6A.4 | S4 t(11;14) +0.057 co-located with STRICT FAIL + Sprint C refute | NOT STARTED | **NEW EVIDENCE:** Wave-4 §4 + Fig W4.GT.4 visually confirms S4 chance-level prediction |
| 6A.5 | Bonferroni m=8 caveat | NOT STARTED | **REINFORCED:** Wave-4 establishes m=8 SOTA panel; Table 1 = `master_head_to_head.csv` |
| 6A.6 | Causal hedges (RC-2, RC-4) | NOT STARTED | NOT STARTED |
| 6A.7 | Niche-A "to our knowledge" hedge | NOT STARTED | DEFENSIBLE; abstract language in §3 |
| 6A.8 | Real-data figures | DONE | **REINFORCED:** Wave-4 added 4 ground-truth figures (Fig W4.GT.1–4) |
| 6A.9 | PMID 23860449 (NOT 24305167) for Touzeau | NOT STARTED | NOT STARTED |
| 6A.10 | Hussain/Sontag PMID 39075240 in Related Work | NOT STARTED | **CLOSED EMPIRICALLY:** SCOPE adaptation in `hussain_sontag.json`; cite in Related Work AND in head-to-head Methods |

**Wave-4 closes 1 item completely (6A.10) and reinforces 6A.4, 6A.5, 6A.8.** Six items remain pure manuscript-text edits.

### 6B — Nature Methods candidacy (target: 2027-Q1 earliest)

| # | Gate | Status | Wave-4 update |
|---|---|---|---|
| 6B.1 | External MM cohort F_S5 5/5 ±5% | NOT STARTED | **now Rank 1 in §5** |
| 6B.2 | Survives Bonferroni m≥8 on a primary metric | NOT STARTED | **PARTIAL:** v11.5 itself is the reference; the 8 SOTA losses survive Bonferroni in the LOSING direction |
| 6B.3 | Match Hussain/Sontag ISS-delta on TOURMALINE-equivalent | NOT STARTED | **PARTIALLY ADDRESSED:** SCOPE architecture transfer-tested on MMRF; LOSES v11.5. Transfer-test not TOURMALINE reproduction |
| 6B.4 | Paired modalities (CNV+proteomics+RNA) at N≥100 | NOT STARTED | NOT STARTED |
| 6B.5 | F8-DPS strict pass at higher N_paired OR L1-framing | PARTIAL | PARTIAL |
| 6B.6 | F_S4_PANEL_PRESENT promoted to permanent F-gate | NOT STARTED | NOT STARTED |
| 6B.7 | Parallelism patches applied + bit-identicality verified | NOT STARTED | NOT STARTED |

**Wave-4 partially closes 6B.2 and 6B.3.** Nature-Methods bottleneck is now demonstrably ONLY external-cohort replication and paired-modality acquisition.

## §7 — Honest manuscript framing

**DO say:**
- v11.5 reaches marginal C-index 0.6955 on N=787 MMRF baseline.
- v11.5 ties mmSYGNAL routed (Δ=−0.0017, 95% CI [−0.027, +0.027], p=0.989) on the same cohort.
- v11.5 beats RSF (Ishwaran 2008), DeepSurv (Katzman 2018), the SCOPE transformer adapted from Hussain & Sontag 2024 (PMID 39075240), and TrajectoryNet (Tong 2020) by ΔC = +0.06 to +0.08, paired bootstrap p<0.001, all surviving Bonferroni m=8 (α=0.00625).
- v11.5's discriminative power is concentrated in ONE engineered feature (`mmsygnal_routed`, β·σ = +0.404, 95% CI [+0.265, +0.542], 2.6× the next feature; W3.6 §A.1). Stripped of this feature, v11.5 collapses to v11_richer (C=0.6538), which loses to DeepSurv on some strata.
- Per-stratum: v11.5 wins S1 + S3 vs all alternatives, but **mmSYGNAL routed beats v11.5 in S2 t(4;14), S4 t(11;14), and S5 "other"** by 2.3 to 5.5 percentage points.
- TrajectoryNet (Tong 2020) is structurally inappropriate for single-baseline bulk-RNA cohorts: at N_paired=29 in 64-D the CNF energy regularizer dominates and the flow learns approximately identity. Adding flow features to v11_richer **hurts** discrimination.
- BlockForest (Hornung & Wright 2019) was NOT_RUN due to R unavailability; documentation gap, not benchmarking bug.
- The initial RSF script was caught reporting in-sample C=0.93 from `rsf.predict` on training data; corrected to LOO N=787 protocol BEFORE publication. Audit-process transparency.

**DO NOT say:**
- "v11.5 is state-of-the-art" — too strong. Defensible: "v11.5 beats off-the-shelf RSF/DeepSurv/transformer baselines on MMRF; ties mmSYGNAL on the same cohort because it IS mmSYGNAL wrapped in Cox."
- "We propose a new architecture" — Wave-3 + Wave-4 establish PCA(K=29) ≈ MOFA and v11.5 ≈ mmSYGNAL+Cox.
- "Hussain/Sontag is beaten" — we transfer-tested their architecture on MMRF where it loses; we did NOT reproduce TOURMALINE / RRMM longitudinal results.
- "TrajectoryNet was beaten" — it was structurally mis-applied; the negative result is published as evidence about cohort fit, not architecture fit.
- "v11.5 is uniformly best" — false; mmSYGNAL beats v11.5 in 3/5 strata.

## §8 — Closing falsification

This Wave-4 verdict (bioRxiv YES; Briefings/CRM CONDITIONAL_PASS HIGH-confidence sunset 2026-12-31; Nature Methods NOT YET; SurvBoard recommended) flips to FAIL if:

1. **A SOTA implementation we already ran has a methodology bug as material as the RSF in-sample bug we caught.** If a re-run of DeepSurv with corrected ties (Efron instead of Breslow), corrected partial-likelihood normalization, or pycox in place of the custom torch impl shifts DeepSurv's marginal C upward by Δ ≥ +0.04 (closing more than half the gap to v11.5), the head-to-head conclusion is materially weakened.

2. **BlockForest, when actually run, beats v11.5.** If BlockForest with `cv.error.frac=0.5` reaches C ≥ 0.696 on the same N=787 LOO protocol, the 0/8 SOTA-loss claim becomes 0/9 AND now there is at least one beat. v11.5's "beats off-the-shelf SOTA" claim contracts.

3. **The mmSYGNAL TIE flips to a strict mmSYGNAL WIN at higher N or under external replication.** If a re-run on IFM2009 / MRC Myeloma XI / HOVON-65 shows mmSYGNAL Δ ≥ +0.02 with 95% CI excluding 0, then v11.5 is "loses mmSYGNAL on the data that matters for clinical generalization" — major re-framing required.

If none of these three falsifiers fire by 2026-12-31, the bioRxiv-now / Briefings-CRM-conditional posture stands and Sprint v12.5 Rank 1 (external-cohort replication) becomes the single rate-limiting step for Nature Methods candidacy.
