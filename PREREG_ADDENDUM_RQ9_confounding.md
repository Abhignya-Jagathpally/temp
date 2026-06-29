# PRE-REGISTRATION AMENDMENT — RQ9 counterfactual test, confounding controls
**Amends `PREREGISTRATION_LANE2.md` §5. Committed BEFORE the counterfactual run. The §4 gate
FAILED and H1/H3 are negative, so §5 is now the SOLE remaining route to a positive C2 result —
which makes guarding it against false positives essential.**

## The threat: confounding-by-indication
Treatment in CoMMpass is observational. Clinicians assign guideline-concordant regimens, so
observed outcomes correlate with trial-proven benefit **through the assignment process**, not
through any causal model. A model can therefore score high "concordance with trial HRs" by
learning *who-got-what*, while its generative flow contributes nothing. A naive §5 pass is then
uninterpretable. These controls make either outcome meaningful.

## Required controls (all pre-committed)
1. **Negative-control trials (must FAIL to benefit).** The model must NOT rank the known-negative
   regimens as beneficial: KEYNOTE-183 (pembro+pom worse PFS/OS), DREAMM-3 (belamaf mono not
   superior), BELLINI (OS harm overall). Report positive-trial and negative-control concordance
   separately. Favoring a negative-control regimen ⇒ the signal is confounding ⇒ FAIL.
2. **Observational-association baseline (GeoFlow must BEAT it).** Fit a confounded, non-causal
   estimator of per-class benefit: a propensity-naive Cox with treatment-class indicators (or the
   marginal per-class TT2L hazard in the cohort). Compute ITS concordance with trial −log(HR).
   **GeoFlow's counterfactual ordering must exceed this baseline's concordance** (non-overlapping
   bootstrap CI). If GeoFlow only ties it, the concordance is confounding, not generative
   capability ⇒ treat as FAIL.
3. **Propensity / IPTW sensitivity.** Re-estimate concordance under IPTW (or a g-formula
   adjustment) for treatment-class propensity given φ. Report with/without; a result that
   evaporates under adjustment is confounding.
4. **Permutation null.** Shuffle the regimen→benefit map B≥1000×; require empirical p<0.05.

## Amended §5 success criterion (all must hold)
SUPPORTED only if: (a) positive-trial Spearman ≥ 0.5 AND sign-agreement ≥ 0.8; AND
(b) the model does **not** favor any negative-control regimen; AND
(c) GeoFlow's concordance **CI-exceeds** the observational-association baseline; AND
(d) the result survives IPTW adjustment and the permutation null (p<0.05).
Otherwise → counterfactual axis FALSIFIED → complete negative (§6), now across ALL axes.

## Claim ceiling (even if SUPPORTED)
Frame strictly as **hypothesis-generating under A5 ignorability, stress-tested against
confounding** — never causal proof, never clinical guidance. The honest headline if it passes:
"a generative flow recovers trial-consistent treatment orderings beyond observational
association" — a *capability* claim, not a discrimination or causal-effect claim.

## Effort / stop rule
The evaluation harness exists (`counterfactual/trial_concordance.py`,
`data/tables/rrmm_trials.csv`, 16 trials incl. the 3 negative controls). Remaining build =
treatment-conditioned GeoFlow (a≠0) on Lane-2 + the observational baseline + IPTW. If control #2
(beat the observational baseline) fails at the fold-0 smoke stage, STOP and write §6 — do not
tune to cross threshold.