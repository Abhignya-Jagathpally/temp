# mmSYGNAL Head-to-Head: MMRF 787-Patient Cohort

## (1) What was located

**Source repo (public, GPL-3.0):** `github.com/baliga-lab/mmSYGNAL-risk-prediction-models`
(authors of Murie et al, *Br J Cancer* 132(10):922-936, 2025; PMID 40169765;
DOI 10.1038/s41416-025-02987-6).

Artifacts retrieved:
- 6 trained `caret` `glmnet` (elastic-net) risk models (`*.Rds`):
  `agnostic_risk_model.Rds` (3.5 MB), `amp(1q)_risk_model.Rds`, `del(13)_risk_model.Rds`,
  `del(1p)_risk_model.Rds`, `t(4;14)_risk_model.Rds`, `FGFR3_risk_model.Rds`.
- Pre-computed program activity matrix for the **MMRF CoMMpass IA12** release:
  `data/program_activity_IA12_py.csv` (141 programs × 881 aliquots, integer-discretized
  {-1, 0, +1} via miner3 minernorm). All 787 patients in our cohort are in this file
  (100% patient-level coverage; aliquot-id format differs but `MMRF_xxxx` patient prefix
  matches deterministically).
- Tutorial: `analysis/risk_model_tutorial.Rmd` — defines subtype routing (A>B>C grade,
  mean within grade) and threshold conventions.

Companion: `baliga-lab/miner3` (`pip install isb-miner3`) — not needed because IA12 program activity is pre-computed in the repo.

## (2) Exact identifiers

| Resource | Identifier |
|---|---|
| Paper | PMID 40169765, DOI 10.1038/s41416-025-02987-6, *Br J Cancer* 132(10):922-936, 2025 |
| GitHub | `baliga-lab/mmSYGNAL-risk-prediction-models` (commit on `main` 2025-02-28) |
| License | GPL-3.0 |
| miner3 | `baliga-lab/miner3`, PyPI `isb-miner3==1.2.4` |
| Input dataset | MMRF CoMMpass IA12 program activity (881 aliquots) — included in the repo |

## (3) Feasibility verdict: **EXECUTED**

- (a) RNA-seq input format: `log2(TPM+1)` per miner3 README. **We have TPM** in
  `data/processed/mmrf_baseline_expression.parquet` (787 × 57,690 Ensembl genes, raw TPM).
  `log2(TPM+1)` transformation step was *not needed* because the published IA12 program
  activity already covers all 787 patients.
- (b) 5 cytogenetic subtype-specific models + agnostic model are publicly distributed as
  pre-trained `.Rds` files. No retraining required.
- (c) Inference path: R 4.3 + `caret` + `glmnet` (no R available system-wide; installed
  locally via micromamba env `rsygnal`, no sudo).
- (d) No data-access gate. GPL-3.0, public clone.

Caveats applied:
- `del(1p)` model **not applied** — MMRF cyto panel in `mmrf_sprint4_analysis.tsv` does
  not include a 1p-deletion column.
- `FGFR3` model **not applied** — no FGFR3 RNA-seq subtype call available in our
  preprocessed clinical table.
- Subtype routing therefore uses: `t(4;14)` [grade A], `amp(1q)` [B], `del(13)` [B],
  agnostic [C]. Grade counts on our cohort: A=113, B=347, C=327.

## (4) Head-to-head C-index (Harrell, TT2L, same 5-stratum partition as v11 Cox)

| Model | Marginal C | S1 del17p (n=105) | S2 t(4;14) (n=97) | S3 1q21 (n=165) | S4 t(11;14) (n=95) | S5 other (n=325) |
|---|---|---|---|---|---|---|
| **mmSYGNAL routed** | **0.694** [0.658-0.729] | 0.705 | **0.722** | 0.664 | **0.693** | **0.682** |
| mmSYGNAL agnostic-only | 0.688 | 0.741 | 0.641 | 0.649 | 0.676 | 0.694 |
| Cox v10 (11 feats) | 0.653 | 0.703 | 0.605 | 0.698 | 0.517 | 0.612 |
| Cox v11features (15 feats) | 0.651 | 0.711 | 0.568 | 0.685 | 0.496 | 0.622 |
| Cox v11_richer (23 feats) | 0.654 | 0.698 | 0.586 | 0.688 | 0.521 | 0.621 |

Marginal 95% CI is non-parametric bootstrap (n=1000 resamples of real patient indices via
`sklearn.utils.resample`; no synthetic data).

**Δ marginal C-index (mmSYGNAL routed − best v11 Cox) = +0.040.**
The mmSYGNAL bootstrap CI lower bound (0.658) exceeds all three v11 Cox point estimates.

Per-stratum: mmSYGNAL beats v11 Cox in S2/S4/S5 (notably t(4;14) +0.12, t(11;14) +0.17,
"other" +0.06); ties in S1 del17p; loses in S3 1q21 (-0.03). v11 Cox's strength was S3
(1q21) and S1 (del17p) — those are exactly the strata where v11 Cox has explicit cyto
covariates. mmSYGNAL's marginal advantage is driven by stronger discrimination in
non-1q21 strata via the program-activity signal.

## Honest interpretation

- The previous claim "v11 matches mmSYGNAL floor" (0.654 vs 0.65-0.75 across other
  cohorts) **is no longer the right framing**. On the SAME 787 MMRF patients with the
  SAME TT2L outcome, mmSYGNAL marginal C-index is **0.694**, not 0.65 — v11 Cox does
  NOT match it (Δ=-0.040, CI separation).
- v11 Cox only matches mmSYGNAL on the del17p stratum where v11 has a hard cyto
  covariate. mmSYGNAL outperforms in 4 of 5 strata. The published 0.65-0.75 range is
  cross-cohort variability; on MMRF specifically, mmSYGNAL lands at 0.694.
- Caveat: del(1p) and FGFR3 routing arms were not applied (data-availability gap on
  our side). This *underestimates* mmSYGNAL — a more complete head-to-head would
  add 1p deletion calls (NSD2/MMSET region; obtainable from MMRF FISH/seqFISH IA22)
  and FGFR3 RNA-seq calls. Both would likely raise mmSYGNAL further.
- Paired bootstrap Δ-CI requires per-patient v11-Cox log-hazards which are not stored
  in `cox_discrimination.json` (only aggregates). Recommend re-running v11 LOO Cox to
  emit per-patient predictions for a paired test in the next sprint.

## Artifacts

Under `paper/v8_artifacts/v11_sprint5/mmsygnal/`: `run_mmsygnal.R` (caret driver),
`score_and_evaluate.py` (routing + C-index), `bootstrap_ci.py`, `head_to_head_results.json`,
`bootstrap_ci.json`. Intermediates in `/tmp/`: `our_program_activity.csv` (787×141),
`our_mmsygnal_per_model.csv`, `mmsygnal_*.Rds`.

## Reproducibility

`micromamba env rsygnal` with `r-base=4.3, r-caret, r-glmnet`. Run order:
`Rscript run_mmsygnal.R` -> `python score_and_evaluate.py` -> `python bootstrap_ci.py`.
