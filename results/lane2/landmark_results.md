# Lane #2 — treatment-conditioned PH-free landmark results

- Cohort: n=712 patients, events=311, programs=141, clinical=10.
- Immortal-time-safe LANDMARK design: at each L, restrict to patients event-free & at-risk at L; features = static x + Z(L) (history<=L only); forward survival from L.

## Forward time-dependent AUC by landmark (mean +/- std over folds)

| Landmark (d) | static_cox | timevarying_cox | treatment_basin | n_at_risk | n_events |
|---|---|---|---|---|---|
| 180 | 0.644±0.063 | 0.656±0.065 | 0.500±0.000 | 615 | 263 |
| 365 | 0.572±0.068 | 0.582±0.068 | 0.500±0.000 | 523 | 204 |
| 730 | 0.518±0.033 | 0.495±0.016 | 0.500±0.000 | 300 | 84 |

## Integrated Brier Score by landmark (lower better)

| Landmark (d) | static_cox | timevarying_cox | treatment_basin |
|---|---|---|---|
| 180 | 0.170 | 0.169 | 0.179 |
| 365 | 0.180 | 0.180 | 0.185 |
| 730 | 0.191 | 0.192 | 0.182 |

## Pre-registered gate: `treatment_nonph_beats_static`

**Criterion:** a treatment model (timevarying_cox or treatment_basin) beats static_cox on forward td-AUC at >=1 landmark with paired-fold bootstrap CI of the delta strictly > 0

| Landmark | model | delta td-AUC vs static | 95% CI | CI>0? |
|---|---|---|---|---|
| 180 | timevarying_cox | 0.012 | [0.000, 0.019] | YES |
| 180 | treatment_basin | -0.144 | [-0.201, -0.084] | no |
| 365 | timevarying_cox | 0.010 | [0.004, 0.014] | YES |
| 365 | treatment_basin | -0.072 | [-0.132, 0.002] | no |
| 730 | timevarying_cox | -0.024 | [-0.045, -0.003] | no |
| 730 | treatment_basin | -0.018 | [-0.051, 0.014] | no |

**GATE VERDICT: PASS**

## Grambsch-Therneau PH evidence

- `n_lines_total` (n_lines_total_only): chi2=19.108, p=1.24e-05, violates_PH=True
- clinical baseline: 0/10 covariates violate PH (expect ~0 — baseline PH holds).

_n_lines is used here ONLY as a PH diagnostic. Its predictive use is immortal-time-biased and is handled time-varyingly in the landmark CV._

## PH-free basin model — collapse diagnostic

- `treatment_basin` collapsed to a single basin at landmark(s) [180, 365, 730] (td-AUC == 0.500, ~0 fold variance).

_A basin td-AUC pinned at ~0.500 with near-zero fold variance is the single-basin collapse documented in docs/LANE2_TREATMENT_NONPH.md: the partial-log-rank objective on this wide, low-signal substrate puts all soft mass on one basin, yielding a constant risk. Reported honestly, not hidden._
