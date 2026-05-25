# Data Quality and Audit Summary

## Cohort

- **Total patients**: 994
- **Total rows**: 994
- **Survival rows**: 994
- **Trajectory pairs**: 0
- **Overall audit passed**: True

## Gates

| Gate | Threshold | Actual | Passed |
| --- | --- | --- | --- |
| min_trajectory_pairs | 10 | 0 | FAIL |
| min_survival_patients | 50 | 994 | PASS |
| min_survival_events | 20 | 994 | PASS |

## Allowed Claims

- technical_pipeline
- tt2l_survival_proxy_prediction
- static_drug_response

## Blocked Claims

- **longitudinal_trajectory**: Need 10 real molecular trajectory pairs, found 0.
- **resistance_emergence**: TT2L is a treatment-transition proxy, not direct molecular resistance emergence.
- **causal_mechanism**: No perturbational validation tied to longitudinal patient trajectory.

## Patient-Disjoint Splits

| Split | Patients | Rows | Survival Rows | Trajectory Rows |
| --- | --- | --- | --- | --- |
| train | 691 | 691 | 691 | 0 |
| test | 148 | 148 | 148 | 0 |
| val | 155 | 155 | 155 | 0 |

Split seed: 42
Method: deterministic_sha256
