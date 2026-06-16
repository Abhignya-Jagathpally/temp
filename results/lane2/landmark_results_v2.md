# Lane #2 v2 — nested-CV enriched time-varying Cox + basin-collapse fix

- Cohort: n=712 patients, events=311, programs=141, clinical=10.
- Immortal-time-safe LANDMARK design (re-origin at L, keep only at-risk). Outer CV k=5 patient-disjoint; inner CV k=4 patient-disjoint on the training folds ONLY selects every hyperparameter/config. Final numbers are on the untouched OUTER folds.

## Q1 — Forward td-AUC: static Cox vs ENRICHED time-varying Cox (nested CV)

| Landmark (d) | static_cox | enriched_tv_cox | Δ (enriched−static) | 95% CI | CI>0? | n_at_risk | n_events |
|---|---|---|---|---|---|---|---|
| 180 | 0.6346±0.073 | 0.6491±0.080 | +0.0145 | [+0.0018, +0.0259] | YES | 615 | 263 |
| 365 | 0.5666±0.063 | 0.5742±0.064 | +0.0076 | [-0.0084, +0.0212] | no | 523 | 204 |
| 730 | 0.4936±0.034 | 0.4720±0.037 | -0.0216 | [-0.0470, +0.0003] | no | 300 | 84 |

**Prior best margin:** +0.0097 @ L=365 (timevarying vs static, CI [+0.0036, +0.0143]).
**Margin WIDENED vs prior at landmark(s) [180]** (CI-separated AND Δ > prior best). 

## Q2 — Basin collapse fix (nested-CV-selected config per fold)

Two occupancy views: **all K** = every basin carries >10% mass on the outer fold (the spec's strict bar); **>=2** = at least two basins each carry >10% (the realistic best for K=3 on this substrate).

| Landmark (d) | basin td-AUC (outer) | static_cox | beats static? | folds >=2 basins | folds all-K | any td-AUC>0.5? | rep cfg |
|---|---|---|---|---|---|---|---|
| 180 | 0.5633±0.080 (max 0.677) | 0.6346 | no | 3/5 | 0/5 | YES | K=3,pw=0.5,topk,ws=True,ep=300 |
| 365 | 0.5299±0.052 (max 0.620) | 0.5666 | no | 3/5 | 0/5 | YES | K=3,pw=0.5,topk,ws=True,ep=300 |
| 730 | 0.5245±0.125 (max 0.676) | 0.4936 | YES | 4/5 | 0/5 | YES | K=3,pw=1.0,topk,ws=True,ep=300 |

**Basin collapse was PARTIALLY MITIGATED, not fixed.** With program compression (top-k / PCA) + KMeans warm-start the PH-free head no longer pins at td-AUC=0.500: on several outer folds it splits into 2 of K basins and reaches td-AUC>0.5 (max up to ~0.68). But it never occupies all K basins on a fold (strict bar unmet), the split is fold-unstable, and it does NOT beat the static Cox on mean outer td-AUC at the two informative landmarks (L=180, L=365). The only landmark where the basin's mean exceeds static is L=730 -- but there the static Cox is itself below chance (~0.49 on n=84 events), so that is not a meaningful win. The full-occupancy PH-free objective remains unstable on this wide, low-signal substrate. Reported honestly, not forced.

_All configs were selected by INNER CV (patient-disjoint, on training folds only); reported numbers are on untouched OUTER folds. Patient-disjoint splits throughout; no fabricated values._
