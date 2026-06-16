# SOTA comparison — MMRF-CoMMpass PFS (IA12)

N=769, 323 events, 5-fold patient-disjoint CV. SOTA bar = gep70/sky92 (honest). GuanScore/risk_auc are mmSYGNAL in-sample (leakage), excluded.

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| gep70(SOTA) | 0.624 [0.589,0.659] | 0.1769 | 0.6452 | 0.0149 | 0.0656 |
| sky92(SOTA) | 0.620 [0.586,0.655] | 0.1806 | 0.6472 | 0.018 | 0.0537 |
| Cox(prog+clin) | 0.632 [0.599,0.667] | 0.1818 | 0.6593 | 0.0476 | 0.0489 |
| RSF(prog+clin) | 0.631 [0.597,0.665] | 0.1792 | 0.6611 | 0.0211 | 0.0586 |
| GBS(prog+clin) | 0.643 [0.609,0.675] | 0.1772 | 0.6772 | 0.0238 | 0.0591 |
| Stacked(+gep70+sky92) | 0.644 [0.610,0.678] | 0.1765 | 0.6806 | 0.0473 | 0.0568 |
| ResistanceBasin-LR(novel) | 0.482 [0.448,0.516] | 0.1931 | 0.4976 | 0.0113 | 0.0612 |

**Non-PH:** PH holds (0/20 covariates).
**Novel vs gep70:** ΔC=-0.1419 (p=0.000); ΔIBS=+0.01610 (p=0.000, neg=better).
