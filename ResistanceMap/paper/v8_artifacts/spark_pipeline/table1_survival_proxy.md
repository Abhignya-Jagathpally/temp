# Table 1: TT2L Survival-Proxy Comparison

Endpoint: Time-to-next-treatment (TT2L) as a resistance proxy.

| Model | Input | C-index | IBS | CI_low | CI_high |
| --- | --- | --- | --- | --- | --- |
| Clinical-Only Cox | ISS + age + gender + bort_1L + n_treatments | 0.660 | 0.251 | 0.660 | 0.660 |
| Clinical Ridge Cox | ISS + age + gender + bort_1L + n_treatments (L2) | 0.660 | 0.250 | 0.660 | 0.660 |
| RNA-Only Cox | Top-5000 RNA genes | 0.643 | 0.259 | 0.441 | 0.846 |
| RNA + Clinical Cox | RNA + clinical features | 0.726 | 0.238 | 0.712 | 0.739 |
| Random Survival Forest | Clinical features (tree ensemble) | 0.715 | 0.419 | 0.706 | 0.723 |
| DeepSurv MLP | Clinical features (neural network) | 0.739 | 0.265 | 0.732 | 0.746 |
| Kaplan-Meier Baseline | Non-parametric KM estimator | 0.500 | 0.250 | 0.500 | 0.500 |
| elasticnet_cox | Unknown | 0.730 | 0.263 | 0.726 | 0.734 |
| gradient_boosted_survival | Unknown | 0.645 | 0.280 | 0.594 | 0.697 |

CI = 95% confidence interval (bootstrap or LOO).
IBS = Integrated Brier Score (lower is better).
