# Table 1: TT2L Survival-Proxy Comparison

Endpoint: Time-to-next-treatment (TT2L) as a resistance proxy.

| Model | Input | C-index | IBS | CI_low | CI_high |
| --- | --- | --- | --- | --- | --- |
| Clinical-Only Cox | ISS + age + gender + bort_1L + n_treatments | 0.660 | 0.251 | 0.660 | 0.660 |
| Clinical Ridge Cox | ISS + age + gender + bort_1L + n_treatments (L2) | 0.660 | 0.250 | 0.660 | 0.660 |
| Kaplan-Meier Baseline | Non-parametric KM estimator | 0.500 | 0.250 | 0.500 | 0.500 |

CI = 95% confidence interval (bootstrap or LOO).
IBS = Integrated Brier Score (lower is better).
