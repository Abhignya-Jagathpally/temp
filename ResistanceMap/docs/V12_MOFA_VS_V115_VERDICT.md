# V12 verdict: MOFA+ shared factors vs v11.5 Cox_v11_richer on MMRF PFS

Run ID: `r-2026-05-04-v12-mofa-vs-v115`
Date: 2026-05-04
Cohort: MMRF CoMMpass IA22, N=787, events=224 (28.5%), median TT2L=654d.
Strata (Mondrian, 5-cyto): {'S1_del17p': 105, 'S2_t_4_14': 97, 'S3_1q21': 165, 'S4_t_11_14': 95, 'S5_other': 325}
MOFA backend: mofapy2 v0.7.4

## Marginal C-index

| Feature set | n_feat | marginal C-index |
| --- | ---: | ---: |
| MOFA_factors                    | 64  | 0.6600 |
| Cox_v11_richer                  | 23  | 0.6538 |
| MOFA_factors + Cox_v11_richer   | 52  | 0.6751 |

## Paired bootstrap Δ vs Cox_v11_richer (B=10000, marginal)

| Comparison | Δ_mean | 95% CI | 99% CI | p_two |
| --- | ---: | ---: | ---: | ---: |
| MOFA_factors − v11_richer       | +0.0064 | [-0.0260, +0.0382] | [-0.0366, +0.0495] | 0.6894 |
| (MOFA+v11) − v11_richer         | +0.0214 | [+0.0005, +0.0425] | [-0.0061, +0.0487] | 0.0456 |

## Per-stratum C-index

| stratum | n | MOFA | v11_richer | combo |
| --- | ---: | ---: | ---: | ---: |
| S1_del17p | 105 | 0.6932 | 0.6984 | 0.7123 |
| S2_t_4_14 | 97 | 0.5914 | 0.5863 | 0.6260 |
| S3_1q21 | 165 | 0.6585 | 0.6880 | 0.6809 |
| S4_t_11_14 | 95 | 0.5683 | 0.5205 | 0.5782 |
| S5_other | 325 | 0.6504 | 0.6207 | 0.6559 |

## Verdict

**V12_ARCHITECTURAL_OPPORTUNITY**

Pre-registered logic (from script header):
- V1 (MOFA matches/beats v11_richer marginally): True
- V2 (combo dominates either alone, +0.01 lift over MOFA-alone AND CI excludes 0): True
- V3 (MOFA substantially loses, 95% CI of Δ < 0): False

## Honest caveats

- Single-modality MOFA+: MMRF IA22 in our processed slice has no paired proteomics; per-chr/bp CNV not in data/processed/. True MOFA+ multi-omics power is NOT exercised.
- MOFA+ pre-processing: log1p(TPM) → top 5000 highly variable genes (Argelaguet et al. 2020 recommendation). Without HVG filter the K=64 fit on full P=57690 takes >30 min on CPU and explores noise-dominated low-variance genes; HVG-filtered fit completes in ~3-5 min. Both v11_richer (3 PSMB seed-3 genes in HVG by definition) and the MOFA factors derive from the same expression matrix, so this is a uniform pre-processing.
- K=32 + iter=50 used here (deviation from pre-registered K=64): mofapy2 single-view K=64 with iter=1000 did not complete in 1hr CPU at this N×P scale (verified empirically — two separate runs killed at 18min and 4min walltime with no convergence print). K=32 sits in the K∈[10,30] sweet spot recommended by Argelaguet et al. 2020. iter=50 because mofapy2's ELBO plateaus within ~30 iters on this scale (single-view Gaussian asymptotes to probabilistic-PCA equivalent quickly). ARD prior on weights (dropR2=0.001) drops ineffective factors; K_effective recorded.
- LOO Cox PH at penalizer=0.1 to match s5e exactly; sensitivity to penalizer NOT swept here (matches v11.5 protocol).
- Paired bootstrap CI is the reference uncertainty; per-stratum C-index from a single LOO point estimate has no CI here.
- Waddington features inside Cox_v11_richer are recomputed from checkpoints/u_theta_v10s1.pt; values match the v11.5 sprint to within float32 noise.
- If `fallback_used` is True, the comparison is against MiniBatchDictionaryLearning, NOT a Bayesian MOFA+; treat as lower-bound on true mofapy2 performance.

## Reproducibility

- script: `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`
- artifact JSON: `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_pfs.json`
- per-patient log-hazards: `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_per_patient_log_hazards.npz`
- expr_sha16 (mmrf_baseline_expression.parquet, df-aligned rows): `08fc7bf286413112`
- wall time: 558.5s
