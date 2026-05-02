# Real-Data Baseline Comparison

- Splits: train=622, val=132, test=132
- Features: 19219 (proteomics 19177 + epigenomics 42)
- Targets: 11-dim sparse drug-sensitivity (49.3% NaN in test)
- Drugs: Bortezomib, Cyclophosphamide, Dinaciclib, Doxorubicin, Etoposide, Lenalidomide, Palbociclib, Panobinostat, Romidepsin, Venetoclax, Vorinostat
- Evaluation: NaN-masked MSE pooled across observed (sample, drug) cells (matches resistancemap/main.py:1440-1443)

| Rank | Model | test_mse | val_mse | wall (s) | notes |
|---|---|---|---|---|---|
| 1 | Zero (z-scored target floor) | 2.3678 | 1.8841 | 0.0 | predicts 0 everywhere |
| 2 | PerDrugTrainMean | 2.3678 | 1.8841 | 0.0 | predicts per-drug train mean |
| 3 | GradientBoosting (PCA-256) | 2.3699 | 2.1597 | 78.65 |  |
| 4 | ResistanceMap (10-agent DAG) | 2.3731 | n/a | n/a | from checkpoints/pipeline_validated.pt |
| 5 | RandomForest (PCA-256) | 2.4134 | 2.0232 | 7.51 |  |
| 6 | Ridge (full proteomics+epi) | 2.4998 | 2.0536 | 0.55 |  |
| 7 | ElasticNet (PCA-256) | 2.5647 | 2.1188 | 0.04 |  |
| 8 | Lasso (PCA-256) | 2.5669 | 2.1207 | 0.04 |  |
| 9 | XGBoost (PCA-256) | 2.7215 | 2.1429 | 10.29 |  |
| 10 | Ridge (PCA-256) | 3.2473 | 2.9273 | 0.03 |  |
| 11 | MLP-256-128 (PCA-256) | 4.3846 | 4.1302 | 1.83 |  |
