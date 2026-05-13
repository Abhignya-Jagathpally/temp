# Real-Data Baseline Comparison

- Splits: train=622, val=132, test=132
- Features: 19219 (proteomics 19177 + epigenomics 42)
- Targets: 11-dim sparse drug-sensitivity (30.1% NaN in test)
- Drugs: Bortezomib, Cyclophosphamide, Dinaciclib, Doxorubicin, Etoposide, Lenalidomide, Palbociclib, Panobinostat, Romidepsin, Venetoclax, Vorinostat
- Evaluation: NaN-masked MSE pooled across observed (sample, drug) cells (matches resistancemap/main.py:1440-1443)

| Rank | Model | test_mse | val_mse | wall (s) | notes |
|---|---|---|---|---|---|
| 1 | Zero (z-scored target floor) | 2.8106 | 1.9623 | 0.0 | predicts 0 everywhere |
| 2 | PerDrugTrainMean | 2.8106 | 1.9623 | 0.0 | predicts per-drug train mean |
| 3 | ResistanceMap (10-agent DAG) | 2.8201 | n/a | n/a | from checkpoints/pipeline_validated.pt |
| 4 | RandomForest (PCA-256) | 2.8609 | 2.1208 | 8.11 |  |
| 5 | GradientBoosting (PCA-256) | 2.8649 | 2.2702 | 107.72 |  |
| 6 | ElasticNet (PCA-256) | 2.9423 | 2.0776 | 0.02 |  |
| 7 | Lasso (PCA-256) | 2.9432 | 2.0783 | 0.02 |  |
| 8 | Ridge (full proteomics+epi) | 2.9723 | 2.1187 | 0.87 |  |
| 9 | Ridge (PCA-256) | 3.1020 | 2.2274 | 0.03 |  |
| 10 | XGBoost (PCA-256) | 3.5786 | 2.2584 | 12.31 |  |
| 11 | MLP-256-128 (PCA-256) | 4.6976 | 4.3402 | 2.2 |  |
