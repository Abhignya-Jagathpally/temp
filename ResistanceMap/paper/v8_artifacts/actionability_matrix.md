# Actionability Decision Matrix — ResistanceMap v6

_Generated from `checkpoints/pipeline_validated.pt`. test_mse=2.3730967044830322 on n_test=132 cell lines × 11 drugs._

**Thresholds (from `validate_pipeline`):**
- Actionable for compound-prioritization screening: Spearman ≥ 0.25 AND n_test ≥ 30
- Well-calibrated for absolute IC50 prediction: additionally MSE < 1.0

## Headline

> ResistanceMap rank-prioritizes compounds for 9/11 drugs (82%). 8/11 are also well-calibrated for absolute IC50 prediction. Known failure modes: ['Panobinostat', 'Romidepsin'].

## Per-drug decision matrix

| Drug | Class | n_test | MSE | MAE | Spearman | Screening? | Calibrated? | Verdict |
|---|---|---|---|---|---|---|---|---|
| Venetoclax | BCL2 | 68 | 0.010 | 0.086 | 0.328 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Bortezomib | Proteasome | 69 | 0.032 | 0.140 | 0.338 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Dinaciclib | CDK | 66 | 0.156 | 0.248 | 0.373 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Palbociclib | CDK | 69 | 0.197 | 0.356 | 0.305 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Cyclophosphamide | DNA-damage | 68 | 0.279 | 0.390 | 0.358 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Vorinostat | HDAC | 69 | 0.413 | 0.467 | 0.310 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Lenalidomide | IMiD | 69 | 0.499 | 0.492 | 0.396 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Etoposide | DNA-damage | 67 | 0.531 | 0.518 | 0.328 | ✅ | ✅ | **Use for screening + IC50 estimation** |
| Doxorubicin | DNA-damage | 69 | 1.666 | 0.634 | 0.339 | ✅ | — | Use for screening only (rank-meaningful, absolute IC50 unreliable) |
| Romidepsin | HDAC | 56 | 0.237 | 0.292 | 0.240 | ❌ | — | **❌ DO NOT USE — failure mode documented** |
| Panobinostat | HDAC | 66 | 22.335 | 1.069 | 0.045 | ❌ | — | **❌ DO NOT USE — failure mode documented** |

## By drug class

| Class | n_drugs | Mean MSE | Mean Spearman | n_actionable | Drugs |
|---|---|---|---|---|---|
| Proteasome | 1 | 0.032 | 0.338 | 1/1 | Bortezomib |
| DNA-damage | 3 | 0.825 | 0.341 | 3/3 | Cyclophosphamide, Doxorubicin, Etoposide |
| CDK | 2 | 0.176 | 0.339 | 2/2 | Dinaciclib, Palbociclib |
| IMiD | 1 | 0.499 | 0.396 | 1/1 | Lenalidomide |
| HDAC | 3 | 7.662 | 0.198 | 1/3 | Panobinostat, Romidepsin, Vorinostat |
| BCL2 | 1 | 0.010 | 0.328 | 1/1 | Venetoclax |

## What this means in practice

### ✅ Use ResistanceMap for

**Compound-prioritization screening** (rank candidate cell lines by predicted resistance) for these drugs:

- **Bortezomib** — also well-calibrated for absolute IC50
- **Cyclophosphamide** — also well-calibrated for absolute IC50
- **Dinaciclib** — also well-calibrated for absolute IC50
- **Doxorubicin**
- **Etoposide** — also well-calibrated for absolute IC50
- **Lenalidomide** — also well-calibrated for absolute IC50
- **Palbociclib** — also well-calibrated for absolute IC50
- **Venetoclax** — also well-calibrated for absolute IC50
- **Vorinostat** — also well-calibrated for absolute IC50

### ❌ Do NOT use ResistanceMap for

**Panobinostat, Romidepsin** — Spearman correlation is at chance level. These are documented failure modes. The omics features available do not contain sufficient signal for this drug class given the current training data scale.

### ⚠️ Use with caution

**Doxorubicin** — rank-correlation is meaningful but absolute IC50 estimates have MSE > 1.0. Use the predictions to *rank* compounds, not to estimate dose-response curves.

## Always disqualified (out of scope)

- Patient-level IC50 prediction (no patient training data; cell-line cohort only)
- Time-to-resistance forecasting (no longitudinal training pairs)
- Pathway-level transition prediction (no perturbation grounding)
- HDAC-class compounds (3 of 3 HDAC drugs underperform — class-level limitation)

## Reproducibility

```bash
python main.py --config configs/default.yaml
python scripts/run_baselines_real.py
python scripts/generate_actionability_report.py
```

All numbers above derive from `checkpoints/pipeline_validated.pt`. 
Source of truth: `logs/per_drug_metrics.csv` + `paper/tables/baseline_comparison.json`.
