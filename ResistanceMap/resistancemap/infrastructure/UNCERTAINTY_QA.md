# Uncertainty Quantification Module (Gap 6.3)

## Module Location
`resistancemap/infrastructure/uncertainty_quantification.py` (755 lines)

## Architecture Overview

### 6-Layer Stack
1. **Deep Ensemble** → Epistemic uncertainty via model averaging
2. **Evidential DL** → Aleatoric/epistemic via Normal Inverse-Gamma
3. **Bayesian ODE** → Credible bands for ODE trajectories (optional)
4. **Conformal Quantile Regression** → Distribution-free prediction intervals
5. **Selective Prediction** → Abstention mechanism for high-uncertainty cases
6. **Clinical Communication** → Risk category mapping + confidence intervals

## Core Classes

### 1. `DeepEnsembleWrapper`
```python
ensemble = DeepEnsembleWrapper(
    model_fn=lambda: MyModel(...),
    n_models=7,
    device=torch.device('cuda:0')
)
ensemble_mean, ensemble_var = ensemble(x)
```

### 2. `EvidentialRegressionHead`
```python
evidential = EvidentialRegressionHead(
    input_dim=128,
    output_dim=1,
    lambda_coef=0.01
)
mu, params = evidential(x)  # params: {'gamma', 'nu', 'alpha', 'beta'}
aleatoric_var, epistemic_var = evidential.compute_uncertainties(params)
loss = evidential.nig_negative_log_likelihood(y, mu, params)
loss += evidential.kl_regularization(params)
```

### 3. `BayesianODEWrapper` (Optional)
```python
bayesian_ode = BayesianODEWrapper(
    ode_solver=solve_ode,  # fn(params, t) -> trajectory
    param_mean=torch.zeros(10),
    param_std=torch.ones(10),
    n_samples=50
)
mean_traj, lower_band, upper_band = bayesian_ode(t_eval)
```

### 4. `ConformalCalibrator`
```python
cqr = ConformalCalibrator(
    input_dim=128,
    output_dim=1,
    alpha=0.05  # Target 95% coverage
)

# Phase 1: Train on full dataset
loss_qr = cqr.quantile_loss(y_train, y_lower, y_upper)

# Phase 2: Calibrate on holdout
cqr.calibrate(x_calib, y_calib)

# Inference: Conformal-adjusted intervals
y_lower_adj, y_upper_adj = cqr.predict_with_conformal(x_test)
```

### 5. `SelectivePredictionGate`
```python
gate = SelectivePredictionGate(
    threshold=0.3,
    use_epistemic=True
)
pred, abstain_flag = gate(prediction, aleatoric_var, epistemic_var)

# Coverage-accuracy trade-off
coverages, accuracies = gate.get_coverage_accuracy_curve(
    predictions, uncertainties, targets
)
```

### 6. `ClinicalUncertaintyCommunicator`
```python
clinical = ClinicalUncertaintyCommunicator(
    bounds={
        'Low': (0.0, 0.3),
        'Moderate': (0.3, 0.6),
        'High': (0.6, 0.85),
        'Very High': (0.85, 1.0)
    }
)
risk_cats, conf_bands = clinical(point_est, std_dev)
# risk_cats: ['Low', 'Moderate', ...]
# conf_bands: [(0.1, 0.4), (0.45, 0.65), ...]
```

## Top-Level Orchestrator

### `UncertaintyQuantifier`
```python
uq = UncertaintyQuantifier(
    base_model_fn=lambda: ResistanceNet(...),
    input_dim=256,
    output_dim=1,
    n_ensemble=7,           # Deep Ensemble size
    alpha=0.05,             # CQR miscoverage level
    selective_threshold=0.3, # Abstention threshold
    ode_solver=optional_ode  # Can omit
)

# Training
loss = uq.compute_loss(
    x_train, y_train,
    lambda_ensemble=0.3,    # MSE weight
    lambda_evidential=0.5,  # NIG+KL weight
    lambda_cqr=0.2          # Quantile loss weight
)
loss.backward()
optimizer.step()

# Calibration (2-phase)
uq.calibrate_conformal(x_calib, y_calib)

# Inference returns UncertaintyOutput
output = uq(x_test)

# Access all 6 layers' outputs:
output.point_estimate           # Layer 1+2 fused estimate
output.aleatoric_variance       # Layer 2 (NIG)
output.epistemic_variance       # Layer 2 (NIG)
output.prediction_interval_lower # Layer 4 (CQR)
output.prediction_interval_upper # Layer 4 (CQR)
output.risk_category            # Layer 6 (clinical)
output.confidence_band          # Layer 6 (clinical)
output.abstain_flag             # Layer 5 (selective)
output.raw_ensemble             # Layer 1 diagnostics
output.evidential_params        # Layer 2 diagnostics
```

## Training Workflow

```python
# 1. Data split
train_data, calib_data, test_data = split(all_data, [0.6, 0.2, 0.2])

# 2. Instantiate
uq = UncertaintyQuantifier(...)

# 3. Train on full training set (all 6 layers simultaneously)
for epoch in range(100):
    for x_batch, y_batch in train_loader:
        loss = uq.compute_loss(x_batch, y_batch)
        loss.backward()
        optimizer.step()

# 4. Calibrate on holdout calibration set (Phase 2)
uq.calibrate_conformal(x_calib, y_calib)

# 5. Evaluate on test set
with torch.no_grad():
    output = uq(x_test)

# Metrics
coverage = (output.prediction_interval_lower <= y_test) & \
           (y_test <= output.prediction_interval_upper)
print(f"Empirical coverage: {coverage.float().mean():.3f}")
print(f"Abstention rate: {output.abstain_flag.float().mean():.3f}")
```

## Key Mathematical Details

### Evidential (NIG) Parameters
- **γ** (gamma): Location parameter (predicted mean)
- **ν** (nu): Degrees of freedom (precision)
- **α** (alpha): Shape parameter (α > 1 for valid variance)
- **β** (beta): Rate parameter (β > 0)

Uncertainties:
```
aleatoric_var = β / (α - 1)
epistemic_var = β / (ν(α - 1))
total_var = aleatoric_var + epistemic_var
```

### Conformal Coverage Guarantee
For test point x, if C(x) = [ŷ_lower, ŷ_upper]:
```
P(Y ∈ C(X)) ≥ 1 - α
```
Holds exactly in finite samples under exchangeability assumption.

### Risk Categorization
Clinical output maps [0, 1] prediction to category:
```
Low:       [0.0, 0.3)    → "Likely safe"
Moderate:  [0.3, 0.6)    → "Monitor closely"
High:      [0.6, 0.85)   → "High concern"
Very High: [0.85, 1.0]   → "Recommend culture-based therapy"
```

## Numerical Stability Notes

1. **Softplus activations**: α = softplus(z) + 1.0, β = softplus(z) + ε
2. **Log-domain KL**: Uses digamma/lgamma for numerical stability
3. **Quantile loss**: Indicator-free formulation: `max(α, 1-α) * |y - ŷ|`
4. **Conformal score**: Empirical quantile uses ceiling formula

## Integration with ResistanceMap Pipeline

```python
# In your training loop
from resistancemap.infrastructure.uncertainty_quantification import (
    UncertaintyQuantifier,
    UncertaintyOutput
)

# After defining your base model
uq_module = UncertaintyQuantifier(
    base_model_fn=build_resistance_net,
    input_dim=resistance_feature_dim,
    output_dim=1,  # Binary resistance prediction
    n_ensemble=5,  # Conservative for production
    alpha=0.1,     # 90% coverage
    selective_threshold=0.4
)

# Use in training loop
output: UncertaintyOutput = uq_module(x_batch)
loss = uq_module.compute_loss(x_batch, y_batch)

# At inference
clinical_recommendations = output.risk_category
confidence_intervals = output.confidence_band
should_abstain = output.abstain_flag.numpy()
```

## Testing Checklist

- [ ] Deep Ensemble: K models produce independent predictions
- [ ] Evidential: NIG parameters satisfy α > 1, β > 0, ν > 0
- [ ] Conformal: Empirical coverage ≈ 1-α on test set
- [ ] Selective: Abstention rate vs. accuracy trade-off makes sense
- [ ] Clinical: Risk categories align with prediction magnitudes
- [ ] Combined loss: All three components contribute meaningfully
- [ ] GPU compatibility: Module runs on CUDA devices
- [ ] Batch dimension: Handles variable batch sizes

## Performance Notes

- **Inference latency**: ~K forward passes (ensemble) + overhead
- **Memory**: ~K * model_size for ensemble + small overhead for heads
- **Recommended ensemble size**: 5-7 for production balance
- **Calibration cost**: One pass over calib_data (linear in size)

