# MORT-FM — Reproducibility

## Environment

* Python 3.11 / 3.12 (pinned in `pyproject.toml`).
* Torch >= 2.1 (CPU smoke works without CUDA).
* Optional: `torchdiffeq`, `torchsde`, `geomloss`, `scanpy`, `muon`,
  `lifelines`. Each loader / dynamics module degrades gracefully if its
  optional dependency is missing.

## Deterministic ingredients

| Source of nondeterminism      | How we handle it                                       |
|-------------------------------|--------------------------------------------------------|
| Patient split                  | `torch.Generator().manual_seed(cfg.seed)` in `split_patients` |
| DataLoader order               | `shuffle=True` with worker seed propagation                  |
| Modality subsampling           | Deterministic head-truncation (no `np.random`)               |
| SDE integration                | `torch.manual_seed(cfg.seed)` before training; integrator is Euler-Maruyama with fixed `dt` |
| `torch.compile`                | Currently disabled in `MORTFMConfig`; enable explicitly      |
| Mixed precision                | `cfg.mixed_precision` flag; bit-exact reproduction requires `mixed_precision=False` and `torch.use_deterministic_algorithms(True)` |

## Recommended `seed` discipline

```python
import torch, random
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
torch.use_deterministic_algorithms(True, warn_only=True)
```

`MORTFMTrainer` does NOT call any of these on your behalf — they are caller
responsibility, because the same instance might be reused with different
seeds for ensembling.

## Checkpoint contents

`MORTFMTrainer.save_checkpoint(name)` writes a dict with:

* `model_state_dict`
* `optimizer_state_dict`
* `config` — `asdict(MORTFMConfig)`
* `history` — list of per-epoch `TrainingMetrics`

This is sufficient to (a) resume training, (b) reproduce inference, (c)
audit which config/hyperparameters produced a given metric. It is *not*
sufficient to reproduce the data: that requires the same upstream loader
versions + same MMRF release. Pin both via:

```
git rev-parse HEAD  # resistancemap version
ls data/raw/mmrf_commpass/  # release ID in MMRF_CoMMpass_IA{XX}_*
```

## Smoke-test as regression gate

```
python scripts/mortfm_smoke_test.py --config configs/mortfm_debug.yaml
```

If this fails on `main`, do not merge. The full unit suite
(`pytest tests/mortfm/`) is the next layer.

## Logging

* `logger = logging.getLogger("resistancemap.<module>")` everywhere.
* Per-epoch metrics go to `logger.info` *and* `MORTFMTrainer.history`.
* Long-form runs should add a `RotatingFileHandler` writing to
  `cfg.log_dir`.

## Numerical reproduction across machines

Bit-exact across CPU↔GPU is **not** guaranteed (cuDNN, SDPA, MMA). For
publication, we record:

* Hardware (GPU name + driver).
* PyTorch version, CUDA version.
* `git rev-parse HEAD`.
* Full config (saved into checkpoint).
* Per-epoch loss components (saved into checkpoint history).
