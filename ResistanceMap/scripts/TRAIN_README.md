# End-to-End Training Script: train.py

## Overview

`train.py` is a comprehensive orchestration script for training the full L0-L5 ResistanceMap pipeline from scratch. It integrates:

- **L0**: Data loading, validation, and train/val/test splitting
- **L1**: Proteome-to-Epigenome VAE (pan-cancer pretraining + hematological finetuning)
- **L2**: Stability scoring using ODE-based chromatin bistability model
- **L3**: Protein Network GNN training on STRING PPI network
- **L4**: Multi-modal fusion (combining VAE, trajectory, GNN, and stability outputs)
- **L5**: Resistance landscape predictor (drug resistance probabilities, state transitions, intervention targets)

## Quick Start

### Using Synthetic Data (Recommended for Testing)

```bash
# Run the full pipeline with synthetic random data
python -m scripts.train --synthetic --device cpu

# Run specific stages
python -m scripts.train --synthetic --device cpu --stages L0,L1,L2

# Run on GPU
python -m scripts.train --synthetic --device cuda
```

### Using Real Data

```bash
# Full pipeline with real CCLE, GDSC, STRING data
python -m scripts.train --device cuda \
  --data-dir /path/to/data \
  --checkpoint-dir checkpoints/prod

# Specific config file
python -m scripts.train --config configs/my_config.yaml --device cuda
```

## Command-Line Arguments

```
--config PATH              Path to YAML config file (default: configs/default.yaml)
--device {cpu,cuda}        Device to use (default: cuda if available, else cpu)
--data-dir PATH            Directory containing raw data (default: data/)
--checkpoint-dir PATH      Directory to save checkpoints (default: checkpoints/)
--stages STAGES            Comma-separated stages to run: L0, L1, L2, L3, L4, L5
                          Special value "all" runs all stages (default: all)
--synthetic                Use synthetic random data instead of real data
```

## Stage Details

### L0: Data Preparation

**Purpose**: Load and validate multi-omics data, create train/val/test splits

**Operations**:
- Loads CCLE proteomics (DepMap)
- Loads CCLE epigenomics (chromatin profiling, ATAC, histone marks)
- Loads STRING protein-protein interaction network
- Loads drug sensitivity data (GDSC + CTRPv2)
- Optionally loads scRNA-seq data (GSE124310, GSE271107)
- Optionally loads MMRF CoMMpass clinical and genomic data

**Synthetic Mode**:
- Generates random proteomics (N × P)
- Generates random epigenomics (N × E)
- Generates random drug sensitivity with ~20% missing values
- Synthesizes random PPI edges
- Assigns random lineage labels

**Output**:
- MultiOmicsDataset object
- Train/val/test index splits (70/15/15)
- Logged data shapes and quality metrics

### L1: VAE Training

**Purpose**: Learn joint proteomics-epigenomics representation via conditional VAE

**Stages**:
1. **Pan-cancer pretraining** (200 epochs, all CCLE cell lines)
   - Learns general proteome → epigenome mapping
   - High learning rate (1e-3)
   - Early stopping on validation loss

2. **Hematological finetuning** (100 epochs, MM-relevant lineages)
   - Specializes to multiple myeloma resistance patterns
   - Lower learning rate (1e-4)
   - Prevents catastrophic forgetting of pretrain knowledge

**Model Architecture**:
- Encoder: Proteins → hidden dims → (μ, log_var)
- Decoder: z ~ N(μ, σ²) → reconstructed epigenomics
- Latent dimension: 64 (configurable)
- Activation: GELU
- KL annealing: Cyclical to prevent posterior collapse

**Output**:
- `checkpoints/l1/vae_pretrain.pt` (pretrained checkpoint)
- `checkpoints/l1/vae_finetune.pt` (final checkpoint)
- Metrics: reconstruction loss, KL divergence, validation ELBO

### L2: Stability Scoring

**Purpose**: Compute epigenetic memory stability scores using bistability ODE

**Model**: ChromatinODE + MemoryStabilityScorer
- **ChromatinODE**: Sneppen-Ringrose bistability model
  - State variables: a (active chromatin), r (repressive chromatin)
  - Parameters derived from reader/writer protein levels
  - Hill functions for cooperative binding

- **MemoryStabilityScorer**:
  - Extracts reader/writer proteins from full proteome
  - Maps protein levels → ODE parameters
  - Integrates ODE to steady state
  - Estimates basin-of-attraction depth via perturbation sampling
  - Normalizes to 0-1 stability score

**Requirements**: `pip install torchdiffeq`

**Output**:
- `checkpoints/l2/stability_scorer.pt`
- Per-sample stability scores (0 = transient, 1 = locked-in)
- Metrics: mean/std stability, distribution plots

### L3: Protein Network Training

**Purpose**: Learn drug-relevant protein representations from PPI network

**Model**: ProteinNetworkModel
- **ESM2Embedder**: facebook/esm2_t33_650M_UR50D (1280-dim embeddings)
- **PPIGraphNetwork**: Graph attention network on STRING edges
  - Features: DropEdge, PairNorm, JumpingKnowledge for over-smoothing prevention
  - Optional GraphMASK for causal edge attribution

- **PathwayAwareEncoder**: Cross-attention with pathway embeddings
- **ResistancePropagator**: Per-protein resistance contribution scores
- **EvidentialClassificationHead**: Uncertainty quantification via Dirichlet

**Requirements**: `pip install torch_geometric`

**Output**:
- `checkpoints/l3/gnn_model.pt`
- Per-protein feature importance for drug sensitivity

### L4: Fusion Training

**Purpose**: Combine four modality streams into unified representation

**Modalities**:
1. Epigenetic state (64-dim from VAE latent)
2. Trajectory state (64-dim ODE predictions)
3. Protein network output (256-dim GNN embeddings)
4. Stability score (1-dim from L2)

**Fusion Options**:
- `cross_attention` (default): Multi-head cross-modal attention + gating
- `tensor`: Outer product tensor fusion
- `concat`: Simple concatenation

**Features**:
- Optional batch correction via ResidualBatchCorrector
- Missing modality handling (fallback embeddings)
- Configurable modality dimensions

**Output**:
- `checkpoints/l4/fusion_model.pt`
- Fused representation (128-dim)
- Metrics: fusion loss, modality alignment scores

### L5: Landscape Training

**Purpose**: Predict resistance landscape and intervention targets

**Predictions**:
- **Drug resistance**: P(resistant) at 3, 6, 12 months for each drug
- **Resistance state**: Current state (sensitive / intermediate / resistant)
- **Basin-of-attraction**: Future state the tumor is drifting toward
- **State transitions**: Markov transition matrix (n_states × n_states)
- **Intervention targets**: Top proteins ranked by resistance contribution + actionability

**Model Architecture**:
- Shared encoder: fused_dim → 256 → 128
- 4 prediction heads:
  1. Drug resistance: (n_drugs × n_timepoints,) → sigmoid
  2. State classification: (n_states,) → softmax OR evidential
  3. Transition matrix: (n_states²,) → softmax
  4. Target ranking: (n_proteins,) → score

**Uncertainty Quantification** (optional):
- EvidentialResistanceHead for epistemic + aleatoric uncertainty
- Dirichlet parameterization for state predictions

**Output**:
- `checkpoints/l5/landscape_model.pt`
- Per-sample landscape predictions
- Metrics: state prediction accuracy, drug resistance AUC, target ranking

## Evaluation

After training all stages, the script computes basic evaluation metrics:

- **Test reconstruction MSE** (from VAE on held-out test samples)
- **Drug sensitivity prediction AUROC** (per drug)
- **State classification accuracy** (sensitive/intermediate/resistant)
- **Target ranking coverage** (% of top intervention targets actionable)

## Configuration

The script respects `configs/default.yaml` (or custom YAML via `--config`):

```yaml
data:
  ccle_proteomics_path: data/raw/ccle_proteomics.csv
  ccle_epigenomics_dir: data/raw/ccle_epigenomics/
  string_ppi_path: data/raw/string_ppi.txt
  gdsc_path: data/raw/gdsc_drug_sensitivity.csv
  ctrpv2_path: data/raw/ctrpv2_drug_sensitivity.csv
  test_fraction: 0.15
  val_fraction: 0.15
  random_seed: 42

vae:
  input_dim: 8000
  epigenome_dim: 50000
  latent_dim: 64
  pretrain_epochs: 200
  finetune_epochs: 100
  pretrain_lr: 1e-3
  finetune_lr: 1e-4

trajectory:
  ode_solver: euler  # euler | dopri5 | rk4
  forecast_horizons: [3, 6, 12]

protein_net:
  gnn_conv_type: gat  # gat | gcn | graphsage
  gnn_layers: 4

fusion:
  fusion_type: cross_attention  # cross_attention | tensor | concat
  fusion_epochs: 150

landscape:
  n_top_targets: 20
  use_evidential: false

hardware:
  device: cuda
  dtype: bfloat16
  pin_memory: true
  num_workers: 8
```

## Missing Data Handling

The script gracefully handles missing optional dependencies:

- **torchdiffeq not available**: L2 stability scoring is skipped with a warning
- **torch_geometric not available**: L3 GNN training uses fallback linear model
- **Real data files missing**: Automatically falls back to synthetic data

## Logging

All output is logged to:
- **Console**: INFO and above
- **File**: `logs/train.log` with full timestamp and module names

Example log output:
```
2026-04-12 14:23:45,123 - resistancemap.scripts.train - INFO - L0: DATA PREPARATION
2026-04-12 14:23:46,456 - resistancemap.scripts.train - INFO - CCLE proteomics: (456, 8000)
2026-04-12 14:23:47,789 - resistancemap.scripts.train - INFO - Common samples across modalities: 234
2026-04-12 14:23:48,012 - resistancemap.scripts.train - INFO - Train/val/test split: 164 / 35 / 35
2026-04-12 14:24:01,234 - resistancemap.scripts.train - INFO - L1: VAE TRAINING
2026-04-12 14:24:02,567 - resistancemap.scripts.train - INFO - VAE model created: 45,234,567 parameters
...
```

## Performance Considerations

- **Batch size**: Defaults to 512 for VAE (uses gradient checkpointing on H100)
- **Mixed precision**: bfloat16 recommended for memory efficiency
- **DataLoader workers**: 8 workers with prefetching for GPU utilization
- **GPU memory**: ~40GB for full pipeline on H100 (adjustable via config)

### Memory Optimization Tips

```yaml
hardware:
  dtype: float16  # More aggressive than bfloat16
  pin_memory: true
  num_workers: 4  # Reduce if OOM
  prefetch_factor: 2

vae:
  batch_size: 256  # Reduce from 512
  gradient_checkpointing: true
```

## Troubleshooting

### OOM (Out of Memory)

1. Reduce batch size in config: `vae.batch_size: 256`
2. Use lower precision: `hardware.dtype: float16`
3. Reduce number of workers: `hardware.num_workers: 4`

### ImportError: No module named 'torch_geometric'

```bash
pip install torch_geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### ImportError: No module named 'torchdiffeq'

```bash
pip install torchdiffeq
```

### Real data not found

Use `--synthetic` flag or ensure data files are in `--data-dir`:
```bash
python -m scripts.train --synthetic --device cpu
```

## Examples

### Full pipeline with 100 synthetic samples, CPU, 30 min training

```bash
python -m scripts.train --synthetic --device cpu --checkpoint-dir checkpoints/test
```

### Real data, GPU, only train VAE (L1)

```bash
python -m scripts.train --device cuda --stages L1 --data-dir /mnt/data
```

### Multi-stage, custom config, save detailed logs

```bash
python -m scripts.train \
  --config configs/research.yaml \
  --device cuda \
  --checkpoint-dir /scratch/resistancemap \
  --stages L0,L1,L2,L4,L5 \
  2>&1 | tee training.log
```

### Synthetic data for integration testing

```bash
python -m scripts.train --synthetic --stages all --device cpu
# Should complete in ~5 minutes
```

## Checkpoint Management

Checkpoints are automatically saved after each stage:

```
checkpoints/
├── l1/
│   ├── vae_pretrain.pt
│   └── vae_finetune.pt
├── l2/
│   └── stability_scorer.pt
├── l3/
│   └── gnn_model.pt
├── l4/
│   └── fusion_model.pt
└── l5/
    └── landscape_model.pt
```

To resume training from a checkpoint:

```bash
python -m scripts.train \
  --config configs/default.yaml \
  --checkpoint-dir checkpoints/prod \
  --stages L4,L5  # Skip L0-L3, reuse their checkpoints
```

## Future Enhancements

- [ ] Distributed training (DDP on multi-GPU)
- [ ] Automated hyperparameter tuning via Optuna
- [ ] Tensorboard integration for real-time monitoring
- [ ] Checkpoint resumption from any stage
- [ ] Cross-validation support
- [ ] Benchmark comparisons with baselines
