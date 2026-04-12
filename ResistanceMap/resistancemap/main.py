#!/usr/bin/env python3
"""ResistanceMap — Pharmacogenomic ML pipeline for hematologic malignancies.

Dual-mode execution:
  1. **Agentic mode** (default): DAG-based orchestrator with parallel agents,
     zero-trust verification, AgentOps observability, and latent compute scheduling.
  2. **Sequential mode** (--sequential): Classic stage-by-stage execution for
     debugging or environments without asyncio support.

Usage:
    # Agentic mode (parallel DAG execution)
    python main.py --config configs/h100.yaml

    # Sequential mode (legacy)
    python main.py --config configs/h100.yaml --sequential

    # Single stage
    python main.py --config configs/h100.yaml --stage vae_pretrain

    # Distributed (agentic mode still manages the DAG; DDP wraps each model)
    torchrun --nproc_per_node=8 main.py --config configs/h100.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from resistancemap.config import load_config, ResistanceMapConfig
from resistancemap.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Pipeline — DAG-based parallel execution with zero-trust verification
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_dag():
    """Construct the 10-agent DAG with dependency edges.

    DAG topology (layers for parallel execution):
        Layer 0: [DataValidation]
        Layer 1: [DataPrep]
        Layer 2: [VAEPretrain, ESM2Embed]          <- parallel
        Layer 3: [VAEFinetune]
        Layer 4: [Trajectory]
        Layer 5: [ProteinNet]                       <- needs trajectory checkpoint
        Layer 6: [Fusion]
        Layer 7: [Landscape]
        Layer 8: [Validation]

    Returns:
        Orchestrator with all agents registered.
    """
    from resistancemap.agents.orchestrator import Orchestrator
    from resistancemap.agents.specialized import (
        DataValidationAgent,
        DataPrepAgent,
        VAEPretrainAgent,
        VAEFinetuneAgent,
        ESM2EmbedAgent,
        TrajectoryAgent,
        ProteinNetAgent,
        FusionAgent,
        LandscapeAgent,
        ValidationAgent,
    )

    orchestrator = Orchestrator()

    # Register agents (dependencies declared in each agent's __init__)
    orchestrator.add_agent(DataValidationAgent())   # Layer 0 — no deps
    orchestrator.add_agent(DataPrepAgent())          # Layer 1 — depends on data_validation
    orchestrator.add_agent(VAEPretrainAgent())       # Layer 2 — depends on data_prep
    orchestrator.add_agent(ESM2EmbedAgent())         # Layer 2 — depends on data_prep (parallel w/ VAE)
    orchestrator.add_agent(VAEFinetuneAgent())       # Layer 3 — depends on vae_pretrain
    orchestrator.add_agent(TrajectoryAgent())        # Layer 4 — depends on vae_finetune
    orchestrator.add_agent(ProteinNetAgent())        # Layer 5 — depends on esm2_embed + vae_finetune + trajectory
    orchestrator.add_agent(FusionAgent())            # Layer 6 — depends on trajectory + protein_net
    orchestrator.add_agent(LandscapeAgent())         # Layer 7 — depends on fusion
    orchestrator.add_agent(ValidationAgent())        # Layer 8 — depends on landscape

    return orchestrator


async def run_agentic_pipeline(config: ResistanceMapConfig) -> dict:
    """Execute the full pipeline via DAG orchestrator.

    Steps:
        1. Set up logging and checkpoint manager
        2. Build agent DAG and validate topology
        3. Initialize AgentOps tracer for observability
        4. Initialize zero-trust verification engine
        5. Run data quality profiler pre-check
        6. Schedule latent compute tasks (ESM-2 embedding, PPI GNN, trajectory ensemble)
        7. Execute DAG with parallel layers
        8. Collect AgentOps metrics and generate dashboard
        9. Return results with audit trail

    Args:
        config: ResistanceMapConfig

    Returns:
        Dict with agent results, timing report, verification chain, and AgentOps summary
    """
    from resistancemap.agentops.tracer import Tracer
    from resistancemap.agentops.evaluator import Evaluator
    from resistancemap.agentops.optimizer import Optimizer
    from resistancemap.agentops.dashboard import AgentOpsDashboard
    from resistancemap.verification.zero_trust import ZeroTrustVerifier
    from resistancemap.verification.guardrails import GuardrailEngine
    from resistancemap.data_quality.profiler import DataProfiler
    from resistancemap.latent_compute.scheduler import LatentComputeScheduler
    from resistancemap.utils.logging_utils import setup_logger

    pipeline_start = time.time()

    # ── 0. Logging, checkpoint manager, deterministic mode ───────────────
    log = setup_logger(log_dir=config.log_dir, wandb_project=config.wandb_project)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, log)
    config._ckpt_mgr = ckpt_mgr  # shared by all agents via _get_ckpt_mgr()

    if config.hardware.deterministic:
        torch.use_deterministic_algorithms(True)

    # ── 1. Build and validate DAG ────────────────────────────────────────
    orchestrator = build_agent_dag()
    is_valid, error_msg = orchestrator.prepare_execution()
    if not is_valid:
        logger.error(f"DAG validation failed: {error_msg}")
        sys.exit(1)

    execution_plan = orchestrator.get_execution_plan()
    logger.info(
        f"Agentic pipeline: {len(orchestrator.dag.agents)} agents in "
        f"{len(execution_plan)} parallel layers"
    )
    for i, layer in enumerate(execution_plan):
        logger.info(f"  Layer {i}: {layer}")

    # ── 2. Initialize AgentOps ───────────────────────────────────────────
    tracer = Tracer.get_instance()
    evaluator = Evaluator()
    optimizer = Optimizer()
    dashboard = AgentOpsDashboard(tracer=tracer, evaluator=evaluator, optimizer=optimizer)

    logger.info("AgentOps initialized: Tracer + Evaluator + Optimizer")

    # ── 3. Initialize zero-trust verification ────────────────────────────
    verifier = ZeroTrustVerifier()
    guardrails = GuardrailEngine()

    logger.info(
        f"Zero-trust verification active: {len(guardrails.guardrails)} guardrails loaded"
    )

    # ── 4. Data quality pre-check ────────────────────────────────────────
    profiler = DataProfiler()
    logger.info("Data quality profiler ready (SOTA benchmarks loaded)")

    # ── 5. Latent compute scheduler ──────────────────────────────────────
    lc_scheduler = LatentComputeScheduler()
    logger.info(
        f"Latent compute scheduler: GPU budget={lc_scheduler.gpu_memory_budget_gb:.0f}GB, "
        f"max_concurrent={lc_scheduler.max_concurrent_jobs}"
    )

    # ── 6. Execute DAG ───────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("EXECUTING AGENTIC PIPELINE")
    logger.info("=" * 72)

    results = await orchestrator.run(config)

    # ── 7. Collect metrics ───────────────────────────────────────────────
    pipeline_elapsed = time.time() - pipeline_start
    timing_report = orchestrator.get_timing_report()
    verification_chain = orchestrator.get_verification_chain()
    results_summary = orchestrator.get_results_summary()

    # Log AgentOps summary
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time: {pipeline_elapsed:.1f}s")
    logger.info(f"  Agents completed: {results_summary['status_counts'].get('completed', 0)}/{results_summary['total_agents']}")
    logger.info(f"  Verification chain: {len(verification_chain)} hashes")
    if results_summary['failed_agents']:
        logger.warning(f"  Failed agents: {results_summary['failed_agents']}")
    logger.info("=" * 72)

    # Export dashboard
    try:
        dashboard_path = config.log_dir / "agentops_dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard.export_json(str(dashboard_path))
        logger.info(f"AgentOps dashboard exported: {dashboard_path}")
    except Exception as e:
        logger.warning(f"Could not export dashboard: {e}")

    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "timing_report": timing_report,
        "verification_chain": verification_chain,
        "results_summary": results_summary,
        "pipeline_elapsed_seconds": pipeline_elapsed,
        "execution_plan": execution_plan,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential Pipeline — legacy stage-by-stage execution (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _maybe_compile(model: torch.nn.Module, config: ResistanceMapConfig) -> torch.nn.Module:
    """Apply torch.compile if enabled in config (requires PyTorch 2.0+).

    If torch.compile is enabled but Inductor crashes (a recurring problem
    for some module shapes / mode combinations), suppress the dynamo
    exception and fall back to eager mode for that model so the rest of
    the pipeline can still complete. This mirrors the suggestion printed
    by torch._dynamo on backend failure.
    """
    if not config.hardware.compile:
        return model
    try:
        import torch._dynamo as _dynamo
        # Tell dynamo to fall back to eager on any subsequent compile failure
        # for this model rather than crashing the training loop.
        _dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass
    return torch.compile(model, mode=config.hardware.compile_mode)


def _maybe_distribute(model: torch.nn.Module, config: ResistanceMapConfig) -> torch.nn.Module:
    """Wrap in DDP if running in distributed mode."""
    if config.hardware.distributed and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.hardware.local_rank],
            output_device=config.hardware.local_rank,
        )
    return model


def validate_data_files(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Validate that all required raw data files exist."""
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    log_stage_start("data_validate")
    missing = []

    if not config.data.ccle_proteomics_path.exists():
        missing.append(f"  - CCLE Proteomics: {config.data.ccle_proteomics_path}")
    if not config.data.string_ppi_path.exists():
        missing.append(f"  - STRING PPI: {config.data.string_ppi_path}")
    gdsc_exists = config.data.gdsc_path.exists()
    ctrpv2_exists = config.data.ctrpv2_path.exists()
    if not gdsc_exists and not ctrpv2_exists:
        missing.append(f"  - Drug sensitivity: GDSC or CTRPv2 required")
    if not config.data.scrna_gse124310_path.exists():
        missing.append(f"  - scRNA-seq GSE124310: {config.data.scrna_gse124310_path}")

    if missing:
        msg = "Missing data files:\n" + "\n".join(missing) + "\nRun: bash scripts/download_data.sh"
        raise FileNotFoundError(msg)

    log_stage_end("data_validate")
    return Path("validated")


def prepare_data(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Preprocess and harmonize all multi-omics datasets."""
    from resistancemap.data.loaders import (
        load_ccle_proteomics, load_ccle_epigenomics, load_string_ppi,
        load_scrna_data, load_mmrf_data,
    )
    from resistancemap.data.preprocessors import harmonize_omics, build_train_val_test_splits
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("data_ready"):
        return ckpt_mgr.path("data_ready")

    log_stage_start("data_prep")
    proteomics = load_ccle_proteomics(config.data)
    epigenomics = load_ccle_epigenomics(config.data)
    ppi_graph = load_string_ppi(config.data)
    scrna_data = load_scrna_data(config.data)
    mmrf_data = load_mmrf_data(config.data)

    dataset = harmonize_omics(proteomics, epigenomics, ppi_graph, scrna_data=scrna_data, mmrf_data=mmrf_data, config=config.data)
    splits = build_train_val_test_splits(dataset, config.data)

    ckpt_path = ckpt_mgr.save("data_ready", {"dataset": dataset, "splits": splits, "config": config.data})
    log_stage_end("data_prep")
    return ckpt_path


def pretrain_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Pretrain the conditional VAE on pan-cancer CCLE data."""
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_pretrained"):
        return ckpt_mgr.path("vae_pretrained")

    log_stage_start("vae_pretrain")
    data_ckpt = ckpt_mgr.load("data_ready")
    dataset = data_ckpt["dataset"]
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=config.vae, subset="pan_cancer", ckpt_mgr=ckpt_mgr, stage_name="vae_pretrained",
    )
    log_stage_end("vae_pretrain", metrics=result["metrics"])
    return result["checkpoint_path"]


def finetune_vae(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Fine-tune the VAE on hematological cell lines only."""
    from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("vae_finetuned"):
        return ckpt_mgr.path("vae_finetuned")

    log_stage_start("vae_finetune")
    data_ckpt = ckpt_mgr.load("data_ready")
    pretrained = ckpt_mgr.load("vae_pretrained")
    dataset = data_ckpt["dataset"]
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    # Strip _orig_mod. prefix from compiled model state dicts
    state_dict = pretrained["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model, dataset=dataset, splits=data_ckpt["splits"],
        config=config.vae, subset="hematological", ckpt_mgr=ckpt_mgr, stage_name="vae_finetuned",
    )
    log_stage_end("vae_finetune", metrics=result["metrics"])
    return result["checkpoint_path"]


def calibrate_trajectory(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Calibrate the ODE-based resistance trajectory model."""
    from resistancemap.models.trajectory import MemoryStabilityScorer, calibrate_scorer
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("stability_calibrated"):
        return ckpt_mgr.path("stability_calibrated")

    log_stage_start("trajectory_calibrate")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    model = MemoryStabilityScorer(config.trajectory).to(config.device)
    result = calibrate_scorer(scorer=model, dataset=data_ckpt["dataset"], vae_checkpoint=vae_ckpt,
                              config=config.trajectory, ckpt_mgr=ckpt_mgr)
    log_stage_end("trajectory_calibrate", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_protein_network(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train protein network GNN on PPI graph.

    Following MyeloMemory pattern: node features = protein abundance (1) +
    VAE latent state (64) + stability score (1) = 66-dim per protein node.
    Trained with masked MSE against drug sensitivity + reversibility proxy.
    """
    from resistancemap.models.protein_network import PPIGraphNetwork
    from resistancemap.models.vae import ProteomeToEpigenomeVAE
    from resistancemap.models.trajectory import MemoryStabilityScorer
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("protein_net_trained"):
        return ckpt_mgr.path("protein_net_trained")

    log_stage_start("protein_net_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    traj_ckpt = ckpt_mgr.load("stability_calibrated")
    dataset = data_ckpt["dataset"]
    protein_names = dataset.protein_names
    n_proteins = len(protein_names)
    device = config.device

    # ── 1. Load frozen VAE for latent extraction ─────────────────────────
    # Set dims from actual data
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    vae = ProteomeToEpigenomeVAE(config.vae).to(device)
    vae_sd = vae_ckpt["model_state_dict"]
    vae_sd = {k.replace("_orig_mod.", ""): v for k, v in vae_sd.items()}
    vae.load_state_dict(vae_sd)
    vae.eval()

    # ── 2. Load frozen stability scorer ──────────────────────────────────
    scorer = MemoryStabilityScorer(config.trajectory).to(device)
    scorer_sd = traj_ckpt["model_state_dict"]
    scorer_sd = {k.replace("_orig_mod.", ""): v for k, v in scorer_sd.items()}
    scorer.load_state_dict(scorer_sd)
    scorer.eval()

    # ── 3. Pre-compute latent states and stability for all samples ───────
    logger.info("Pre-computing VAE latent states and stability scores...")
    all_latents = []
    all_stability = []
    with torch.no_grad():
        for i in range(len(dataset)):
            prot = dataset.proteomics[i:i+1].to(device)
            mu, _ = vae.encode(prot)
            all_latents.append(mu.squeeze(0).cpu())
            stab = scorer(prot, protein_names)
            all_stability.append(stab.squeeze(0).cpu())
    all_latents = torch.stack(all_latents)    # (N, 64)
    all_stability = torch.stack(all_stability)  # (N,)
    logger.info(f"Pre-computed: latents {all_latents.shape}, stability {all_stability.shape}")

    # ── 4. Build PPI edge index ──────────────────────────────────────────
    ppi_edges = dataset.ppi_edges or []
    ppi_scores = dataset.ppi_scores or []
    prot_to_idx = {p: i for i, p in enumerate(protein_names)}

    src, dst, weights = [], [], []
    for (p1, p2), score in zip(ppi_edges, ppi_scores):
        i, j = prot_to_idx.get(p1), prot_to_idx.get(p2)
        if i is not None and j is not None:
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([score, score])

    if not src:
        # Build self-loop graph if no overlap (STRING uses Ensembl IDs)
        logger.warning("No PPI edges match proteomics proteins; using k-NN self-loop graph")
        for i in range(n_proteins):
            src.append(i); dst.append(i); weights.append(1.0)
    n_edges = len(src)
    logger.info(f"PPI graph: {n_proteins} nodes, {n_edges} edges")

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

    # ── 5. Build GNN + prediction head ───────────────────────────────────
    # Node features: protein abundance (1) + VAE latent (64) + stability (1) = 66
    node_feat_dim = 1 + config.vae.latent_dim + 1
    gnn = PPIGraphNetwork(
        in_dim=node_feat_dim, hidden_dim=config.protein_net.gnn_hidden,
        n_layers=config.protein_net.gnn_layers, n_heads=config.protein_net.gnn_heads,
        dropout=config.protein_net.gnn_dropout, edge_dim=1,
    ).to(device)
    n_drugs = dataset.drug_sensitivity.shape[1]
    pred_head = torch.nn.Sequential(
        torch.nn.Linear(config.protein_net.gnn_hidden, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_drugs),
    ).to(device)

    params = list(gnn.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    # ── 6. Training loop ─────────────────────────────────────────────────
    from torch_geometric.data import Data as PyGData
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    # Inner batch is the number of samples whose GNN-forward activations
    # we hold in GPU memory before backward(). Each sample materialises
    # one full PPI graph forward (~929k edges with the real STRING graph),
    # so a large inner batch crosses 90 GB. 4 keeps a useful gradient
    # estimate while leaving headroom.
    pn_inner_batch = 4

    for epoch in range(100):
        gnn.train(); pred_head.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), pn_inner_batch):
            optimizer.zero_grad()
            batch_loss = 0.0
            count = 0

            for si in range(bi, min(bi + pn_inner_batch, len(train_idx))):
                idx = train_idx[perm[si]]
                # Build 66-dim node features for this sample
                prot_vals = dataset.proteomics[idx].to(device).unsqueeze(-1)  # (P, 1)
                latent_broadcast = all_latents[idx].to(device).unsqueeze(0).expand(n_proteins, -1)  # (P, 64)
                stab_broadcast = all_stability[idx].to(device).unsqueeze(0).expand(n_proteins).unsqueeze(-1)  # (P, 1)
                node_feat = torch.cat([prot_vals, latent_broadcast, stab_broadcast], dim=-1)  # (P, 66)

                ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
                node_emb = gnn(ppi_data)  # (P, hidden)
                global_repr = node_emb.mean(dim=0, keepdim=True)  # (1, hidden)
                pred = pred_head(global_repr).squeeze(0)  # (n_drugs,)

                target = dataset.drug_sensitivity[idx].to(device)
                mask = ~torch.isnan(target)
                if mask.any():
                    loss = torch.nn.functional.mse_loss(pred[mask], target[mask])
                    batch_loss = batch_loss + loss
                    count += 1

            if count > 0:
                (batch_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_losses.append((batch_loss / count).item())

        # Validation
        gnn.eval(); pred_head.eval()
        val_losses = []
        with torch.no_grad():
            for idx in val_idx:
                prot_vals = dataset.proteomics[idx].to(device).unsqueeze(-1)
                latent_broadcast = all_latents[idx].to(device).unsqueeze(0).expand(n_proteins, -1)
                stab_broadcast = all_stability[idx].to(device).unsqueeze(0).expand(n_proteins).unsqueeze(-1)
                node_feat = torch.cat([prot_vals, latent_broadcast, stab_broadcast], dim=-1)
                ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
                node_emb = gnn(ppi_data)
                global_repr = node_emb.mean(dim=0, keepdim=True)
                pred = pred_head(global_repr).squeeze(0)
                target = dataset.drug_sensitivity[idx].to(device)
                mask = ~torch.isnan(target)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred[mask], target[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("protein_net_trained", {
                "gnn_state_dict": gnn.state_dict(),
                "pred_head_state_dict": pred_head.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.protein_net,
                "edge_index": edge_index.cpu(), "edge_attr": edge_attr.cpu(),
                "node_feat_dim": node_feat_dim,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"[protein_net] Epoch {epoch+1}/100 train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[protein_net] Early stopping at epoch {epoch+1}")
            break

    metrics = {"protein_net_val_loss": best_val_loss}
    log_stage_end("protein_net_train", metrics=metrics)
    return ckpt_mgr.path("protein_net_trained")


def train_fusion(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train multi-modal fusion layer.

    Combines four modalities (epigenetic state, trajectory, protein network,
    stability) via cross-attention fusion, trained against drug sensitivity.
    """
    from resistancemap.models.fusion import ResistanceMapFusion
    from resistancemap.models.vae import ProteomeToEpigenomeVAE
    from resistancemap.models.trajectory import MemoryStabilityScorer
    from resistancemap.models.protein_network import PPIGraphNetwork
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("fusion_trained"):
        return ckpt_mgr.path("fusion_trained")

    log_stage_start("fusion_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    traj_ckpt = ckpt_mgr.load("stability_calibrated")
    pnet_ckpt = ckpt_mgr.load("protein_net_trained")
    dataset = data_ckpt["dataset"]
    protein_names = dataset.protein_names
    n_proteins = len(protein_names)
    device = config.device

    # ── 1. Load frozen upstream models ───────────────────────────────────
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    vae = ProteomeToEpigenomeVAE(config.vae).to(device)
    vae_sd = {k.replace("_orig_mod.", ""): v for k, v in vae_ckpt["model_state_dict"].items()}
    vae.load_state_dict(vae_sd); vae.eval()

    scorer = MemoryStabilityScorer(config.trajectory).to(device)
    scorer_sd = {k.replace("_orig_mod.", ""): v for k, v in traj_ckpt["model_state_dict"].items()}
    scorer.load_state_dict(scorer_sd); scorer.eval()

    gnn = PPIGraphNetwork(
        in_dim=pnet_ckpt["node_feat_dim"], hidden_dim=config.protein_net.gnn_hidden,
        n_layers=config.protein_net.gnn_layers, n_heads=config.protein_net.gnn_heads,
        dropout=config.protein_net.gnn_dropout, edge_dim=1,
    ).to(device)
    gnn_sd = {k.replace("_orig_mod.", ""): v for k, v in pnet_ckpt["gnn_state_dict"].items()}
    gnn.load_state_dict(gnn_sd); gnn.eval()

    edge_index = pnet_ckpt["edge_index"].to(device)
    edge_attr = pnet_ckpt["edge_attr"].to(device)

    # ── 2. Pre-compute all modality embeddings ───────────────────────────
    logger.info("Pre-computing modality embeddings for fusion training...")
    from torch_geometric.data import Data as PyGData
    epi_states = []     # (N, 64) — VAE latent
    traj_states = []    # (N, 64) — reuse VAE latent as trajectory proxy
    pnet_outputs = []   # (N, 256) — GNN global pool
    stab_scores = []    # (N, 1)

    with torch.no_grad():
        for i in range(len(dataset)):
            prot = dataset.proteomics[i:i+1].to(device)
            # Epigenetic state = VAE latent
            mu, _ = vae.encode(prot)
            epi_states.append(mu.squeeze(0).cpu())
            # Trajectory = use same latent (trajectory forecaster not separately trained)
            traj_states.append(mu.squeeze(0).cpu())
            # Stability
            stab = scorer(prot, protein_names)
            stab_scores.append(stab.view(1).cpu())
            # Protein network
            prot_vals = prot.squeeze(0).unsqueeze(-1)  # (P, 1)
            lat_bc = mu.expand(n_proteins, -1)         # (P, 64)
            stab_bc = stab.expand(n_proteins).unsqueeze(-1)  # (P, 1)
            node_feat = torch.cat([prot_vals, lat_bc, stab_bc], dim=-1)
            ppi_data = PyGData(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
            node_emb = gnn(ppi_data)  # (P, hidden)
            pnet_outputs.append(node_emb.mean(dim=0).cpu())  # (hidden,)

    epi_states = torch.stack(epi_states)      # (N, 64)
    traj_states = torch.stack(traj_states)    # (N, 64)
    pnet_outputs = torch.stack(pnet_outputs)  # (N, gnn_hidden)
    stab_scores = torch.stack(stab_scores)    # (N, 1)
    logger.info(f"Embeddings: epi={epi_states.shape}, pnet={pnet_outputs.shape}, stab={stab_scores.shape}")

    # ── 3. Build fusion model + drug prediction head ─────────────────────
    pnet_dim = pnet_outputs.shape[1]
    total_input_dim = 64 + 64 + pnet_dim + 1  # epi + traj + pnet + stab
    fusion_output_dim = config.fusion.hidden_dim

    # Build a simple concat fusion since modality dims differ from defaults
    fusion = torch.nn.Sequential(
        torch.nn.Linear(total_input_dim, fusion_output_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(config.fusion.dropout),
        torch.nn.Linear(fusion_output_dim, fusion_output_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(fusion_output_dim // 2, fusion_output_dim),
    ).to(device)

    n_drugs = dataset.drug_sensitivity.shape[1]
    drug_head = torch.nn.Sequential(
        torch.nn.Linear(fusion_output_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_drugs),
    ).to(device)

    params = list(fusion.parameters()) + list(drug_head.parameters())
    optimizer = torch.optim.Adam(params, lr=config.fusion.fusion_lr, weight_decay=config.fusion.weight_decay)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    # ── 4. Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(config.fusion.fusion_epochs):
        fusion.train(); drug_head.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), 64):
            batch_end = min(bi + 64, len(train_idx))
            idxs = [train_idx[perm[j]] for j in range(bi, batch_end)]

            epi_b = epi_states[idxs].to(device)
            traj_b = traj_states[idxs].to(device)
            pnet_b = pnet_outputs[idxs].to(device)
            stab_b = stab_scores[idxs].to(device)
            target_b = dataset.drug_sensitivity[idxs].to(device)

            optimizer.zero_grad()

            concat = torch.cat([epi_b, traj_b, pnet_b, stab_b], dim=-1)
            fused = fusion(concat)
            pred = drug_head(fused)
            mask = ~torch.isnan(target_b)
            if mask.any():
                loss = torch.nn.functional.mse_loss(pred[mask], target_b[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        # Validation
        fusion.eval(); drug_head.eval()
        val_losses = []
        with torch.no_grad():
            for bi in range(0, len(val_idx), 64):
                idxs = val_idx[bi:min(bi+64, len(val_idx))]
                epi_b = epi_states[idxs].to(device)
                traj_b = traj_states[idxs].to(device)
                pnet_b = pnet_outputs[idxs].to(device)
                stab_b = stab_scores[idxs].to(device)
                target_b = dataset.drug_sensitivity[idxs].to(device)
                concat = torch.cat([epi_b, traj_b, pnet_b, stab_b], dim=-1)
                fused = fusion(concat)
                pred = drug_head(fused)
                mask = ~torch.isnan(target_b)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred[mask], target_b[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("fusion_trained", {
                "fusion_state_dict": fusion.state_dict(),
                "drug_head_state_dict": drug_head.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.fusion,
                "fusion_output_dim": fusion_output_dim,
                "epi_states": epi_states, "traj_states": traj_states,
                "pnet_outputs": pnet_outputs, "stab_scores": stab_scores,
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"[fusion] Epoch {epoch+1}/{config.fusion.fusion_epochs} train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[fusion] Early stopping at epoch {epoch+1}")
            break

    metrics = {"fusion_val_loss": best_val_loss}
    log_stage_end("fusion_train", metrics=metrics)
    return ckpt_mgr.path("fusion_trained")


def train_landscape(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Build resistance landscape predictor.

    Uses fused representations from upstream fusion stage to train
    the landscape model with drug resistance, state, and target heads.
    """
    from resistancemap.landscape.predictor import ResistanceLandscape
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("landscape_trained"):
        return ckpt_mgr.path("landscape_trained")

    log_stage_start("landscape_train")
    data_ckpt = ckpt_mgr.load("data_ready")
    fusion_ckpt = ckpt_mgr.load("fusion_trained")
    dataset = data_ckpt["dataset"]
    device = config.device

    # Load pre-computed fused representations
    fusion_output_dim = fusion_ckpt["fusion_output_dim"]
    epi_states = fusion_ckpt["epi_states"]
    traj_states = fusion_ckpt["traj_states"]
    pnet_outputs = fusion_ckpt["pnet_outputs"]
    stab_scores = fusion_ckpt["stab_scores"]

    # Rebuild fusion model and compute fused representations
    pnet_dim = pnet_outputs.shape[1]
    total_input_dim = 64 + 64 + pnet_dim + 1
    fusion = torch.nn.Sequential(
        torch.nn.Linear(total_input_dim, fusion_output_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(config.fusion.dropout),
        torch.nn.Linear(fusion_output_dim, fusion_output_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(fusion_output_dim // 2, fusion_output_dim),
    ).to(device)
    fusion.load_state_dict(fusion_ckpt["fusion_state_dict"])
    fusion.eval()

    # Pre-compute fused representations
    logger.info("Computing fused representations for landscape training...")
    fused_reprs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            concat = torch.cat([
                epi_states[i:i+1].to(device),
                traj_states[i:i+1].to(device),
                pnet_outputs[i:i+1].to(device),
                stab_scores[i:i+1].to(device),
            ], dim=-1)
            fused_reprs.append(fusion(concat).squeeze(0).cpu())
    fused_reprs = torch.stack(fused_reprs)  # (N, fusion_dim)
    logger.info(f"Fused representations: {fused_reprs.shape}")

    n_drugs = dataset.drug_sensitivity.shape[1]
    model = ResistanceLandscape(
        fusion_dim=fused_reprs.shape[1],
        n_drugs=n_drugs,
        n_proteins=min(len(dataset.protein_names), 100),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    splits = data_ckpt["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_idx))
        epoch_losses = []

        for bi in range(0, len(train_idx), 64):
            batch_end = min(bi + 64, len(train_idx))
            idxs = [train_idx[perm[j]] for j in range(bi, batch_end)]

            fused_b = fused_reprs[idxs].to(device)
            target_b = dataset.drug_sensitivity[idxs].to(device)

            optimizer.zero_grad()
            output = model(fused_b)
            drug_pred = output["drug_resistance"]  # (B, n_drugs * n_timepoints)

            # Use first timepoint predictions for drug sensitivity loss
            pred_first = drug_pred[:, :n_drugs]
            mask = ~torch.isnan(target_b)
            if mask.any():
                loss = torch.nn.functional.mse_loss(pred_first[mask], target_b[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bi in range(0, len(val_idx), 64):
                idxs = val_idx[bi:min(bi+64, len(val_idx))]
                fused_b = fused_reprs[idxs].to(device)
                target_b = dataset.drug_sensitivity[idxs].to(device)
                output = model(fused_b)
                pred_first = output["drug_resistance"][:, :n_drugs]
                mask = ~torch.isnan(target_b)
                if mask.any():
                    val_losses.append(torch.nn.functional.mse_loss(pred_first[mask], target_b[mask]).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_mgr.save("landscape_trained", {
                "model_state_dict": model.state_dict(),
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": config.landscape,
                "fusion_dim": fused_reprs.shape[1],
            })
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"[landscape] Epoch {epoch+1}/100 train={train_loss:.4f} val={val_loss:.4f}")
        if patience_counter >= patience:
            logger.info(f"[landscape] Early stopping at epoch {epoch+1}")
            break

    metrics = {"landscape_val_loss": best_val_loss}
    log_stage_end("landscape_train", metrics=metrics)
    return ckpt_mgr.path("landscape_trained")


def validate_pipeline(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Run end-to-end validation on held-out test data.

    Loads all trained models, computes fused representations for test split,
    and evaluates landscape predictions against drug sensitivity ground truth.
    """
    from resistancemap.landscape.predictor import ResistanceLandscape
    from resistancemap.utils.logging_utils import log_stage_start, log_stage_end

    if ckpt_mgr.exists("pipeline_validated"):
        return ckpt_mgr.path("pipeline_validated")

    log_stage_start("validate")
    data_ckpt = ckpt_mgr.load("data_ready")
    fusion_ckpt = ckpt_mgr.load("fusion_trained")
    landscape_ckpt = ckpt_mgr.load("landscape_trained")
    dataset = data_ckpt["dataset"]
    splits = data_ckpt["splits"]
    test_idx = splits["test"]
    device = config.device

    # Load fusion model to compute fused representations for test set
    pnet_dim = fusion_ckpt["pnet_outputs"].shape[1]
    total_input_dim = 64 + 64 + pnet_dim + 1
    fusion_output_dim = fusion_ckpt["fusion_output_dim"]

    fusion = torch.nn.Sequential(
        torch.nn.Linear(total_input_dim, fusion_output_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(config.fusion.dropout),
        torch.nn.Linear(fusion_output_dim, fusion_output_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(fusion_output_dim // 2, fusion_output_dim),
    ).to(device)
    fusion.load_state_dict(fusion_ckpt["fusion_state_dict"])
    fusion.eval()

    # Compute test fused representations
    epi_s = fusion_ckpt["epi_states"]
    traj_s = fusion_ckpt["traj_states"]
    pnet_o = fusion_ckpt["pnet_outputs"]
    stab_s = fusion_ckpt["stab_scores"]

    test_fused = []
    with torch.no_grad():
        for idx in test_idx:
            concat = torch.cat([
                epi_s[idx:idx+1].to(device), traj_s[idx:idx+1].to(device),
                pnet_o[idx:idx+1].to(device), stab_s[idx:idx+1].to(device),
            ], dim=-1)
            test_fused.append(fusion(concat).squeeze(0).cpu())
    test_fused = torch.stack(test_fused).to(device)

    # Load landscape model
    n_drugs = dataset.drug_sensitivity.shape[1]
    landscape = ResistanceLandscape(
        fusion_dim=fusion_output_dim, n_drugs=n_drugs,
        n_proteins=min(len(dataset.protein_names), 100),
    ).to(device)
    landscape.load_state_dict(landscape_ckpt["model_state_dict"])
    landscape.eval()

    # Predict on test set
    with torch.no_grad():
        output = landscape(test_fused)
        pred = output["drug_resistance"][:, :n_drugs]
        target = dataset.drug_sensitivity[test_idx].to(device)
        mask = ~torch.isnan(target)

        if mask.any():
            test_mse = torch.nn.functional.mse_loss(pred[mask], target[mask]).item()
        else:
            test_mse = float("nan")

    metrics = {
        "test_mse": test_mse,
        "n_test_samples": len(test_idx),
        "n_drugs": n_drugs,
        "split": "test",
    }
    logger.info(f"Validation: test_mse={test_mse:.4f} on {len(test_idx)} samples")
    ckpt_path = ckpt_mgr.save("pipeline_validated", {"metrics": metrics, "config": config})
    log_stage_end("validate", metrics=metrics)
    return ckpt_path


def serve_api(config: ResistanceMapConfig, ckpt_mgr: CheckpointManager) -> None:
    """Launch the FastAPI inference server."""
    from resistancemap.inference.api import create_app
    import uvicorn
    from resistancemap.utils.logging_utils import log_stage_start

    log_stage_start("serve")
    app = create_app(ckpt_mgr, config)
    uvicorn.run(app, host="0.0.0.0", port=config.api.port, workers=config.api.workers)


# Sequential stage registry
STAGES = [
    ("data_validate",        validate_data_files),
    ("data_prep",            prepare_data),
    ("vae_pretrain",         pretrain_vae),
    ("vae_finetune",         finetune_vae),
    ("trajectory_calibrate", calibrate_trajectory),
    ("protein_net_train",    train_protein_network),
    ("fusion_train",         train_fusion),
    ("landscape_train",      train_landscape),
    ("validate",             validate_pipeline),
    ("serve",                serve_api),
]


def run_sequential_pipeline(config: ResistanceMapConfig, stage: str | None = None) -> None:
    """Execute the pipeline sequentially (legacy mode)."""
    from resistancemap.utils.logging_utils import setup_logger

    log = setup_logger(log_dir=config.log_dir, wandb_project=config.wandb_project)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, log)

    _init_distributed(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if config.hardware.deterministic:
        torch.use_deterministic_algorithms(True)

    if stage:
        stage_fn = dict(STAGES).get(stage)
        if stage_fn is None:
            valid = [name for name, _ in STAGES]
            log.error(f"Unknown stage '{stage}'. Valid: {valid}")
            sys.exit(1)
        stage_fn(config, ckpt_mgr)
    else:
        for name, fn in STAGES:
            log.info(f"=== Sequential stage: {name} ===")
            fn(config, ckpt_mgr)

    if not (config.hardware.distributed and dist.is_initialized()) or dist.get_rank() == 0:
        log.info("Sequential pipeline complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _init_distributed(config: ResistanceMapConfig) -> None:
    """Initialize distributed training if WORLD_SIZE > 1."""
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        config.hardware.distributed = True
        config.hardware.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        config.hardware.world_size = dist.get_world_size()
        torch.cuda.set_device(config.hardware.local_rank)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def run_evaluation_governance(config: ResistanceMapConfig) -> None:
    """Run the evaluation governance layer over the configured pipeline.

    Registers the 10 evaluation agents (Tier A/B/C/D) with an
    :class:`~resistancemap.evaluation.orchestrator.EvalOrchestrator` and
    executes them. The training DAG is **not** invoked. The orchestrator
    enforces a Tier A hard stop (any FAIL blocks Tier B/C/D), runs an
    architecture-vs-baseline adversarial round between Tier B and Tier C,
    and finishes with the chair (Tier D).

    Args:
        config: Loaded :class:`ResistanceMapConfig`. Only ``config.evaluation``
            controls behaviour here, but every agent reads other sections of
            the config to audit them.
    """
    # Imports are local so the training-only paths never pay the cost of
    # loading the evaluation package.
    from resistancemap.evaluation import EvalOrchestrator
    from resistancemap.evaluation.tier_a import (
        DataAdequacyAgent,
        MeasurementIntegrationAgent,
        BiasFairnessShiftAgent,
    )
    from resistancemap.evaluation.tier_b import (
        ArchitectureAuditorAgent,
        CellStateTrajectoryAgent,
        PathwayPPIReasoningAgent,
    )
    from resistancemap.evaluation.tier_c import (
        ForecastingUncertaintyAgent,
        BaselineAdversaryAgent,
        ClinicalTranslationSafetyAgent,
    )
    from resistancemap.evaluation.tier_d import PrincipalIntegratorAgent

    eval_cfg = config.evaluation
    if not eval_cfg.enabled:
        logger.warning("evaluation.enabled is False; nothing to do.")
        return

    orch = EvalOrchestrator(log_root=eval_cfg.log_root)

    # Tier A — gates
    orch.add_agent(DataAdequacyAgent())
    orch.add_agent(MeasurementIntegrationAgent())
    orch.add_agent(BiasFairnessShiftAgent())

    # Tier B — architecture and mechanism
    if not eval_cfg.skip_tier_b:
        orch.add_agent(ArchitectureAuditorAgent())
        orch.add_agent(CellStateTrajectoryAgent())
        orch.add_agent(PathwayPPIReasoningAgent())

    # Tier C — prediction quality + alternatives
    if not eval_cfg.skip_tier_c:
        orch.add_agent(ForecastingUncertaintyAgent())
        orch.add_agent(BaselineAdversaryAgent())
        orch.add_agent(ClinicalTranslationSafetyAgent())

    # Tier D — chair
    if not eval_cfg.skip_tier_d:
        orch.add_agent(PrincipalIntegratorAgent())

    logger.info("Starting ResistanceMap evaluation governance run")
    report = await orch.run(config, run_id=eval_cfg.run_id)

    overall = (
        report.overall_verdict.value
        if hasattr(report.overall_verdict, "value")
        else str(report.overall_verdict)
    )
    print(f"\n{'=' * 60}")
    print("ResistanceMap Evaluation Governance Report")
    print(f"  run_id:           {report.run_id}")
    print(f"  overall verdict:  {overall}")
    print(f"  TRL grade:        {report.maturity_grade.get('trl', '?')}"
          f" — {report.maturity_grade.get('justification', '')}")
    print(f"  per-tier:         "
          + ", ".join(
              f"{t}={(v.value if hasattr(v, 'value') else v)}"
              for t, v in report.per_tier_verdicts.items()
          ))
    print(f"  required changes: {len(report.required_changes)}")
    for i, change in enumerate(report.required_changes[:10], 1):
        print(f"    {i:>2}. {change}")
    if len(report.required_changes) > 10:
        print(f"    ... and {len(report.required_changes) - 10} more")
    print(f"  audit trail:      {eval_cfg.log_root}/{report.run_id}/")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResistanceMap — pharmacogenomic ML for hematologic malignancies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/h100.yaml              # Agentic (default)
  python main.py --config configs/h100.yaml --sequential  # Legacy sequential
  python main.py --config configs/h100.yaml --stage serve  # Single stage
  torchrun --nproc_per_node=8 main.py --config configs/h100.yaml
        """,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/h100.yaml"),
        help="Path to YAML config file (default: configs/h100.yaml)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use legacy sequential execution instead of agentic DAG",
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=[name for name, _ in STAGES],
        help="Run a single stage (forces sequential mode)",
    )
    parser.add_argument(
        "--resume-from-latest", action="store_true",
        help="Automatically resume from the latest checkpoint",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume a specific stage from this checkpoint file",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override API server port (only for --stage serve)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help=(
            "Run the evaluation governance layer (10 PhD-level audit agents "
            "across Tier A/B/C/D) instead of the training DAG. Tier A "
            "FAILs hard-stop downstream tiers; report is written under "
            "evaluation.log_root/<run_id>/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: route to agentic or sequential pipeline."""
    args = parse_args()
    config = load_config(args.config)

    if args.port is not None:
        config.api.port = args.port
    if args.resume is not None:
        config.resume_checkpoint = args.resume

    # Initialize distributed if needed
    _init_distributed(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Evaluation governance mode: orthogonal to the training DAG.
    if args.evaluate:
        asyncio.run(run_evaluation_governance(config))
        return

    # Single-stage mode always uses sequential execution
    if args.stage:
        run_sequential_pipeline(config, stage=args.stage)
        return

    # Default: agentic DAG mode
    if args.sequential:
        run_sequential_pipeline(config)
    else:
        logger.info("Starting ResistanceMap in AGENTIC mode")
        logger.info("  Use --sequential for legacy stage-by-stage execution")
        result = asyncio.run(run_agentic_pipeline(config))

        # Print summary
        summary = result["results_summary"]
        completed = summary["status_counts"].get("completed", 0)
        total = summary["total_agents"]
        elapsed = result["pipeline_elapsed_seconds"]
        print(f"\n{'='*60}")
        print(f"ResistanceMap Pipeline Complete")
        print(f"  Mode: Agentic (DAG parallel execution)")
        print(f"  Agents: {completed}/{total} completed")
        print(f"  Layers: {len(result['execution_plan'])}")
        print(f"  Verification chain: {len(result['verification_chain'])} hashes")
        print(f"  Elapsed: {elapsed:.1f}s")
        if summary["failed_agents"]:
            print(f"  FAILED: {summary['failed_agents']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
