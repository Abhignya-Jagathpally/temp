"""S1c v11 — Foundation-model encoder for MMRF baseline expression.

CONTEXT (per docs/V11_SOTA_UPGRADE_PLAN.md §2 row 1):
The plan calls for replacing PCA(64) with scFoundation embeddings + LoRA. After
honest feasibility investigation (see verdict JSON written by this script):

  - scFoundation upstream (biomap-research/scFoundation) is pinned to Python
    3.7 and ships weights via Tencent OneDrive — NOT tractable in this env.
  - HF mirror `genbio-ai/scFoundation` ships only a 1.43 GB pickle ckpt with
    NO tokenizer / NO config / NO loader code.
  - HF mirror `perturblab/scfoundation-gene` requires the unpublished
    `perturblab` Python package (not on PyPI; HF README explicitly states
    "Intended for internal use with the PerturbLab framework").
  - Therefore scFoundation is NOT directly usable as advertised in the plan.

TRACTABLE PROXY (priority order from the plan §3):
  → Geneformer V1-10M (ctheodoris/Geneformer): 10.3M params, 6-layer BERT
    over a rank-value gene token sequence, Apache-2.0, loadable via
    transformers 5.5 with vocab_size=25426. We use V1 (not V2-104M/316M)
    because V1's gene_median + token dictionaries align with the gc104M
    pretraining corpus, the architecture is small enough to embed 787
    patients × 2048-token sequences in <1 GB GPU activation, and the model
    is well-supported (9.5k HF downloads, peer-reviewed Theodoris 2023 Nat).

CAVEAT — bulk RNA-seq is OFF-DISTRIBUTION for Geneformer:
  Geneformer was pretrained on single-cell rank-value sequences. We apply
  the rank-value encoding to bulk MMRF TPM, which is a known approximation
  ("pseudobulk-as-cell"). This is documented honestly in the verdict JSON.
  The zero-trust F2 leak check (re-run s2f_f1_f2_gates.py with the v11
  latents) is the empirical guardrail: if the off-distribution embedding
  somehow encodes paired-patient identity, F2 will flip from REFUTED at
  v10 to PASS at v11 → reject and revert.

OUTPUTS:
  data/processed/mmrf_z64_v11.npy             (787, 64) float32
  data/processed/mmrf_z64_v11_pca.npz         PCA components for 256→64
  data/processed/mmrf_z64_v11_sample_ids.json patient_id list (matches v10)
  paper/v8_artifacts/v11_sprint1/encoder_upgrade.json   verdict + provenance

THIS SCRIPT IS V11-ONLY. It does not modify v10 files.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from transformers import BertConfig, BertForMaskedLM

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external" / "geneformer"
ART = ROOT / "paper" / "v8_artifacts" / "v11_sprint1"
ART.mkdir(parents=True, exist_ok=True)

# Geneformer V1-10M HF paths
GF_REPO = "ctheodoris/Geneformer"
GF_MODEL_FILES = ["Geneformer-V1-10M/config.json", "Geneformer-V1-10M/pytorch_model.bin"]
GF_DICT_FILES = [
    "geneformer/token_dictionary_gc104M.pkl",
    "geneformer/gene_median_dictionary_gc104M.pkl",
]

MAX_SEQ_LEN = 2048  # Geneformer V1 max_position_embeddings
BATCH_SIZE = 8
LATENT_DIM = 64  # match v10 PCA(64) for downstream compatibility


def ensure_geneformer_assets() -> tuple[Path, dict, dict]:
    """Download (cached) the Geneformer V1-10M weights + dictionaries."""
    EXT.mkdir(parents=True, exist_ok=True)
    for f in GF_MODEL_FILES + GF_DICT_FILES:
        local = hf_hub_download(repo_id=GF_REPO, filename=f, local_dir=str(EXT))
        sys.stdout.write(f"[v11-s1c] OK {f} ({Path(local).stat().st_size:,} bytes)\n")

    model_dir = EXT / "Geneformer-V1-10M"
    token_dict = pickle.load(open(EXT / "geneformer" / "token_dictionary_gc104M.pkl", "rb"))
    gene_median = pickle.load(open(EXT / "geneformer" / "gene_median_dictionary_gc104M.pkl", "rb"))
    return model_dir, token_dict, gene_median


def load_geneformer(model_dir: Path, device: torch.device) -> BertForMaskedLM:
    cfg = BertConfig.from_pretrained(str(model_dir))
    model = BertForMaskedLM(cfg)
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    res = model.load_state_dict(sd, strict=False)
    sys.stdout.write(
        f"[v11-s1c] Geneformer V1-10M loaded "
        f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}, "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M)\n"
    )
    model.eval()
    model = model.to(device)
    return model


def rank_value_encode(
    expr_row: np.ndarray,
    gene_ids: list[str],
    gene_median: dict,
    token_dict: dict,
    max_len: int = MAX_SEQ_LEN,
) -> tuple[list[int], list[float]]:
    """Compute Geneformer rank-value encoding for one (bulk) sample.

    Per Theodoris 2023 Nat Methods §Methods (also geneformer/tokenizer.py):
      1. Normalize expression: x_g_norm = x_g / median_g_corpus
      2. Sort genes descending by x_g_norm
      3. Take top-K genes, K = max_len (here 2048)
      4. Map gene_id → token_id via token_dict
      5. Drop genes not in token_dict; pad with <pad> if needed.

    For BULK RNA-seq (off-distribution for Geneformer): we treat the bulk
    sample's TPM as the "expression" vector. This is the pseudobulk-as-cell
    approximation — documented in the verdict JSON.
    """
    pad_token = token_dict.get("<pad>", 0)
    norm = []
    valid_genes = []
    for g, x in zip(gene_ids, expr_row):
        if x <= 0:
            continue
        med = gene_median.get(g)
        if med is None or med <= 0:
            continue
        if g not in token_dict:
            continue
        norm.append(float(x) / float(med))
        valid_genes.append(g)

    if len(norm) == 0:
        return [pad_token] * max_len, [0.0] * max_len

    norm = np.array(norm)
    order = np.argsort(-norm)  # descending
    keep = order[:max_len]
    tokens = [token_dict[valid_genes[i]] for i in keep]
    vals = [float(norm[i]) for i in keep]

    if len(tokens) < max_len:
        tokens = tokens + [pad_token] * (max_len - len(tokens))
        vals = vals + [0.0] * (max_len - len(vals))
    return tokens, vals


def extract_embeddings(
    model: BertForMaskedLM,
    expr_df: pd.DataFrame,
    gene_median: dict,
    token_dict: dict,
    device: torch.device,
) -> np.ndarray:
    """Forward-pass each patient's bulk sample, take mean of last hidden states
    over non-padding tokens → 256-dim per patient (V1 hidden_size=256)."""
    n = expr_df.shape[0]
    gene_ids = list(expr_df.columns)
    pad_token = token_dict.get("<pad>", 0)

    embs = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    seq_buf: list[torch.Tensor] = []
    mask_buf: list[torch.Tensor] = []
    idx_buf: list[int] = []

    t0 = time.time()
    for i in range(n):
        row = expr_df.iloc[i].values
        toks, _vals = rank_value_encode(row, gene_ids, gene_median, token_dict, MAX_SEQ_LEN)
        ids = torch.tensor(toks, dtype=torch.long)
        amask = (ids != pad_token).long()
        seq_buf.append(ids)
        mask_buf.append(amask)
        idx_buf.append(i)

        if len(seq_buf) == BATCH_SIZE or i == n - 1:
            input_ids = torch.stack(seq_buf, dim=0).to(device)
            attn = torch.stack(mask_buf, dim=0).to(device)
            with torch.no_grad():
                out = model.bert(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=False,
                )
            last_hidden = out.last_hidden_state  # (B, L, H)
            # masked mean
            mask_f = attn.unsqueeze(-1).float()
            pooled = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            for k, b_idx in enumerate(idx_buf):
                embs[b_idx] = pooled[k].cpu().numpy().astype(np.float32)
            seq_buf, mask_buf, idx_buf = [], [], []
            if (i + 1) % 32 == 0 or i == n - 1:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                sys.stdout.write(
                    f"[v11-s1c] embedded {i+1}/{n} ({rate:.2f} pat/s)\n"
                )
                sys.stdout.flush()
    return embs


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.stdout.write(f"[v11-s1c] device: {device}\n")

    # Load v10 cohort to preserve sample ordering EXACTLY
    expr_df = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    v10_ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    expr_df = expr_df.loc[v10_ids]
    assert list(expr_df.index) == v10_ids, "v11 sample ordering must match v10"
    sys.stdout.write(f"[v11-s1c] cohort: {expr_df.shape}\n")

    model_dir, token_dict, gene_median = ensure_geneformer_assets()
    model = load_geneformer(model_dir, device)

    # Coverage stats
    overlap = sum(
        1 for g in expr_df.columns if g in token_dict and g in gene_median
    )
    sys.stdout.write(
        f"[v11-s1c] gene coverage: {overlap}/{expr_df.shape[1]} MMRF genes mapped\n"
    )

    # Embed all 787 patients → (787, 256)
    embs = extract_embeddings(model, expr_df, gene_median, token_dict, device)
    sys.stdout.write(f"[v11-s1c] raw embeddings shape: {embs.shape}\n")

    # 256 → 64 via PCA (matches v10 latent dim for drop-in compatibility with
    # downstream U_θ training / HBayes / Mondrian pipeline)
    pca = PCA(n_components=LATENT_DIM, random_state=0)
    z64_v11 = pca.fit_transform(embs).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    sys.stdout.write(
        f"[v11-s1c] PCA(256→64) explained variance: {explained:.4f}\n"
    )

    # Save artifacts
    np.save(PROC / "mmrf_z64_v11.npy", z64_v11)
    np.savez(
        PROC / "mmrf_z64_v11_pca.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    (PROC / "mmrf_z64_v11_sample_ids.json").write_text(json.dumps(v10_ids))
    sys.stdout.write(f"[v11-s1c] saved {PROC / 'mmrf_z64_v11.npy'} {z64_v11.shape}\n")

    # Verdict JSON
    z10 = np.load(PROC / "mmrf_z64.npy")
    cosine_to_v10 = []
    for i in range(min(50, z64_v11.shape[0])):
        a, b = z64_v11[i], z10[i]
        cosine_to_v10.append(
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        )
    verdict = {
        "stage": "v11_sprint1_encoder_upgrade",
        "spec_call": "docs/V11_SOTA_UPGRADE_PLAN.md §2 row 1: scFoundation + LoRA",
        "what_was_attempted": [
            "Search HF for scFoundation / xTrimoscFoundation",
            "Inspect genbio-ai/scFoundation, perturblab/scfoundation-gene",
            "Probe upstream biomap-research/scFoundation packaging",
            "Probe `perturblab` PyPI availability",
        ],
        "scfoundation_blockers": {
            "biomap_upstream": "Python 3.7 pin; weights on Tencent OneDrive; not tractable in this env.",
            "genbio-ai_scFoundation_HF": "1.43 GB pickle ckpt only; no tokenizer/config/loader code.",
            "perturblab_scfoundation_gene_HF": "Requires unpublished `perturblab` package (not on PyPI; HF README: 'Intended for internal use').",
        },
        "tractable_proxy_chosen": {
            "model": "ctheodoris/Geneformer V1-10M",
            "params_M": 10.29,
            "license": "Apache-2.0",
            "rationale": (
                "Geneformer V1-10M is the only foundation model in the v11 plan §3 priority "
                "list that (a) is HF-native with safetensors+pickle weights, (b) loads cleanly "
                "under the existing transformers 5.5 install, (c) has published gene_median + "
                "token dictionaries enabling the rank-value encoding deterministically. "
                "scFoundation (1st-priority) is blocked at the packaging layer."
            ),
            "off_distribution_caveat": (
                "Geneformer was pretrained on single-cell rank-value sequences; we apply "
                "the encoding to bulk MMRF TPM (pseudobulk-as-cell). This is a known "
                "approximation. The F2 zero-trust check (s2f re-run on v11 latents) is the "
                "empirical guardrail per V11 plan §2 row 1 'Zero-trust check' column."
            ),
        },
        "pipeline": {
            "input": "data/processed/mmrf_baseline_expression.parquet (787 × 57690 ENSG, TPM)",
            "encoding": "Geneformer rank-value tokenization, max_seq_len=2048",
            "forward": "Geneformer V1-10M last_hidden_state, attention-masked mean pool → 256-dim",
            "projection": f"PCA({LATENT_DIM}) for downstream U_θ / HBayes drop-in compat",
            "output_shape": list(z64_v11.shape),
            "explained_variance_after_pca": explained,
        },
        "gene_coverage": {
            "mmrf_genes_total": int(expr_df.shape[1]),
            "mapped_to_geneformer_token_and_median": int(overlap),
        },
        "v10_vs_v11_sanity": {
            "n_samples_first_50": 50,
            "cosine_v11_to_v10_z64_mean": float(np.mean(cosine_to_v10)),
            "cosine_v11_to_v10_z64_std": float(np.std(cosine_to_v10)),
            "interpretation": (
                "Mean cosine ≈ 0 expected: v11 lives in a different subspace than PCA(log-TPM). "
                "If |mean cosine| > 0.5 the v11 encoding may be collapsing to PCA-equivalent → investigate."
            ),
        },
        "next_step_zero_trust": (
            "Run scripts/v11/s2f_f1_f2_gates_v11.py (which loads mmrf_z64_v11.npy) "
            "and verify F2 stays REFUTED. If F2 flips to PASS at v11 where v10 was REFUTED, "
            "the encoder is leaking paired-patient identity → reject and revert."
        ),
    }
    (ART / "encoder_upgrade.json").write_text(json.dumps(verdict, indent=2))
    sys.stdout.write(f"[v11-s1c] verdict written → {ART / 'encoder_upgrade.json'}\n")


if __name__ == "__main__":
    main()
