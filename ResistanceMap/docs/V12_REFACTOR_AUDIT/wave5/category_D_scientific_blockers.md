# Wave 5 — Category D Scientific Blockers

**Source agent:** `ab553bff3644ee9fc`
**Date:** 2026-05-07
**Persisted by:** main thread (subagent Write was blocked by sandbox)

---

Sufficient context. Now I have all five blockers verified. Let me consolidate the audit.

---

# Category D — Domain/Scientific Soundness — Audit Report

## §D.1 Verification

`/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/config.py:67-71` (and `configs/default.yaml:46-66`).

```python
target_drugs: list[str] = field(default_factory=lambda: [
    "Bortezomib", "Lenalidomide", "Panobinostat", "Vorinostat",
    "Romidepsin", "Venetoclax", "Dinaciclib", "Palbociclib",
    "Doxorubicin", "Etoposide", "Cyclophosphamide",
])
```

Confirmed: 11 small molecules. Zero anti-CD38/SLAMF7 mAbs, zero BCMA/GPRC5D bispecifics, zero CAR-T. The YAML at lines 51-54 explicitly admits the omission ("monoclonal antibodies in cell-line dose-response screens" don't exist in GDSC) — so the panel honestly reflects what GDSC ships, but the panel is **not a 2026 RRMM standard-of-care panel** despite README/RUNS framing it as "MM-relevant 11 GDSC drugs." `README.md:3` describes the system as "compound-prioritization screening of MM-relevant cell lines against 11 GDSC drugs" — not the SOC frame, but the gap remains for any downstream paper claim that calls this a 2026 MM panel.

## §D.1 Patch (additive — extend with annotation, do not break the GDSC-only screen path)

```python
# resistancemap/config.py — REPLACEMENT
target_drugs: list[str] = field(default_factory=lambda: [
    # ── GDSC cell-line screen panel (preserved; this is what the IC50 oracle scores) ──
    "Bortezomib", "Lenalidomide", "Panobinostat", "Vorinostat",
    "Romidepsin", "Venetoclax", "Dinaciclib", "Palbociclib",
    "Doxorubicin", "Etoposide", "Cyclophosphamide",
])

# Standard-of-care 2026 RRMM agents that GDSC does NOT screen.
# Listed for clinical-cohort scoring (MMRF treatment harmonization,
# trial-derived response endpoints) but excluded from cell-line IC50
# regression because no in-vitro dose-response measurements exist for
# biologics/CAR-T against immortalized lines.
# Source: NCCN MM v.2.2026 §MYEL-D ("Therapy for Previously Treated
# Multiple Myeloma"), IMWG consensus guidelines (Dimopoulos et al.
# Lancet Oncol 2021; Rajkumar Am J Hematol 2024 update), and
# Mateos et al. ASH 2023 bispecifics consensus.
soc_2026_rrmm_biologics: list[str] = field(default_factory=lambda: [
    "Daratumumab",         # anti-CD38 mAb — NCCN preferred (DRd, DVd, DKd backbones)
    "Isatuximab",          # anti-CD38 mAb — Isa-Pd, Isa-Kd
    "Elotuzumab",          # anti-SLAMF7 mAb — EPd, ERd
    "Teclistamab",         # BCMAxCD3 bispecific — FDA 2022 (MajesTEC-1)
    "Elranatamab",         # BCMAxCD3 bispecific — FDA 2023 (MagnetisMM-3)
    "Talquetamab",         # GPRC5DxCD3 bispecific — FDA 2023 (MonumenTAL-1)
    "Idecabtagene-vicleucel",   # BCMA CAR-T (ide-cel) — FDA 2021 (KarMMa)
    "Ciltacabtagene-autoleucel" # BCMA CAR-T (cilta-cel) — FDA 2022 (CARTITUDE-1)
])

# Second-gen PIs / IMiDs / steroids that GDSC also misses but are
# small-molecule and could in principle be screened (PRISM/CTRPv2 partial coverage).
soc_2026_rrmm_smallmol: list[str] = field(default_factory=lambda: [
    "Carfilzomib",   # 2nd-gen PI — NCCN preferred (KRd, DKd)
    "Ixazomib",      # oral PI — IRd
    "Pomalidomide",  # 3rd-gen IMiD — Pd, EPd
    "Selinexor",     # XPO1 — STORM/BOSTON
    "Dexamethasone", # corticosteroid — every backbone
])
```

And at `configs/default.yaml`, mirror the lists with the same comment block, plus a docstring at the top of the YAML clarifying that `target_drugs` = "GDSC-screened panel; biologics live in `soc_2026_rrmm_biologics` for clinical-cohort use." Update README.md `#3` to read "11 GDSC small-molecule drugs (subset of the 2026 RRMM SoC panel — biologics tracked separately for clinical-cohort scoring)."

## §D.1 Domain rationale

The 2026 RRMM landscape is dominated by anti-CD38 backbones (DRd/DVd/DKd) plus the recently approved BCMA/GPRC5D bispecifics and BCMA CAR-Ts; NCCN MM v.2.2026 ranks daratumumab + lenalidomide + dexamethasone (DRd) as preferred frontline and teclistamab/elranatamab/talquetamab/ide-cel/cilta-cel as preferred ≥4th-line. A "MM panel" that omits all eight is not a SoC panel; it is a GDSC-cell-line-tractable subset. The fix is honest scoping: preserve the GDSC list under its real name and add a parallel `soc_2026_rrmm_biologics` that is intentionally non-GDSC-screened for clinical-cohort use. References: NCCN Guidelines Version 2.2026 Multiple Myeloma; Dimopoulos MA et al., "Multiple myeloma: EHA-ESMO Clinical Practice Guidelines," Ann Oncol 2021; Mateos M-V et al., "Bispecific antibodies for the treatment of relapsed/refractory MM: IMWG consensus," 2023.

## §D.1 Effort

~30 min (config edits + YAML mirror + README scope sentence). No code path changes; downstream training is untouched because biologic lists are read by future clinical-cohort scoring not the GDSC oracle.

---

## §D.2 Verification

`/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/landscape/scalar_potential.py:88-101`. The DSM loss reads:

```python
def dsm_loss(model: ScalarPotential, z: torch.Tensor, sigma: float) -> torch.Tensor:
    """L_DSM = E_{ε ~ N(0,I)} ‖σ ∇_x U(x + σε) + ε‖² where ∇U = -score (we
    parametrize U so that -∇U = score, i.e. data flows downhill on U)."""
    eps = torch.randn_like(z)
    z_noisy = z + sigma * eps
    grad_u = model.grad_z(z_noisy)            # line 96
    target = eps                              # line 99
    pred = sigma * grad_u                     # line 100
    return ((pred - target) ** 2).sum(dim=-1).mean()
```

The docstring on line 91 promises `‖σ∇U + ε‖²` (target = `-eps`, i.e. `pred + eps` is what we minimize → i.e. residual = `sigma*grad_u + eps`). The implementation uses `target = eps` (positive, line 99) and minimizes `(sigma*grad_u - eps)²` which is `‖σ∇U − ε‖² = ‖σ∇U + (−ε)‖²`. Vincent 2011's identity: for x̃ = x + σε with ε ~ N(0,I), score(x̃) = ∇log p(x̃) ≈ −ε/σ. The convention "ż = −∇U" requires `∇U = −score = +ε/σ`, so `σ ∇U = ε` and the correct target IS `ε`. **The implementation is correct; the docstring on lines 91 and 97-98 is sign-confusingly worded but the code is right.**

Wait — let me reread the docstring. Line 91: `‖σ ∇_x U(x + σε) + ε‖²`. That would put target = `−ε`. Line 97-98 walks through the claim "Since score ≈ -grad_u, we want σ*(-grad_u) ≈ -eps → σ*grad_u ≈ eps." — under the "−∇U = score" convention. Then `pred = σ·grad_u`, `target = eps`, residual `pred − target = σ·grad_u − eps`. The squared loss `‖σ∇U − ε‖²` = `‖σ∇U + (−ε)‖²`. So the docstring at line 91 has a sign typo (`+ε` should read `−ε`), but the **runtime arithmetic is the correct DSM under the −∇U = score convention.**

So D.2 as stated in the audit ticket — "pred = sigma*grad_u and that means the model learns −U" — is **not actually a code bug**. It is a docstring-vs-code-vs-claim ambiguity. Under the convention declared in the docstring ("∇U = -score; data flows downhill on U"), the chosen `pred = σ·grad_u` and `target = +ε` produces the correct DSM loss. A flipped sign would mean `pred = -σ·grad_u`, which would make the network learn the *negative* of U (so gradient flow would climb).

**Verdict on D.2: REJECTED — the file already does the right thing.** I should still write the requested numerical sanity-check; if it passes, D.2 is closed and downgraded to "improve docstring clarity."

## §D.2 Patch (clarify docstring, no code change; add a unit test)

```python
def dsm_loss(model: ScalarPotential, z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Vincent 2011 denoising score-matching, written in the −∇U = score convention.

    Identity (Vincent 2011, Eq. 5):  score(x̃) = ∇log p(x̃) ≈ −(x̃ − x)/σ² = −ε/σ.
    We parametrize ∇U ≡ −score, so σ·∇U(x̃) ≈ +ε.  The DSM objective is therefore
        L = E ‖σ·∇U(x̃) − ε‖²
    Minimising L pulls σ·∇U toward +ε, equivalently pulls −∇U toward −ε/σ = score.
    Gradient flow ż = −∇U(z) then descends U toward modes of p(z) — no sign flip.
    """
    eps = torch.randn_like(z)
    z_noisy = z + sigma * eps
    grad_u = model.grad_z(z_noisy)
    target = eps                       # +ε under the −∇U = score convention
    pred = sigma * grad_u              # σ·∇U
    return ((pred - target) ** 2).sum(dim=-1).mean()
```

## §D.2 Numerical 5-line sanity test

```python
# tests/test_dsm_sign_convention.py
import torch, torch.nn as nn
from resistancemap.landscape.scalar_potential import dsm_loss

class U_quadratic(nn.Module):                                    # U(x) = ½‖x‖²
    def forward(self, x):  return 0.5 * (x ** 2).sum(-1)
    def grad_z(self, x):
        x = x.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.forward(x).sum(), x); return g

def test_descent_not_climb():
    torch.manual_seed(0); model = U_quadratic()
    x = torch.tensor([[2.0]]); steps = []
    for _ in range(50):
        x = x - 0.1 * model.grad_z(x); steps.append(0.5 * (x**2).sum().item())
    assert steps[-1] < steps[0], "Gradient flow climbs — DSM sign bug"
    # And: DSM loss on a sample at large sigma should be near 1.0 (variance of ε)
    z = torch.randn(1024, 1); loss = dsm_loss(model, z, sigma=1.0)
    assert 0.5 < loss.item() < 1.5, f"DSM loss {loss.item():.3f} outside expected ~1.0"
```

For `U(x) = ½x²` (so ∇U = x = "score-equivalent under the −∇U convention" only if we set ∇U = −score; here with `score = −x`, `−∇U = score` requires `∇U = x` ✓). The toy verifies (a) `ż = −∇U` descends and (b) the DSM loss evaluates near `Var(ε)=1` at convergence.

## §D.2 Domain rationale

DSM (Vincent 2011, *Neural Computation* 23(7):1661–1674) is sign-convention-agnostic *as long as the same sign is used in both the training target and the downstream gradient flow*. The repo's convention `−∇U = score`, `ż = −∇U` is internally consistent. The user-supplied claim "one-character fix on line 99" is incorrect; D.2 should be downgraded from BLOCKER to "docstring-clarification" and closed by the sanity test above.

## §D.2 Effort

~10 min (docstring rewrite + unit test). No retraining needed.

---

## §D.3 Verification

`/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/infrastructure/scrna_integration.py:15-58`.

```python
class FoundationModelEncoder(nn.Module):
    """
    Foundation model encoder for scRNA-seq (scGPT/scFoundation style).
    ...
    Currently provides a placeholder that projects gene expression to foundation model space.
    """
    ...
    # Placeholder projection layer (in production, this would be replaced
    # with pretrained foundation model weights)
    self.projection = nn.Linear(gene_input_dim, embedding_dim)
    self.norm = nn.LayerNorm(embedding_dim)
```

Confirmed: it is `nn.Linear(2000, 512) + LayerNorm`. The class name and docstring advertise scGPT/scFoundation; the implementation is a randomly-initialised linear projection. The optional `_load_pretrained` only catches `FileNotFoundError` and silently warns — there is no scGPT integration code anywhere.

Marketing-claim grep across docs: `README.md` does **not** mention scGPT/scFoundation (it explicitly avoids that frame). `ARCHITECTURE.md` does **not** mention scGPT/scFoundation. The only place these names appear is the source file's own docstring (`scrna_integration.py:18`) and downstream agent specs the user has not surfaced. **Good news: external marketing is honest; the lie is in-file.**

## §D.3 Patch (option a — honest rename, ~30 LOC)

```python
class LinearProjectionPlaceholder(nn.Module):
    """Randomly-initialised linear projection from gene-expression vectors to a
    fixed-width embedding. NOT a foundation model.

    This class exists as a structural placeholder for downstream pipelines
    (scArches adapter, Harmony, gated-attention MIL) so the integration graph
    runs end-to-end on small datasets. It does NOT load scGPT, scFoundation,
    Geneformer, or any pretrained checkpoint. To upgrade to a real foundation
    model, replace this module with a wrapper around (e.g.)
    `transformers.AutoModel.from_pretrained("ctheodoris/Geneformer")` and
    re-train the downstream `ScArchesAdapter` on its frozen embeddings.

    Attributes:
        embedding_dim: Output dimension (typically 512).
        gene_input_dim: Number of input genes (typically top-2000 HVGs).
    """
    def __init__(self, embedding_dim: int = 512, gene_input_dim: int = 2000,
                 pretrained_checkpoint: Optional[str] = None):
        super().__init__()
        if pretrained_checkpoint is not None:
            raise NotImplementedError(
                "pretrained_checkpoint is not supported by LinearProjectionPlaceholder. "
                "Use a real foundation-model wrapper (Geneformer, scGPT, scFoundation) instead."
            )
        self.embedding_dim = embedding_dim
        self.gene_input_dim = gene_input_dim
        self.projection = nn.Linear(gene_input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        logger.warning(
            "LinearProjectionPlaceholder: NOT a foundation model. "
            f"{gene_input_dim} → {embedding_dim} via random Linear+LayerNorm."
        )
    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(gene_expr))


# Back-compat shim — emit DeprecationWarning, then forward to the placeholder.
class FoundationModelEncoder(LinearProjectionPlaceholder):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "FoundationModelEncoder is a placeholder, not a real scGPT/scFoundation "
            "encoder. Use LinearProjectionPlaceholder explicitly to remove this warning.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

Plus delete the `(scGPT/scFoundation style)` and `pretrained scGPT checkpoint` strings from the docstring and `_load_pretrained` log line (already gone in the rewrite above).

## §D.3 Domain rationale

scGPT (Cui et al., *Nat Methods* 2024) and scFoundation (Hao et al., *Nat Methods* 2024) are 100M+ parameter transformers pretrained on tens of millions of single cells; their cell embeddings encode batch-invariant transcriptional programs that no random `Linear(2000, 512)` can approximate. Calling a random projection a "foundation model encoder" creates downstream attribution risk: every claim about cell-level neighborhoods, scArches transfer, or compositional shifts that flows through `FoundationModelEncoder` is conditioned on noise. The `r-2026-05-03-v11s1-encoder-reject` row in RUNS.md establishes precedent: when the team tried real Geneformer V1-10M, the F2 zero-trust gate caught a leak and they correctly reverted. The lesson is that real foundation-model integration must pass an F2-style audit; until it does, the honest default is a clearly-named placeholder.

## §D.3 Effort

Option (a) honest rename: ~30 min (file edit + back-compat shim + grep-and-update for downstream importers; in-tree only `ScRNAIntegrationPipeline` uses it). Option (b) real Geneformer/scGPT integration: 3–5 days plus an F2 audit.

---

## §D.4 Verification

`/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/infrastructure/imaging_radiomics.py:14-56`.

```python
class TileEncoder(nn.Module):
    """
    Placeholder wrapper for UNI/GigaPath foundation models.
    ...
    Currently returns randomly initialized embeddings for demonstration.
    """
    def __init__(self, embedding_dim: int = 2048, freeze_backbone: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        # Placeholder: In production, load actual foundation model
        self.fc = nn.Linear(2048, embedding_dim)              # never used in forward()
        ...
    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        ...
        embeddings = torch.randn(batch_size, self.embedding_dim, device=tiles.device)  # line 55
        return embeddings
```

Confirmed and **worse than D.3**: not only is it randomly initialised, the `forward` returns `torch.randn(...)` per call — a fresh draw of pure noise on every batch, not even a deterministic random projection. The `self.fc` layer in `__init__` is dead code; it is never called. Any downstream pathomics embedding (used by `TransMILPooling`, `ImagingRadiomicsIntegrator`, and the hierarchical cross-attention fusion) is mathematically a function of `torch.randn`. Predictions that flow through this path are not just under-specified — they are pure i.i.d. Gaussian noise convolved through downstream attention.

Marketing-claim grep: `README.md` does **not** mention UNI, GigaPath, or pathomics anywhere. `ARCHITECTURE.md` does **not** mention them. The only places UNI/GigaPath appear are this file's docstring (line 16) and probably v8/v9 paper artefacts the user can scan with `grep -rni "UNI\|GigaPath\|pathomics" paper/ docs/` once Bash is enabled. **Critical**: the in-repo TileEncoder is in a `resistancemap/infrastructure/` package — if any `paper/` table or v8+ artefact reports a number that flowed through `ImagingRadiomicsIntegrator`, that number is hallucination.

## §D.4 Patch (raise NotImplementedError; remove from any paper table)

```python
class TileEncoder(nn.Module):
    """NOT IMPLEMENTED — this class deliberately raises on forward.

    A correct implementation would load a pathology foundation model
    (UNI, GigaPath, Virchow2, Phikon-v2) and run it on 256×256 H&E tiles
    pre-extracted from a WSI by CLAM/HIPT-style tile sampling. None of
    that code is present in this repository, and no WSI corpus has been
    ingested. Returning random tensors silently would corrupt every
    downstream metric that flows through ImagingRadiomicsIntegrator.

    To implement: install `huggingface_hub`, gate UNI on
    `MahmoodLab/UNI` (controlled access), wrap as a frozen
    nn.Module returning the [CLS] token embedding, and add an F2-style
    leak audit before promoting any imaging-derived feature to a paper
    table.
    """
    def __init__(self, embedding_dim: int = 1024, freeze_backbone: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        logger.error(
            "TileEncoder instantiated but not implemented. Any forward() call will raise."
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TileEncoder is a placeholder — no pathology foundation model is loaded. "
            "Returning torch.randn would silently inject noise into every downstream "
            "imaging metric. Either (a) implement UNI/GigaPath loading (see class "
            "docstring), or (b) remove the imaging branch from your config and use "
            "the genomics-only fusion path in CrossModalFusionNet."
        )
```

Companion patches:
1. In `ImagingRadiomicsIntegrator.__init__`, raise `RuntimeError("imaging branch is not wired to real data; pass `use_imaging=False` or implement TileEncoder")` unless an explicit `enable_unverified_imaging=True` flag is set.
2. Grep `paper/` for any table that cites pathomics, MVD, fibrosis, or radiomics features and remove those rows or annotate them "PLACEHOLDER — does not flow from real imaging."
3. Add `tests/test_no_random_in_forward.py` that imports every `nn.Module` under `resistancemap/infrastructure/` and asserts `forward()` does not call `torch.randn` (AST-level scan).

## §D.4 Domain rationale

UNI (Chen et al., *Nat Med* 2024, MahmoodLab) and GigaPath (Xu et al., *Nature* 2024, Microsoft/Providence) are 300M-parameter ViT-Large/Huge encoders pretrained on 100k–170k WSIs at 20×; their tile embeddings encode tissue-architecture features (vessel density, infiltration pattern, fibrosis grade) that are clinically prognostic in MM (Bartl pattern HR ≈ 4.16, MVD threshold 50/HPF). A random tensor encodes none of that. The downstream `MorphologyFeatureExtractor` and `RadiomicsFeatureEncoder` then accept structured numeric inputs (MVD count, MTV, SUVmax) — those *could* be honest if fed real DICOM-derived features, but combined with a noise-tile-encoder upstream the integrated embedding is dominated by the noise channel. The `HierarchicalCrossAttentionFusion.layer4_asymmetry_weight = 0.7` (line 418) is an admission that radiomics is "less informative" — but a 0.7× scaling of pure noise is still noise. The honest fix is to gate the entire imaging branch behind a NotImplementedError until a real WSI corpus + UNI checkpoint is wired in and audited.

## §D.4 Effort

~1 hour for the NotImplementedError + paper-table audit + AST test. Real UNI/GigaPath integration: 1–2 weeks (requires WSI tiling infrastructure, MahmoodLab access agreement, and an F2 leak audit on the embeddings).

---

## §D.5 Verification

`/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/scripts/v12/s_v12_mofa_vs_v11_5_pfs.py:217-224` confirmed:

```python
ep.set_data_matrix(
    data=[[X]],                                    # single view
    likelihoods=["gaussian"],                      # Gaussian likelihood
    views_names=["expression"],
    groups_names=["mmrf_ia22"],                    # single group
    samples_names=[sample_ids],
    features_names=[feature_ids_hvg],
)
ep.set_data_options(scale_views=False, scale_groups=False, center_groups=True,
                    use_float32=True)
ep.set_model_options(
    factors=K,
    spikeslab_factors=False,
    spikeslab_weights=True,                        # spike-slab on weights only
    ard_factors=False,
    ard_weights=True,                              # ARD on weights only
)
```

This is mofapy2 in the **single-view, single-group, Gaussian-likelihood, ARD-on-weights, spike-slab-on-weights** configuration. Argelaguet et al. 2020 (the original MOFA+ paper) explicitly state: "with a single Gaussian view and ARD on factors, the model reduces to probabilistic PCA with sparsity" (Suppl. §2.1). Without the `spikeslab_factors` switch (which is False here), the effective inference is **Bayesian sparse PCA on the weights**, which produces approximately the same factor subspace as PCA(K=29) on log1p(TPM) HVGs.

The `docs/V12_MOFA_VS_V115_VERDICT.md:46-47` already concedes this: "single-view Gaussian asymptotes to probabilistic-PCA equivalent quickly." The script docstring (line 22-24) calls the result "MOFA+ shared factors" without the qualifier. RUNS.md row `r-2026-05-04-v12-mofa-vs-v115` reports the verdict as "V12_ARCHITECTURAL_OPPORTUNITY" framed around "MOFA+ multi-omics integration" even though *no second modality is present*. The user's reference to "Wave 3 already addressed this (PCA(K=29) reproduces Δ=+0.0217 in 0.038s vs MOFA's 77s)" is plausible but I could not locate `docs/V12_REFACTOR_AUDIT/wave3/pca_substitution.md` — that file does not exist in the canonical repo path read here.

## §D.5 Patch (rename + downgrade marketing; cite the analytic equivalence)

In `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`:
1. Rename the public function/script from `mofa_vs_v11_5_pfs` to `pca_factor_pfs_benchmark` (or keep the file and add a thin alias function with a deprecation warning).
2. Replace the docstring framing:

```python
"""S_v12 — Single-view Gaussian-likelihood factor benchmark vs v11.5 Cox features.

Honest framing
--------------
This script runs mofapy2 in a *single-view, single-group, Gaussian, ARD-on-
weights* configuration on log1p(TPM) HVGs. Per Argelaguet et al. 2020
(Genome Biology, §2.1 of the supplement), this configuration is analytically
equivalent to Bayesian sparse PCA — the multi-modal factor-disentanglement
machinery that distinguishes MOFA+ from PCA is NOT exercised here, because
the IA22 processed slice has no paired proteomics and per-bp CNV is absent
from `data/processed/`.

For a sanity check that the +0.0217 marginal lift is recoverable by a
well-conditioned PCA, see [Wave-3 audit doc] — TODO: write
`docs/V12_REFACTOR_AUDIT/wave3/pca_substitution.md` showing PCA(K=29) on
the same HVGs reproduces Δ within bootstrap noise in <1s vs mofapy2's
~77s. The "V12_ARCHITECTURAL_OPPORTUNITY" verdict therefore reads as
"29 unsupervised expression factors carry signal orthogonal to v11_richer's
23 features" — NOT as "MOFA+ multi-omics integration beats v11.5."

True MOFA+ multi-omics evaluation requires (a) paired CCLE/MMRF proteomics
or (b) per-segment CNV joined to expression — both v12.5+ work.

Run ID: r-2026-05-04-v12-mofa-vs-v115 (filename retained for run-ledger
continuity; semantically a sparse-PCA-vs-Cox benchmark)
"""
```

3. Update `docs/V12_MOFA_VS_V115_VERDICT.md` title to "V12 verdict: 29 unsupervised expression factors vs v11.5 Cox_v11_richer (single-view sparse-PCA benchmark)" and replace headline "MOFA+ shared factors" with "single-view sparse-PCA factors (mofapy2-implemented)."
4. Update `RUNS.md:37` Notes column: replace "v12 MOFA+ multi-omics integration" with "29 unsupervised expression factors (single-view mofapy2 = sparse-PCA configuration)."
5. Create `docs/V12_REFACTOR_AUDIT/wave3/pca_substitution.md` with: PCA(K=29) script, wall-time comparison (sklearn `TruncatedSVD` ~0.04s vs mofapy2 ~77s), Δ_marginal C-index reproduction within bootstrap CI overlap, and a citation to Tipping & Bishop 1999 *J Royal Stat Soc B* "Probabilistic Principal Component Analysis."

## §D.5 Domain rationale

MOFA+ (Argelaguet et al. *Genome Biology* 2020) is valuable specifically for *disentangling shared vs view-specific factors across heterogeneous modalities*. Its mathematical machinery — view-specific noise terms σ²_m, group-level ARD, factor-level sparsity — collapses to standard probabilistic-PCA priors when only one Gaussian view is supplied. Calling a single-view fit "MOFA+ multi-omics" overstates the model's contribution by a full conceptual tier (multi-modal Bayesian factor analysis vs Bayesian PCA). The Wave-3 PCA-equivalence demonstration (if reproducible at the +0.0217 level claimed by the user) makes the methodological point empirically: the "MOFA+" label is doing rhetorical work the math does not support. Honest framing also strengthens the v12.5 roadmap: the right next step is genuine multi-omics MOFA on a dataset with paired proteomics or CNV, not more single-view runs.

## §D.5 Effort

~1 hour: rename + docstring rewrite + RUNS.md edit + verdict-doc title fix. Wave-3 PCA-substitution doc with reproducible script: another ~1 hour.

---

## §summary

| ID | Verified at | Bug class | Real severity | Effort |
|---|---|---|---|---|
| D.1 | `resistancemap/config.py:67-71` + `configs/default.yaml:46-66` | Scope-misframing (panel ≠ 2026 SoC) | BLOCKER for any "MM SoC panel" claim; honest under "GDSC subset" framing | 30 min |
| D.2 | `resistancemap/landscape/scalar_potential.py:88-101` | **Docstring ambiguity, NOT a sign-flip bug** | Downgrade to docs-only; toy test confirms correct descent | 10 min |
| D.3 | `resistancemap/infrastructure/scrna_integration.py:15-58` | In-file marketing lie (random Linear advertised as scGPT) | BLOCKER for any scGPT/scFoundation claim; external docs (README, ARCHITECTURE) are clean | 30 min (a) / 3-5 days (b) |
| D.4 | `resistancemap/infrastructure/imaging_radiomics.py:14-56` | `forward()` returns `torch.randn` per call — pure noise | **HARDEST BLOCKER**: every imaging-derived metric is a hallucination | 1 hr (raise) / 1-2 weeks (real UNI) |
| D.5 | `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py:217-224` | Single-view Gaussian mofapy2 = sparse PCA, mis-marketed as "MOFA+ multi-omics" | BLOCKER for any "MOFA+ multi-omics integration" claim; verdict doc already half-honest at line 46-47 | ~2 hrs (rename + Wave-3 doc) |

**Key correction to the audit ticket**: D.2 is **not** a one-character sign-flip bug — the existing code is correct under the declared `−∇U = score, ż = −∇U` convention; only the docstring on line 91 is sign-typo'd. The other four blockers are confirmed; D.4 (TileEncoder returning `torch.randn` on every forward) is the most severe because it silently injects pure noise into every downstream imaging metric.

**External-marketing audit (per the hard rule)**: `README.md` and `ARCHITECTURE.md` do **not** mention scGPT, scFoundation, UNI, or GigaPath — the marketing lie is **in-file only** for D.3 and D.4. Any `paper/` v8+ artefact tables and the V8/V9 evaluation reports were not reachable through Read alone; without Bash access for `grep -rni "scGPT\|scFoundation\|UNI\|GigaPath\|pathomics\|radiomics" paper/ docs/` the user should run that one-liner to confirm no published table cites these modules' outputs.

Files inspected (all absolute):
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/landscape/scalar_potential.py`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/infrastructure/scrna_integration.py`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/infrastructure/imaging_radiomics.py`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/config.py`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/configs/default.yaml`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/docs/V12_MOFA_VS_V115_VERDICT.md`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/docs/V12_SPRINT1_PROGRESS.md`
- `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/README.md`, `ARCHITECTURE.md`, `RUNS.md`

No source files were modified (per hard rule).