"""Resistance-mechanism classifier head (v20 Phase 5 interpretability).

Attributes a latent resistance state ``z`` to one of a small, biologically
named set of MM drug-resistance mechanisms, so a progression prediction can be
decomposed into "which resistance program is driving this patient's risk".

This is an *interpretability* head: its labels are hypotheses to be cross-checked
against pathway / CRISPR evidence (the k-of-N causal gate), never asserted as
ground-truth mechanism. Training requires real mechanism labels; absent those it
is used only for unsupervised attribution (softmax over learned prototypes).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

# Named MM resistance mechanisms (ordered; index == class id).
RESISTANCE_MECHANISMS: List[str] = [
    "proteasome_bypass",        # PI resistance (e.g. PSMB5 / proteostasis rewiring)
    "crbn_imid_loss",           # IMiD resistance (CRBN / IKZF1-3 axis)
    "hdac_compensation",        # HDACi resistance (chromatin-modifier compensation)
    "bcl2_dependence_shift",    # venetoclax (BCL2/MCL1 balance)
    "drug_efflux",              # ABC-transporter mediated
    "clonal_selection",         # outgrowth of a pre-existing resistant clone
    "immune_evasion",           # antigen loss / microenvironment (e.g. BCMA loss)
    "unknown",                  # explicit abstention class
]


class MechanismClassifier(nn.Module):
    """Latent ``z`` -> resistance-mechanism logits."""

    def __init__(self, d_latent: int, mechanisms: Sequence[str] = RESISTANCE_MECHANISMS,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mechanisms = list(mechanisms)
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, len(self.mechanisms)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits, shape (B, n_mechanisms)."""
        return self.net(z)

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Softmax mechanism probabilities, shape (B, n_mechanisms)."""
        return torch.softmax(self.forward(z), dim=-1)

    def predict(self, z: torch.Tensor) -> List[str]:
        """Argmax mechanism label per row."""
        idx = self.forward(z).argmax(dim=-1).tolist()
        return [self.mechanisms[i] for i in idx]

    def loss(self, z: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -1) -> torch.Tensor:
        """Cross-entropy on real mechanism labels (``-1`` rows ignored)."""
        return nn.functional.cross_entropy(
            self.forward(z), labels.long(), ignore_index=ignore_index,
        )
