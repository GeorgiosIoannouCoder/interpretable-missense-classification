"""Small MLP classifier head over frozen ESM-2 residue embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ESM2HeadMLP(nn.Module):
    """A small MLP classifier head: ``hidden_dim -> hidden -> 1``.

    Parameters
    ----------
    in_dim : int
        Input embedding dimensionality (1280 for ESM-2 650M).
    hidden : int
        Hidden-layer width.
    dropout : float
        Dropout probability applied after the input and after the hidden layer.
    norm : str
        ``"layer"`` for ``nn.LayerNorm`` (default) or ``"batch"`` for ``nn.BatchNorm1d``.
    """

    def __init__(self, in_dim: int = 1280, hidden: int = 256, dropout: float = 0.2, norm: str = "layer") -> None:
        super().__init__()
        if norm == "layer":
            norm_in = nn.LayerNorm(in_dim)
            norm_h = nn.LayerNorm(hidden)
        elif norm == "batch":
            norm_in = nn.BatchNorm1d(in_dim)
            norm_h = nn.BatchNorm1d(hidden)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.net = nn.Sequential(
            norm_in,
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            norm_h,
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning **logits** of shape ``(batch,)``."""
        return self.net(x).squeeze(-1)
