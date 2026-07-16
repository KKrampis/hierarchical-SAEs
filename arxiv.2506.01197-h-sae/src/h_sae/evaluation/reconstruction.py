"""Reconstruction-quality evaluation (Section 4, "Reconstruction")."""

from __future__ import annotations

import torch


def one_minus_explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """1 - explained variance (normalized L2 reconstruction loss), matching
    Figure 3's "1 - Explained Variance" metric. Lower is better."""
    residual_var = (x - x_hat).pow(2).sum()
    total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    return (residual_var / total_var).item()
