"""Simplified absorption/splitting probe (Section 4.1), adapted for synthetic
data with a known ground-truth parent-concept label per sample.

The paper's actual benchmark (SAEBench's first-letter classification task,
Karvonen et al. 2025) requires real token/letter labels; here we substitute
the synthetic dataset's known parent-concept label as the classification
target, and measure what fraction of a linear probe's decoder-projected
"energy" is NOT explained by a single dominant SAE feature (lower = less
splitting/absorption, mirroring the paper's metric direction).
"""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression


def train_linear_probe(activations: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    y = positive_mask.numpy().astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        direction = activations.mean(dim=0)
    else:
        probe = LogisticRegression(max_iter=1000)
        probe.fit(activations.numpy(), y)
        direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
    return direction / direction.norm().clamp_min(1e-8)


def simplified_absorption_score(
    x: torch.Tensor,
    z: torch.Tensor,
    parent_labels: torch.Tensor,
    target_label: int,
    decoder: torch.Tensor,
) -> float:
    """Returns the fraction of the probe-aligned reconstruction "energy" not
    explained by a single dominant top-level feature (lower = less
    splitting/absorption).

    x: [N, d] activations, z: [N, m_top] top-level codes, decoder: [m_top, d].
    """
    positive_mask = parent_labels == target_label
    if not (positive_mask.any() and (~positive_mask).any()):
        return float("nan")

    direction = train_linear_probe(x, positive_mask)
    decoder_scores = (decoder @ direction).abs()  # [m_top]
    avg_activation = z[positive_mask].abs().mean(dim=0)  # [m_top]
    weighted = decoder_scores * avg_activation

    total = weighted.sum()
    if total <= 0:
        return float("nan")
    top1_fraction = weighted.max() / total
    return 1.0 - top1_fraction.item()
