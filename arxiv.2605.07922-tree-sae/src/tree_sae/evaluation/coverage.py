"""Activation coverage (Eq. 4) and the MCS post-hoc hierarchy-detection baseline.

Best MCS variant per Appendix I: "non-scaling, binary" -- binarize both
parent and child activations (nonzero -> 1) and compute plain Pearson
correlation, with no additional magnitude scaling.
"""

from __future__ import annotations

import torch


def binarize(feature_acts: torch.Tensor) -> torch.Tensor:
    return feature_acts > 0


def activation_coverage_score(parent_active: torch.Tensor, child_active: torch.Tensor) -> float:
    """S_cov(fp, fc) = P(parent active | child active) (Eq. 4)."""
    child_count = child_active.sum().item()
    if child_count == 0:
        return float("nan")
    both = (parent_active & child_active).sum().item()
    return both / child_count


def mcs_score(parent_active: torch.Tensor, child_active: torch.Tensor) -> float:
    """Non-scaling, binary-activation correlation (Appendix I, best MCS variant)."""
    p = parent_active.to(torch.float32)
    c = child_active.to(torch.float32)
    p = p - p.mean()
    c = c - c.mean()
    denom = (p.norm() * c.norm()).clamp_min(1e-8)
    return ((p * c).sum() / denom).item()


def top_k_children_by_mcs(
    activations_bool: torch.Tensor,
    parent_index: int,
    candidate_indices: list[int],
    k: int = 5,
) -> list[int]:
    """Returns the `k` candidate features most MCS-correlated with `parent_index`."""
    parent_col = activations_bool[:, parent_index]
    scores = [(mcs_score(parent_col, activations_bool[:, c]), c) for c in candidate_indices if c != parent_index]
    scores.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scores[:k]]
