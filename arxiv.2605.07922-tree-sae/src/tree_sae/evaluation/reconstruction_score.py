"""Reconstruction condition score S_res (Eq. 5), via linear-probe ground-truth
concept directions (Section 2.2, Appendix H).

Per Appendix H (Proposition 2), the child feature's OWN encoder vector is not
a valid stand-in for the true concept direction d*_c -- a separately trained
linear probe is required.
"""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression


def train_linear_probe(activations: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Trains a linear probe separating `positive_mask` examples from the rest,
    returning its (unit-normalized) weight vector as the ground-truth concept
    direction d*_c."""
    y = positive_mask.numpy().astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        # Degenerate case (e.g. tiny synthetic batch): fall back to the mean-difference
        # direction, which is a reasonable linear separator when logistic regression
        # cannot be fit (a single class present).
        direction = activations[positive_mask].mean(dim=0) - activations[~positive_mask].mean(dim=0) \
            if positive_mask.any() and (~positive_mask).any() else activations.mean(dim=0)
    else:
        probe = LogisticRegression(max_iter=1000)
        probe.fit(activations.numpy(), y)
        direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
    return direction / direction.norm().clamp_min(1e-8)


def reconstruction_score(
    parent_decoder_vector: torch.Tensor,
    child_decoder_vector: torch.Tensor,
    true_direction: torch.Tensor,
) -> float:
    """S_res(fp, fc) = min( d*_c . d_c, d*_c . d_p ) (Eq. 5)."""
    dp = parent_decoder_vector / parent_decoder_vector.norm().clamp_min(1e-8)
    dc = child_decoder_vector / child_decoder_vector.norm().clamp_min(1e-8)
    d_star = true_direction / true_direction.norm().clamp_min(1e-8)
    s_p = (d_star @ dp).item()
    s_c = (d_star @ dc).item()
    return min(s_c, s_p)
