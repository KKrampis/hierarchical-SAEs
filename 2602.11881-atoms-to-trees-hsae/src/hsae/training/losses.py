"""HSAE loss functions (Section 3.1).

    L_SAE,l(x) = ||SAE_hat_l(x) - x||_2^2 + lambda_l * sum_i 1{sigma(e_{l,i}^T x) > 0}
    L_PC,(l,i)(x) = ||f_{l,i}(x) - sum_{j in C_(l,i)} f_{l+1,j}(x)||_2^2   if C_(l,i) != ∅
                  = 0                                                      if C_(l,i) == ∅
    L_HSAE = E_x[ sum_{l=1}^{L} L_SAE,l(x) + rho * sum_{l=1}^{L-1} sum_i L_PC,(l,i)(x) ]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hsae.algorithms.hierarchy import UNASSIGNED
from hsae.models.hsae import HSAE


@dataclass
class HSAELossOutput:
    total: torch.Tensor
    sae_losses: list[torch.Tensor] = field(default_factory=list)
    pc_losses: list[torch.Tensor] = field(default_factory=list)
    mse_per_level: list[torch.Tensor] = field(default_factory=list)
    l0_per_level: list[torch.Tensor] = field(default_factory=list)


def level_sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    active_mask: torch.Tensor,
    lambda_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss, mean_mse, mean_l0) for one level."""
    mse = (recon - x).pow(2).sum(dim=-1)  # [B]
    l0 = active_mask.sum(dim=-1)  # [B]
    loss = (mse + lambda_l * l0).mean()
    return loss, mse.mean().detach(), l0.mean().detach()


def parent_children_loss(
    parent_feature_outputs: torch.Tensor,
    children_sum: torch.Tensor,
    parent_idx_of_children: torch.Tensor,
    n_parents: int,
) -> torch.Tensor:
    """Sum over parent features i of L_PC,(l,i)(x), averaged over the batch.

    parent_feature_outputs: [B, n_parents, d_model] = f_{l,i}(x)
    children_sum: [B, n_parents, d_model] = sum_{j in C_(l,i)} f_{l+1,j}(x)
    parent_idx_of_children: [n_children] child->parent assignment at this boundary,
        used only to determine which parents have a non-empty children set.
    """
    has_children = torch.zeros(n_parents, dtype=torch.bool, device=parent_feature_outputs.device)
    assigned = parent_idx_of_children[parent_idx_of_children != UNASSIGNED]
    if assigned.numel() > 0:
        has_children[assigned.unique()] = True

    squared_error = (parent_feature_outputs - children_sum).pow(2).sum(dim=-1)  # [B, n_parents]
    squared_error = squared_error * has_children.unsqueeze(0)
    return squared_error.sum(dim=1).mean()


def compute_hsae_loss(
    model: HSAE,
    x: torch.Tensor,
    feature_acts: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    feature_outputs: list[torch.Tensor],
    lambdas: torch.Tensor,
    rho: float,
) -> HSAELossOutput:
    sae_losses, pc_losses, mse_per_level, l0_per_level = [], [], [], []

    for level in range(model.num_levels):
        active_mask = model.levels[level].active_mask(x)
        loss, mse, l0 = level_sae_loss(x, reconstructions[level], active_mask, lambdas[level])
        sae_losses.append(loss)
        mse_per_level.append(mse)
        l0_per_level.append(l0)

    for level in range(model.num_levels - 1):
        children_sum = model.children_sum_feature_outputs(level, feature_outputs)
        pc_loss = parent_children_loss(
            parent_feature_outputs=feature_outputs[level],
            children_sum=children_sum,
            parent_idx_of_children=model.parent_idx[level],
            n_parents=model.levels[level].dict_size,
        )
        pc_losses.append(pc_loss)

    total = torch.stack(sae_losses).sum() + rho * torch.stack(pc_losses).sum()
    return HSAELossOutput(
        total=total,
        sae_losses=sae_losses,
        pc_losses=pc_losses,
        mse_per_level=mse_per_level,
        l0_per_level=l0_per_level,
    )
