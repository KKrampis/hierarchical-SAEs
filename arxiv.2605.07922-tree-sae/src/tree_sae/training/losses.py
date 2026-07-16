"""Tree SAE loss functions (Section 4, Eq. 7-9).

    L_recons(x)  = sum_{l=1}^{L} || cumulative_recon[l] - x ||_2^2
    L_aux,l(x)   = || e_hat_l + cumulative_recon[l] - x ||_2^2
    L_tree(x)    = L_recons(x) + sum_{l=1}^{L} alpha_l * L_aux,l(x)

e_hat_l is the reconstruction of the current residual error using the
top-`aux_topk` currently-dead features AT THAT LAYER ONLY, using their raw
(ungated) pre-activations -- the paper's per-layer variant of the standard
Gao et al. (2024) auxiliary loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tree_sae.models.tree_sae import TreeSAE


@dataclass
class TreeSAELossOutput:
    total: torch.Tensor
    recons_loss: torch.Tensor
    aux_losses: list[torch.Tensor] = field(default_factory=list)
    mse_per_level: list[torch.Tensor] = field(default_factory=list)


def layer_aux_reconstruction(
    model: TreeSAE,
    pre_activation: torch.Tensor,
    level: int,
    dead_mask: torch.Tensor,
    aux_topk: int,
) -> torch.Tensor:
    """e_hat_l: reconstruction contribution from the top-`aux_topk` dead features
    restricted to layer `level`'s block, using their raw pre-activation values."""
    block = model.layer_slice(level)
    layer_pre = pre_activation[:, block]
    masked = F.relu(layer_pre) * dead_mask.to(layer_pre.dtype).unsqueeze(0)

    k = min(aux_topk, masked.shape[-1])
    if k == 0 or not torch.any(dead_mask):
        return torch.zeros(pre_activation.shape[0], model.d_model, device=pre_activation.device)

    values, indices = torch.topk(masked, k=k, dim=-1)
    aux_acts = torch.zeros_like(masked)
    aux_acts.scatter_(1, indices, values)
    return aux_acts @ model.W_dec[block]


def compute_tree_sae_loss(
    model: TreeSAE,
    x: torch.Tensor,
    pre_activation: torch.Tensor,
    cumulative: list[torch.Tensor],
    dead_masks: list[torch.Tensor],
    aux_topk: int,
    layer_alphas: list[float],
) -> TreeSAELossOutput:
    mse_per_level = [(recon - x).pow(2).sum(dim=-1).mean() for recon in cumulative]
    recons_loss = torch.stack(mse_per_level).sum()

    aux_losses = []
    for level in range(model.num_levels):
        alpha = layer_alphas[level]
        if alpha == 0.0:
            aux_losses.append(x.new_zeros(()))
            continue
        e_hat_l = layer_aux_reconstruction(model, pre_activation, level, dead_masks[level], aux_topk)
        aux_loss = (e_hat_l + cumulative[level] - x).pow(2).sum(dim=-1).mean()
        aux_losses.append(aux_loss)

    weighted_aux = sum(alpha * loss for alpha, loss in zip(layer_alphas, aux_losses))
    total = recons_loss + weighted_aux

    return TreeSAELossOutput(total=total, recons_loss=recons_loss, aux_losses=aux_losses, mse_per_level=mse_per_level)
