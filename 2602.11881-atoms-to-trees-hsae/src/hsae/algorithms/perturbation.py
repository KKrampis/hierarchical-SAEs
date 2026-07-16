"""Parent-Children Feature Perturbation (Section 3.2).

Implements:
    z_{l,i} ~ Bernoulli(r)
    f~_{l,i}(x) = z_{l,i} f_{l,i}(x) + (1 - z_{l,i}) sum_{j in C_(l,i)} f_{l+1,j}(x)
    SAE_hat_l(x) = sum_i f~_{l,i}(x)

For parent features with an empty children set (C_(l,i) = ∅), the gate is
forced to 1 (always use the feature's own output), matching the convention
used by the parent-children constraint loss (L_PC = 0 when C_(l,i) = ∅).
"""

from __future__ import annotations

import torch

from hsae.algorithms.hierarchy import UNASSIGNED, children_membership_matrix


def children_sum_outputs(
    child_feature_outputs: torch.Tensor,
    parent_idx: torch.Tensor,
    n_parents: int,
) -> torch.Tensor:
    """sum_{j in C_(l,i)} f_{l+1,j}(x) for every parent i.

    child_feature_outputs: [B, n_children, d_model]
    parent_idx: [n_children] (child -> parent assignment, UNASSIGNED if none)
    returns: [B, n_parents, d_model]
    """
    membership = children_membership_matrix(parent_idx, n_parents)  # [n_parents, n_children]
    return torch.einsum("pc,bcd->bpd", membership, child_feature_outputs)


def perturbed_reconstruction(
    parent_feature_outputs: torch.Tensor,
    child_feature_outputs: torch.Tensor,
    parent_idx: torch.Tensor,
    perturbation_rate: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Computes SAE_hat_l(x) using the Bernoulli-gated parent-children perturbation.

    parent_feature_outputs: [B, n_parents, d_model], f_{l,i}(x) for every parent feature.
    child_feature_outputs: [B, n_children, d_model], f_{l+1,j}(x) for every child feature.
    parent_idx: [n_children] child -> parent assignment at this level boundary.
    Returns: [B, d_model], the perturbed reconstruction SAE_hat_l(x) = sum_i f~_{l,i}(x).
    """
    batch_size, n_parents, d_model = parent_feature_outputs.shape
    has_children = torch.zeros(n_parents, dtype=torch.bool, device=parent_idx.device)
    assigned_parents = parent_idx[parent_idx != UNASSIGNED]
    if assigned_parents.numel() > 0:
        has_children[assigned_parents.unique()] = True

    z = torch.bernoulli(
        torch.full((batch_size, n_parents), perturbation_rate, device=parent_feature_outputs.device),
        generator=generator,
    )
    # Features with no children must always use their own output (equivalent to z forced to 1).
    z = torch.where(has_children.unsqueeze(0), z, torch.ones_like(z))

    children_sum = children_sum_outputs(child_feature_outputs, parent_idx, n_parents)
    perturbed = z.unsqueeze(-1) * parent_feature_outputs + (1 - z.unsqueeze(-1)) * children_sum
    return perturbed.sum(dim=1)
