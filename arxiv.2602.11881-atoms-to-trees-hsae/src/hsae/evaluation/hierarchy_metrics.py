"""Hierarchy-consistency evaluation metrics (Section 4.2).

- Parent-Child Alignment: Hamming distance between the ground-truth parent
  activation pattern and a logical-OR reconstruction from its children.
- Parent-given-child probability P(activ_p | activ_c): hierarchical necessity.
- Child-given-parent probability P(activ_c | activ_p): average semantic
  relevance / coverage of children within their parent.

All three are averaged over every assigned parent-child pair, matching
"averaged over all assigned parent-child pairs" (Section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hsae.algorithms.hierarchy import UNASSIGNED, children_membership_matrix
from hsae.models.hsae import HSAE


@dataclass
class LevelHierarchyMetrics:
    hamming_distance: float
    parent_given_child: float
    child_given_parent: float
    num_assigned_pairs: int


def binarize(feature_acts: torch.Tensor) -> torch.Tensor:
    return feature_acts > 0


def hamming_distance_metric(
    child_active: torch.Tensor,
    parent_active: torch.Tensor,
    parent_idx: torch.Tensor,
) -> float:
    """Average per-parent Hamming distance between ground-truth parent activation
    and the logical-OR of its assigned children's activations, over parents with
    at least one child."""
    n_parents = parent_active.shape[1]
    membership = children_membership_matrix(parent_idx, n_parents)  # [n_parents, n_children]
    has_children = membership.sum(dim=1) > 0
    if not torch.any(has_children):
        return float("nan")

    predicted = (membership @ child_active.float().T).T > 0  # [N, n_parents], logical-OR via sum>0
    mismatches = (predicted != parent_active).float().sum(dim=0)  # [n_parents]
    return mismatches[has_children].mean().item()


def conditional_probabilities(
    child_active: torch.Tensor,
    parent_active: torch.Tensor,
    parent_idx: torch.Tensor,
) -> tuple[float, float, int]:
    """Returns (mean P(parent|child), mean P(child|parent), num_pairs) averaged over
    every assigned (parent, child) pair."""
    parent_given_child, child_given_parent = [], []
    for child_index in range(parent_idx.shape[0]):
        parent_index = int(parent_idx[child_index])
        if parent_index == UNASSIGNED:
            continue
        child_col = child_active[:, child_index]
        parent_col = parent_active[:, parent_index]
        n_child_active = child_col.sum().item()
        n_parent_active = parent_col.sum().item()
        both_active = (child_col & parent_col).sum().item()
        if n_child_active > 0:
            parent_given_child.append(both_active / n_child_active)
        if n_parent_active > 0:
            child_given_parent.append(both_active / n_parent_active)

    mean_pgc = sum(parent_given_child) / len(parent_given_child) if parent_given_child else float("nan")
    mean_cgp = sum(child_given_parent) / len(child_given_parent) if child_given_parent else float("nan")
    return mean_pgc, mean_cgp, len(parent_given_child)


def evaluate_hierarchy(model: HSAE, activations: torch.Tensor) -> list[LevelHierarchyMetrics]:
    """Computes all three metrics for every level boundary of a trained (or
    post-hoc-assigned) HSAE, given a batch of held-out activations."""
    with torch.no_grad():
        feature_acts = [sae.encode(activations) for sae in model.levels]
    active = [binarize(acts) for acts in feature_acts]

    results = []
    for level in range(model.num_levels - 1):
        parent_active = active[level]
        child_active = active[level + 1]
        parent_idx = model.parent_idx[level]

        hd = hamming_distance_metric(child_active, parent_active, parent_idx)
        pgc, cgp, n_pairs = conditional_probabilities(child_active, parent_active, parent_idx)
        results.append(
            LevelHierarchyMetrics(
                hamming_distance=hd,
                parent_given_child=pgc,
                child_given_parent=cgp,
                num_assigned_pairs=n_pairs,
            )
        )
    return results


def average_metrics(per_level: list[LevelHierarchyMetrics]) -> LevelHierarchyMetrics:
    valid = [m for m in per_level if m.num_assigned_pairs > 0]
    if not valid:
        return LevelHierarchyMetrics(float("nan"), float("nan"), float("nan"), 0)
    return LevelHierarchyMetrics(
        hamming_distance=sum(m.hamming_distance for m in valid) / len(valid),
        parent_given_child=sum(m.parent_given_child for m in valid) / len(valid),
        child_given_parent=sum(m.child_given_parent for m in valid) / len(valid),
        num_assigned_pairs=sum(m.num_assigned_pairs for m in valid),
    )
