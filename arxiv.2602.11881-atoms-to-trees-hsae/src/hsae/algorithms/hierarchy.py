"""Similarity-Based Hierarchy Update and Partial Tree quantile filtering.

Implements the supporting algorithms "Similarity-Based Hierarchy Update" and
"Partial Tree Structure" from pipeline/01_algorithm_extraction.yaml
(Section 3.3 of the paper).

Each level-(l+1) feature is assigned to at most one level-l parent (its most
similar candidate), so children sets are disjoint across parents by
construction (Section 3, C_(l,i) ∩ C_(l,k) = ∅ for i != k). Features whose
best-match similarity falls at or below the bottom `exclusion_quantile` of
the level are left unassigned (parent index -1), implementing the paper's
Partial Tree relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

UNASSIGNED = -1


@dataclass
class CoactivationTracker:
    """EMA-tracked co-activation statistics between adjacent-level features.

    Alternative similarity metric mentioned in Section 3.3: "similarity can be
    defined using feature co-activation statistics ... with statistics tracked
    via an exponential moving average". The exact EMA momentum for this
    specific alternative is not given in the paper; `momentum` is exposed as a
    tunable default (see HierarchyConfig.coactivation_ema_momentum, marked
    INFERRED in pipeline/03_implementation_plan.yaml).
    """

    momentum: float = 0.99
    _coactivation: dict[int, torch.Tensor] = field(default_factory=dict)

    def update(self, level: int, child_active: torch.Tensor, parent_active: torch.Tensor) -> None:
        """child_active: [B, n_children] bool, parent_active: [B, n_parents] bool."""
        batch_stat = (child_active.float().T @ parent_active.float()) / child_active.shape[0]
        if level not in self._coactivation:
            self._coactivation[level] = batch_stat
        else:
            m = self.momentum
            self._coactivation[level] = m * self._coactivation[level] + (1 - m) * batch_stat

    def similarity(self, level: int, n_children: int, n_parents: int) -> torch.Tensor:
        if level not in self._coactivation:
            return torch.zeros(n_children, n_parents)
        return self._coactivation[level]


def _cosine_similarity_matrix(child_vectors: torch.Tensor, parent_vectors: torch.Tensor) -> torch.Tensor:
    """child_vectors: [n_children, d], parent_vectors: [n_parents, d] -> [n_children, n_parents]."""
    child_norm = torch.nn.functional.normalize(child_vectors, dim=-1)
    parent_norm = torch.nn.functional.normalize(parent_vectors, dim=-1)
    return child_norm @ parent_norm.T


def compute_similarity(
    child_sae,
    parent_sae,
    metric: str,
    coactivation_tracker: CoactivationTracker | None = None,
    level: int | None = None,
) -> torch.Tensor:
    """Returns an [n_children, n_parents] similarity matrix under the requested metric.

    metric: "encoder" (default, Section 3.3), "decoder", or "coactivation".
    """
    if metric == "encoder":
        return _cosine_similarity_matrix(child_sae.W_enc.T, parent_sae.W_enc.T)
    if metric == "decoder":
        return _cosine_similarity_matrix(child_sae.W_dec, parent_sae.W_dec)
    if metric == "coactivation":
        if coactivation_tracker is None or level is None:
            raise ValueError("coactivation metric requires a CoactivationTracker and level index")
        return coactivation_tracker.similarity(level, child_sae.dict_size, parent_sae.dict_size)
    raise ValueError(f"Unknown similarity metric: {metric!r}")


def assign_parents(
    similarity: torch.Tensor,
    exclusion_quantile: float,
    max_children_per_parent: int | None = None,
) -> torch.Tensor:
    """Assigns each child (row) to its most similar parent (column), applying the
    bottom-`exclusion_quantile` Partial Tree filter.

    `max_children_per_parent` implements the "Binary Tree" ablation variant
    (Appendix F.3: "restricts each parent to a maximum of two children"): for
    any parent whose assigned children exceed the cap, only the `cap`
    highest-similarity children are kept and the rest are unassigned. This is
    a simple greedy approximation of the paper's constrained assignment (the
    paper does not specify an exact tie-breaking/reassignment algorithm).

    Returns a LongTensor of shape [n_children] with values in {-1, 0, ..., n_parents-1};
    -1 means "unassigned" (independent feature, Section 3.3 "Partial Tree Structure").
    """
    best_score, best_parent = similarity.max(dim=1)
    if exclusion_quantile > 0:
        threshold = torch.quantile(best_score, exclusion_quantile)
        excluded = best_score <= threshold
    else:
        excluded = torch.zeros_like(best_score, dtype=torch.bool)
    parent_idx = best_parent.clone()
    parent_idx[excluded] = UNASSIGNED

    if max_children_per_parent is not None:
        for parent in parent_idx.unique().tolist():
            if parent == UNASSIGNED:
                continue
            members = (parent_idx == parent).nonzero(as_tuple=True)[0]
            if members.numel() <= max_children_per_parent:
                continue
            member_scores = best_score[members]
            keep = members[torch.topk(member_scores, k=max_children_per_parent).indices]
            drop = members[~torch.isin(members, keep)]
            parent_idx[drop] = UNASSIGNED

    return parent_idx


def children_sets(parent_idx: torch.Tensor, n_parents: int) -> list[list[int]]:
    """Inverts a parent_idx assignment (per child) into per-parent children-index lists,
    i.e. C_(l,i) for every parent i at level l."""
    result: list[list[int]] = [[] for _ in range(n_parents)]
    for child_index, parent in enumerate(parent_idx.tolist()):
        if parent != UNASSIGNED:
            result[parent].append(child_index)
    return result


def children_membership_matrix(parent_idx: torch.Tensor, n_parents: int) -> torch.Tensor:
    """Builds an [n_parents, n_children] float 0/1 matrix M where M[i, j] = 1 iff child j
    is assigned to parent i. Multiplying M by a [n_children, ...] tensor of per-child
    feature outputs yields, for each parent, the sum over its assigned children
    (sum_{j in C_(l,i)} f_{l+1,j}(x)) used by both the parent-children constraint loss
    and the feature perturbation mechanism.
    """
    n_children = parent_idx.shape[0]
    membership = torch.zeros(n_parents, n_children, dtype=torch.float32, device=parent_idx.device)
    assigned = parent_idx != UNASSIGNED
    membership[parent_idx[assigned], torch.arange(n_children, device=parent_idx.device)[assigned]] = 1.0
    return membership


def update_level_hierarchy(
    child_sae,
    parent_sae,
    metric: str = "encoder",
    exclusion_quantile: float = 0.20,
    coactivation_tracker: CoactivationTracker | None = None,
    level: int | None = None,
    max_children_per_parent: int | None = None,
) -> torch.Tensor:
    """End-to-end Similarity-Based Hierarchy Update + Partial Tree filter for one
    pair of adjacent levels. Returns the parent_idx tensor described in `assign_parents`."""
    similarity = compute_similarity(child_sae, parent_sae, metric, coactivation_tracker, level)
    return assign_parents(similarity, exclusion_quantile, max_children_per_parent)
