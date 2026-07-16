"""The paper's novel hierarchical-pair metric (Section 5.2).

For each sampled parent-child pair, trains a linear probe on the child to
obtain its true concept direction d*_c, ranks EVERY feature in the
dictionary by cosine similarity of its decoder vector with d*_c, and checks
whether BOTH the parent and the child feature fall within the top-5 highest
-correlation features. The reported metric is the fraction of pairs where
this holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tree_sae.evaluation.reconstruction_score import train_linear_probe
from tree_sae.models.tree_sae import TreeSAE


@dataclass
class HierarchyPair:
    parent_index: int
    child_index: int


def sample_dense_parents(activations_bool: torch.Tensor, parent_pool: list[int], num_parents: int, density_percentile: float = 50.0, seed: int = 0) -> list[int]:
    """Samples `num_parents` features from `parent_pool` whose activation density
    is above the given percentile (Section 5.2: "100 random parent features
    that are above the 50% most dense feature")."""
    densities = activations_bool[:, parent_pool].float().mean(dim=0)
    threshold = torch.quantile(densities, density_percentile / 100.0)
    eligible = [p for p, d in zip(parent_pool, densities.tolist()) if d >= threshold]
    generator = torch.Generator().manual_seed(seed)
    if len(eligible) <= num_parents:
        return eligible
    perm = torch.randperm(len(eligible), generator=generator)[:num_parents]
    return [eligible[i] for i in perm.tolist()]


def hierarchical_pair_metric(
    model: TreeSAE,
    activations: torch.Tensor,
    activations_bool: torch.Tensor,
    pairs: list[HierarchyPair],
    top_k: int = 5,
) -> float:
    """Fraction of `pairs` where both parent and child are among the top-`top_k`
    decoder-vector correlations with the child's true (probe) concept direction."""
    if not pairs:
        return float("nan")

    decoder_normed = F.normalize(model.W_dec, dim=-1)
    hits = 0
    for pair in pairs:
        child_active = activations_bool[:, pair.child_index]
        if not (child_active.any() and (~child_active).any()):
            continue
        direction = train_linear_probe(activations, child_active)
        scores = decoder_normed @ direction
        top_indices = set(torch.topk(scores, k=min(top_k, scores.shape[0])).indices.tolist())
        if pair.parent_index in top_indices and pair.child_index in top_indices:
            hits += 1
    return hits / len(pairs)
