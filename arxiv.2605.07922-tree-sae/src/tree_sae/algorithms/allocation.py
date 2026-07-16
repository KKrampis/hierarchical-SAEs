"""Algorithm 1 (Greedy allocation) and Algorithm 2 (Dynamic feature reallocation).

From pipeline/01_algorithm_extraction.yaml, Appendix G of the paper.

Algorithm 1 solves Eq. 10 (the max-min-fair optimal number of children per
parent) in O(s_l log m_l) via a max-heap, per Theorem 4.1.

Algorithm 2 periodically recomputes the optimal counts via Algorithm 1 and
reassigns only DEAD children to under-served parents via a first-fit
strategy, leaving already-alive children untouched.
"""

from __future__ import annotations

import heapq

import torch

from tree_sae.models.tree_sae import TreeSAE


def greedy_allocation(capacities: torch.Tensor, eligible: torch.Tensor, s_l: int) -> torch.Tensor:
    """Algorithm 1: returns k*_l, a LongTensor of length m_l = len(capacities), the
    optimal (max-min-fair) number of children assigned to each candidate parent.

    Ineligible parents, or parents with zero capacity, never receive children.
    If the eligible parents' total capacity is insufficient to place all s_l
    children, fewer than s_l are assigned (the remainder is left for the caller
    to handle, e.g. by leaving those dead features dead until the next round).
    """
    m_l = capacities.shape[0]
    counts = torch.zeros(m_l, dtype=torch.long)
    heap: list[tuple[float, int]] = []
    for p in range(m_l):
        if bool(eligible[p]) and capacities[p].item() > 0:
            heapq.heappush(heap, (-capacities[p].item() / 1.0, p))

    assigned = 0
    while heap and assigned < s_l:
        neg_payoff, p = heapq.heappop(heap)
        counts[p] += 1
        assigned += 1
        next_payoff = capacities[p].item() / (counts[p].item() + 1)
        heapq.heappush(heap, (-next_payoff, p))

    return counts


def dynamic_reallocation(
    model: TreeSAE,
    level: int,
    capacities: torch.Tensor,
    eligible: torch.Tensor,
    dead_mask: torch.Tensor,
) -> None:
    """Algorithm 2 for a single layer: recomputes optimal child counts (Algorithm 1)
    then first-fit-reassigns only the DEAD features in this layer's block to
    under-served parents. Mutates `model.parent_idx` in place for this layer's block.

    capacities: [m_l] capacity estimate for every candidate parent (features at
        layers 0..level-1).
    eligible: [m_l] bool, parent eligibility (activation-rate threshold).
    dead_mask: [s_l] bool, whether each feature IN THIS LAYER's block is currently dead.
    """
    m_l = model.offsets[level]
    s_l = model.layer_sizes[level]
    block = model.layer_slice(level)

    k_star = greedy_allocation(capacities, eligible, s_l)

    current_parents = model.parent_idx[block].clone()
    alive_mask = ~dead_mask
    alive_parents = current_parents[alive_mask]
    alive_parents = alive_parents[alive_parents >= 0]
    alive_count_per_parent = torch.bincount(alive_parents, minlength=m_l) if alive_parents.numel() else torch.zeros(
        m_l, dtype=torch.long
    )

    deficits = (k_star - alive_count_per_parent).clamp(min=0)
    dead_positions = dead_mask.nonzero(as_tuple=True)[0].tolist()

    pointer = 0
    for parent in range(m_l):
        need = int(deficits[parent].item())
        for _ in range(need):
            if pointer >= len(dead_positions):
                break
            local_idx = dead_positions[pointer]
            pointer += 1
            current_parents[local_idx] = parent
        if pointer >= len(dead_positions):
            break

    model.parent_idx[block] = current_parents


def reassign_dead_to_root(model: TreeSAE, dead_steps_threshold: int) -> None:
    """One-time rule (~50% of training, Appendix G): move every currently-dead
    feature at layers >= 1 directly to the root, so it can activate freely
    (ungated) afterward."""
    dead = model.is_dead(dead_steps_threshold)
    for level in range(1, model.num_levels):
        block = model.layer_slice(level)
        dead_in_block = dead[block]
        current = model.parent_idx[block].clone()
        current[dead_in_block] = -1
        model.parent_idx[block] = current
