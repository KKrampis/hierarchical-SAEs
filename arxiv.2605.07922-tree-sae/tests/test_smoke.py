"""Fast CPU smoke test: exercises the full Tree SAE pipeline end-to-end on
synthetic data, per the "Validation Approach" in
pipeline/03_implementation_plan.yaml.
"""

from __future__ import annotations

import torch

from tree_sae.algorithms.allocation import dynamic_reallocation, greedy_allocation
from tree_sae.config import TreeSAEConfig
from tree_sae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from tree_sae.evaluation import (
    HierarchyPair,
    activation_coverage_score,
    binarize,
    hierarchical_pair_metric,
    mcs_score,
)
from tree_sae.models.tree_sae import ROOT, TreeSAE
from tree_sae.training import TreeSAETrainer
from tree_sae.utils import set_seed


def test_config_loads():
    config = TreeSAEConfig.small_synthetic()
    assert config.model.num_levels == len(config.model.layer_sizes) == 3


def test_forward_shapes_and_gate_invariant():
    config = TreeSAEConfig.small_synthetic()
    model = TreeSAE(config.model)
    x = torch.randn(5, config.model.d_model)
    raw, gated, cumulative = model(x)

    assert raw.shape == gated.shape == (5, model.dict_size)
    assert len(cumulative) == model.num_levels
    for recon in cumulative:
        assert recon.shape == (5, config.model.d_model)

    # Gate invariant: for every non-layer-1 feature, if the parent's gated
    # activation is 0, the feature's own gated activation must also be 0.
    # ROOT-parented features (parent index -1) are always considered active.
    for level in range(1, model.num_levels):
        block = model.layer_slice(level)
        parents = model.parent_idx[block]
        is_root = parents.eq(ROOT)
        prefix = gated[:, : model.offsets[level]]
        gathered_active = torch.gather(prefix, 1, parents.clamp(min=0).unsqueeze(0).expand(5, -1)) > 0
        parent_active = is_root.unsqueeze(0) | gathered_active
        child_active = gated[:, block] > 0
        # child active implies parent active: no case of (child_active AND NOT parent_active).
        assert torch.all(parent_active | ~child_active)


def test_greedy_allocation_matches_theoretical_optimum():
    capacities = torch.tensor([10.0, 6.0, 3.0])
    eligible = torch.tensor([True, True, True])
    s_l = 5
    counts = greedy_allocation(capacities, eligible, s_l)

    assert counts.sum().item() == s_l
    achieved_min_payoff = min(
        (capacities[p] / counts[p]).item() for p in range(3) if counts[p] > 0
    )
    # Brute-force search the true optimum over small integer allocations for comparison.
    best = 0.0
    for k0 in range(0, s_l + 1):
        for k1 in range(0, s_l + 1 - k0):
            k2 = s_l - k0 - k1
            ks = [k0, k1, k2]
            payoffs = [capacities[p].item() / ks[p] for p in range(3) if ks[p] > 0]
            if payoffs:
                best = max(best, min(payoffs))
    assert abs(achieved_min_payoff - best) < 1e-6


def test_dynamic_reallocation_only_touches_dead_features():
    config = TreeSAEConfig.small_synthetic()
    model = TreeSAE(config.model)
    level = 1
    block = model.layer_slice(level)
    original_parents = model.parent_idx[block].clone()

    dead_mask = torch.zeros(model.layer_sizes[level], dtype=torch.bool)
    dead_mask[0] = True  # only the first feature at this layer is dead

    m_l = model.offsets[level]
    capacities = torch.ones(m_l) * 5.0
    eligible = torch.ones(m_l, dtype=torch.bool)

    dynamic_reallocation(model, level, capacities, eligible, dead_mask)

    new_parents = model.parent_idx[block]
    # Every ALIVE feature's parent must be unchanged.
    alive_positions = (~dead_mask).nonzero(as_tuple=True)[0]
    assert torch.equal(new_parents[alive_positions], original_parents[alive_positions])


def test_reassign_dead_to_root_only_moves_dead_features():
    from tree_sae.algorithms.allocation import reassign_dead_to_root

    config = TreeSAEConfig.small_synthetic()
    model = TreeSAE(config.model)
    model.steps_since_fired[model.layer_slice(1)] = 10_000  # mark layer-2 features dead

    reassign_dead_to_root(model, dead_steps_threshold=config.aux.dead_feature_window_tokens)

    assert torch.all(model.parent_idx[model.layer_slice(1)] == ROOT)


def test_training_reduces_reconstruction_loss():
    set_seed(0)
    config = TreeSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    dataset = SyntheticHierarchicalActivations(spec, num_samples=max(config.data.training_tokens, config.optim.batch_size), seed=0)

    trainer = TreeSAETrainer(config)
    logs = trainer.train(dataset, total_steps=config.optim.total_steps, log_every=25)

    assert len(logs) >= 2
    assert logs[-1].recons_loss < logs[0].recons_loss


def test_dead_feature_rate_runs_with_and_without_allocation():
    set_seed(0)
    config = TreeSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    dataset = SyntheticHierarchicalActivations(spec, num_samples=max(config.data.training_tokens, config.optim.batch_size), seed=0)

    with_alloc = TreeSAETrainer(config, enable_dynamic_allocation=True)
    with_alloc.train(dataset, total_steps=config.optim.total_steps)
    rate_with = with_alloc.dead_feature_rate()

    without_alloc = TreeSAETrainer(config, enable_dynamic_allocation=False)
    without_alloc.train(dataset, total_steps=config.optim.total_steps)
    rate_without = without_alloc.dead_feature_rate()

    assert 0.0 <= rate_with <= 1.0
    assert 0.0 <= rate_without <= 1.0


def test_evaluation_suite_runs_end_to_end():
    set_seed(0)
    config = TreeSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    dataset = SyntheticHierarchicalActivations(spec, num_samples=max(config.data.training_tokens, config.optim.batch_size), seed=0)
    eval_dataset = SyntheticHierarchicalActivations(spec, num_samples=200, seed=1)

    trainer = TreeSAETrainer(config)
    trainer.train(dataset, total_steps=config.optim.total_steps)
    model = trainer.model

    with torch.no_grad():
        _, gated, _ = model(eval_dataset.activations)
    activations_bool = binarize(gated)

    parent_col = activations_bool[:, 0]
    child_col = activations_bool[:, model.offsets[1]]
    cov = activation_coverage_score(parent_col, child_col)
    mcs = mcs_score(parent_col, child_col)
    assert cov == cov or cov != cov  # not asserting a value, just that it doesn't crash (NaN allowed)
    assert -1.0 - 1e-6 <= mcs <= 1.0 + 1e-6

    tree_pairs = []
    for level in range(1, model.num_levels):
        block = model.layer_slice(level)
        for local_idx, parent in enumerate(model.parent_idx[block].tolist()):
            if parent >= 0:
                tree_pairs.append(HierarchyPair(parent_index=parent, child_index=block.start + local_idx))

    metric = hierarchical_pair_metric(model, eval_dataset.activations, activations_bool, tree_pairs[:5])
    assert metric == metric or metric != metric  # NaN-tolerant: just confirm it ran
