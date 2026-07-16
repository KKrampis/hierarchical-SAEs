"""Fast CPU smoke test: exercises the full HSAE pipeline end-to-end on
synthetic data, per the "Validation Approach" in
pipeline/03_implementation_plan.yaml (Section 1: Unit Tests, Section 2:
Integration Tests).
"""

from __future__ import annotations

import torch

from hsae.algorithms.hierarchy import UNASSIGNED, assign_parents
from hsae.config import HSAEConfig
from hsae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from hsae.evaluation import average_metrics, evaluate_hierarchy, train_baseline
from hsae.models.hsae import HSAE
from hsae.training import HSAETrainer
from hsae.utils import set_seed


def test_config_loads():
    config = HSAEConfig.small_synthetic()
    assert config.model.num_levels == len(config.model.dict_sizes) == 3


def test_hsae_forward_shapes():
    config = HSAEConfig.small_synthetic()
    model = HSAE(config.model)
    x = torch.randn(4, config.model.d_model)
    feature_acts, reconstructions, feature_outputs = model(x, perturbation_rate=0.0)

    assert len(feature_acts) == model.num_levels
    for level, dict_size in enumerate(config.model.dict_sizes):
        assert feature_acts[level].shape == (4, dict_size)
        assert reconstructions[level].shape == (4, config.model.d_model)
        assert feature_outputs[level].shape == (4, dict_size, config.model.d_model)


def test_assign_parents_respects_quantile_and_disjointness():
    torch.manual_seed(0)
    similarity = torch.rand(40, 10)
    parent_idx = assign_parents(similarity, exclusion_quantile=0.20)

    assigned = parent_idx[parent_idx != UNASSIGNED]
    # Roughly 20% should be excluded (allow slack for the discrete quantile boundary).
    frac_unassigned = (parent_idx == UNASSIGNED).float().mean().item()
    assert 0.05 <= frac_unassigned <= 0.40

    # Disjointness is automatic (each child has exactly one parent), but verify no
    # child index appears twice across per-parent children lists.
    from hsae.algorithms.hierarchy import children_sets

    groups = children_sets(parent_idx, n_parents=10)
    all_children = [child for group in groups for child in group]
    assert len(all_children) == len(set(all_children)) == assigned.numel()


def test_parent_children_loss_zero_for_empty_children():
    from hsae.training.losses import parent_children_loss

    parent_outputs = torch.randn(3, 5, 8)
    children_sum = torch.zeros(3, 5, 8)
    parent_idx = torch.full((5,), UNASSIGNED, dtype=torch.long)  # no child assigned to any parent
    loss = parent_children_loss(parent_outputs, children_sum, parent_idx, n_parents=5)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_hierarchy_update_produces_valid_assignment():
    config = HSAEConfig.small_synthetic()
    model = HSAE(config.model)
    model.update_hierarchy(
        metric=config.hierarchy.similarity_metric,
        exclusion_quantile=config.hierarchy.exclusion_quantile,
    )
    for level in range(model.num_levels - 1):
        parent_idx = model.parent_idx[level]
        assigned = parent_idx[parent_idx != UNASSIGNED]
        assert torch.all(assigned >= 0)
        assert torch.all(assigned < model.levels[level].dict_size)


def test_training_reduces_reconstruction_loss():
    set_seed(0)
    config = HSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(d_model=config.model.d_model, dict_sizes=config.model.dict_sizes)
    dataset = SyntheticHierarchicalActivations(spec, num_samples=config.data.dataset_size, seed=0)

    trainer = HSAETrainer(config)
    logs = trainer.train(dataset, total_steps=config.optim.total_steps, log_every=25)

    assert len(logs) >= 2
    first_mse = sum(logs[0].mse_per_level)
    last_mse = sum(logs[-1].mse_per_level)
    assert last_mse < first_mse


def test_evaluation_and_baseline_run_end_to_end():
    set_seed(0)
    config = HSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(d_model=config.model.d_model, dict_sizes=config.model.dict_sizes)
    train_dataset = SyntheticHierarchicalActivations(spec, num_samples=config.data.dataset_size, seed=0)
    eval_dataset = SyntheticHierarchicalActivations(spec, num_samples=200, seed=1)

    trainer = HSAETrainer(config)
    trainer.train(train_dataset, total_steps=config.optim.total_steps)
    hsae_metrics = average_metrics(evaluate_hierarchy(trainer.model, eval_dataset.activations))
    assert hsae_metrics.num_assigned_pairs > 0
    assert hsae_metrics.hamming_distance == hsae_metrics.hamming_distance  # not NaN

    baseline_model = train_baseline(config, train_dataset, total_steps=config.optim.total_steps)
    baseline_metrics = average_metrics(evaluate_hierarchy(baseline_model, eval_dataset.activations))
    assert baseline_metrics.num_assigned_pairs > 0
