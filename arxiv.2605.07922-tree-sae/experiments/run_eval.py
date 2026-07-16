"""Train a Tree SAE and evaluate its built-in Tree Structure hierarchy
detection against the MCS post-hoc baseline (Section 5.2), plus the
reconstruction condition score S_res for a handful of sampled pairs.

Usage:
    uv run python experiments/run_eval.py --steps 300
"""

from __future__ import annotations

import argparse

import torch

from tree_sae.config import TreeSAEConfig
from tree_sae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from tree_sae.evaluation import (
    HierarchyPair,
    activation_coverage_score,
    binarize,
    hierarchical_pair_metric,
    reconstruction_score,
    sample_dense_parents,
    top_k_children_by_mcs,
    train_linear_probe,
)
from tree_sae.training import TreeSAETrainer
from tree_sae.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tree SAE hierarchy discovery vs. MCS baseline")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-parents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TreeSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    train_dataset = SyntheticHierarchicalActivations(
        spec, num_samples=max(config.data.training_tokens, config.optim.batch_size), seed=config.seed
    )
    eval_dataset = SyntheticHierarchicalActivations(spec, num_samples=500, seed=config.seed + 1)

    trainer = TreeSAETrainer(config)
    trainer.train(train_dataset, total_steps=args.steps)
    model = trainer.model

    with torch.no_grad():
        _, gated, _ = model(eval_dataset.activations)
    activations_bool = binarize(gated)

    # Tree Structure pairs (this model's own allocation).
    tree_pairs = []
    for level in range(1, model.num_levels):
        block = model.layer_slice(level)
        for local_idx, parent in enumerate(model.parent_idx[block].tolist()):
            if parent >= 0:
                tree_pairs.append(HierarchyPair(parent_index=parent, child_index=block.start + local_idx))

    tree_metric = hierarchical_pair_metric(model, eval_dataset.activations, activations_bool, tree_pairs[: args.num_parents * 3])

    # MCS baseline pairs: for each sampled dense parent, take its top-5 MCS-correlated candidates
    # among all EARLIER-layer-eligible-pool features (mirroring the paper's setup where MCS is
    # applied to the same trained SAE's independent features, ignoring the built-in allocation).
    dense_parents = sample_dense_parents(activations_bool, list(range(model.offsets[1])), args.num_parents, seed=config.seed)
    mcs_pairs = []
    for parent in dense_parents:
        candidates = [i for i in range(model.offsets[1], model.dict_size)]
        children = top_k_children_by_mcs(activations_bool, parent, candidates, k=5)
        mcs_pairs.extend(HierarchyPair(parent_index=parent, child_index=c) for c in children)

    mcs_metric = hierarchical_pair_metric(model, eval_dataset.activations, activations_bool, mcs_pairs)

    print(f"Tree Structure hierarchical-pair metric: {tree_metric:.3f} ({len(tree_pairs)} pairs found)")
    print(f"MCS baseline hierarchical-pair metric:   {mcs_metric:.3f} ({len(mcs_pairs)} pairs evaluated)")

    if tree_pairs:
        pair = tree_pairs[0]
        child_active = activations_bool[:, pair.child_index]
        if child_active.any() and (~child_active).any():
            direction = train_linear_probe(eval_dataset.activations, child_active)
            s_res = reconstruction_score(model.W_dec[pair.parent_index], model.W_dec[pair.child_index], direction)
            s_cov = activation_coverage_score(activations_bool[:, pair.parent_index], child_active)
            print(f"Example pair (parent={pair.parent_index}, child={pair.child_index}): S_cov={s_cov:.3f} S_res={s_res:.3f}")


if __name__ == "__main__":
    main()
