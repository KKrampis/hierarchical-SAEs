"""Compare dead-feature rate with vs. without dynamic allocation (Algorithm 2),
a small/CPU-scale reproduction of the *direction* of Figure 14.

Caveat: at tiny synthetic scale (a few hundred steps, dictionary sizes in the
tens) dead-feature-rate trajectories are noisy -- this script demonstrates
that dynamic allocation is wired up and runnable end-to-end, not a strict
pass/fail correctness test (see tests/test_smoke.py for that). The paper's
full-scale result (near-zero dead-feature rate by 40k steps with dynamic
allocation, vs. persistent dead features without it, on a 4-layer L0=32
GPT2-small Tree SAE) is recorded in pipeline/01_algorithm_extraction.yaml.

Usage:
    uv run python experiments/run_ablation.py --steps 300
"""

from __future__ import annotations

import argparse

from tree_sae.config import TreeSAEConfig
from tree_sae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from tree_sae.training import TreeSAETrainer
from tree_sae.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic allocation ablation (Figure 14 direction)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TreeSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    dataset = SyntheticHierarchicalActivations(
        spec, num_samples=max(config.data.training_tokens, config.optim.batch_size), seed=config.seed
    )

    with_alloc = TreeSAETrainer(config, enable_dynamic_allocation=True)
    with_alloc.train(dataset, total_steps=args.steps)
    with_rate = with_alloc.dead_feature_rate()

    without_alloc = TreeSAETrainer(config, enable_dynamic_allocation=False)
    without_alloc.train(dataset, total_steps=args.steps)
    without_rate = without_alloc.dead_feature_rate()

    print(f"Dead-feature rate WITH dynamic allocation:    {with_rate:.3f}")
    print(f"Dead-feature rate WITHOUT dynamic allocation: {without_rate:.3f}")


if __name__ == "__main__":
    main()
