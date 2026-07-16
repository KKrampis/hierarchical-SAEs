"""Train a Tree SAE on synthetic activations.

Usage:
    uv run python experiments/run_train.py --steps 300
"""

from __future__ import annotations

import argparse

from tree_sae.config import TreeSAEConfig
from tree_sae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from tree_sae.training import TreeSAETrainer
from tree_sae.utils import set_seed


def build_dataset(config: TreeSAEConfig):
    spec = SyntheticTreeSpec(d_model=config.model.d_model, layer_sizes=config.model.layer_sizes)
    num_samples = max(config.data.training_tokens, config.optim.batch_size)
    return SyntheticHierarchicalActivations(spec, num_samples=num_samples, seed=config.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tree SAE on synthetic activations")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TreeSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    dataset = build_dataset(config)
    trainer = TreeSAETrainer(config)
    logs = trainer.train(dataset, total_steps=args.steps)

    for log in logs[-3:]:
        print(f"step={log.step} total_loss={log.total_loss:.4f} recons_loss={log.recons_loss:.4f}")
    print(f"final dead-feature rate: {trainer.dead_feature_rate():.3f}")


if __name__ == "__main__":
    main()
