"""Train an HSAE (or, with --baseline, independent SAEs for the post-hoc baseline).

Usage:
    uv run python experiments/run_train.py --synthetic --steps 300
"""

from __future__ import annotations

import argparse

from hsae.config import HSAEConfig
from hsae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from hsae.evaluation import train_baseline
from hsae.training import HSAETrainer
from hsae.utils import set_seed


def build_dataset(config: HSAEConfig):
    spec = SyntheticTreeSpec(d_model=config.model.d_model, dict_sizes=config.model.dict_sizes)
    return SyntheticHierarchicalActivations(spec, num_samples=config.data.dataset_size, seed=config.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HSAE on synthetic activations")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data (default)")
    parser.add_argument("--baseline", action="store_true", help="Train the post-hoc Baseline instead of HSAE")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = HSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    dataset = build_dataset(config)

    if args.baseline:
        model = train_baseline(config, dataset, total_steps=args.steps)
    else:
        trainer = HSAETrainer(config)
        logs = trainer.train(dataset, total_steps=args.steps)
        model = trainer.model
        for log in logs[-3:]:
            print(f"step={log.step} loss={log.total_loss:.4f} l0={log.l0_per_level}")

    for level in range(model.num_levels - 1):
        num_assigned = sum(1 for p in model.parent_idx[level].tolist() if p != -1)
        print(f"level {level}->{level + 1}: {num_assigned}/{model.parent_idx[level].numel()} features assigned a parent")


if __name__ == "__main__":
    main()
