"""Train an HSAE and a post-hoc Baseline on the same synthetic data, then
report hierarchy-consistency metrics for both (reproducing the *shape* of
Figure 3's comparison, at synthetic/CPU scale rather than the paper's
full gemma2-2b scale).

Usage:
    uv run python experiments/run_eval.py --steps 300
"""

from __future__ import annotations

import argparse

from hsae.config import HSAEConfig
from hsae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from hsae.evaluation import average_metrics, evaluate_hierarchy, train_baseline
from hsae.training import HSAETrainer
from hsae.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HSAE vs. post-hoc Baseline hierarchy consistency")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = HSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    spec = SyntheticTreeSpec(d_model=config.model.d_model, dict_sizes=config.model.dict_sizes)
    train_dataset = SyntheticHierarchicalActivations(spec, num_samples=config.data.dataset_size, seed=config.seed)
    eval_dataset = SyntheticHierarchicalActivations(spec, num_samples=config.data.dataset_size, seed=config.seed + 1)
    eval_activations = eval_dataset.activations

    hsae_trainer = HSAETrainer(config)
    hsae_trainer.train(train_dataset, total_steps=args.steps)
    hsae_metrics = average_metrics(evaluate_hierarchy(hsae_trainer.model, eval_activations))

    baseline_model = train_baseline(config, train_dataset, total_steps=args.steps)
    baseline_metrics = average_metrics(evaluate_hierarchy(baseline_model, eval_activations))

    print("Hierarchy-consistency comparison (lower Hamming distance is better; higher P(.|.) is better):")
    print(f"  HSAE:     hamming={hsae_metrics.hamming_distance:.3f}  "
          f"P(parent|child)={hsae_metrics.parent_given_child:.3f}  "
          f"P(child|parent)={hsae_metrics.child_given_parent:.3f}  "
          f"pairs={hsae_metrics.num_assigned_pairs}")
    print(f"  Baseline: hamming={baseline_metrics.hamming_distance:.3f}  "
          f"P(parent|child)={baseline_metrics.parent_given_child:.3f}  "
          f"P(child|parent)={baseline_metrics.child_given_parent:.3f}  "
          f"pairs={baseline_metrics.num_assigned_pairs}")


if __name__ == "__main__":
    main()
