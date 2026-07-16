"""Run the paper's ablation studies (Section 4.5 / Figure 4: mechanism
ablation; Appendix F.3 / Figure 9: tree topology ablation) on synthetic/CPU
scale data, as a demonstration that every ablated mechanism is actually
wired up and runnable end-to-end.

Caveat: at this tiny scale (dict sizes ~8-32, a few hundred steps, run on
CPU for speed) hierarchy-consistency metrics have high run-to-run variance,
so the exact ranking between variants is not guaranteed to match the
paper's qualitative ordering (full HSAE best, then w/o Perturbation, then
w/o Constraint, then post-hoc Baseline) on every invocation -- unlike
`tests/test_smoke.py`, this script is a demonstration/reproduction tool, not
a pass/fail correctness test. The paper's full-scale numeric targets
(gemma2-2b, 100M activations, A800 GPU) are recorded in
pipeline/01_algorithm_extraction.yaml; increasing `--steps` and the
dictionary sizes in HSAEConfig moves this script's results toward that
regime.

Usage:
    uv run python experiments/run_ablation.py --steps 300
"""

from __future__ import annotations

import argparse
import copy

from hsae.config import HSAEConfig
from hsae.data import SyntheticHierarchicalActivations, SyntheticTreeSpec
from hsae.evaluation import average_metrics, evaluate_hierarchy, train_baseline
from hsae.training import HSAETrainer
from hsae.utils import set_seed


def run_variant(name: str, config: HSAEConfig, train_dataset, eval_activations, steps, **trainer_kwargs) -> None:
    trainer = HSAETrainer(config=config, **trainer_kwargs)
    trainer.train(train_dataset, total_steps=steps)
    metrics = average_metrics(evaluate_hierarchy(trainer.model, eval_activations))
    print(f"  {name:<16s} hamming={metrics.hamming_distance:.3f}  "
          f"P(parent|child)={metrics.parent_given_child:.3f}  "
          f"P(child|parent)={metrics.child_given_parent:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce HSAE ablation studies on synthetic data")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_config = HSAEConfig.small_synthetic()
    base_config.seed = args.seed
    set_seed(base_config.seed)

    spec = SyntheticTreeSpec(d_model=base_config.model.d_model, dict_sizes=base_config.model.dict_sizes)
    train_dataset = SyntheticHierarchicalActivations(spec, num_samples=base_config.data.dataset_size, seed=base_config.seed)
    eval_dataset = SyntheticHierarchicalActivations(spec, num_samples=base_config.data.dataset_size, seed=base_config.seed + 1)
    eval_activations = eval_dataset.activations

    print("Mechanism ablation (Section 4.5 / Figure 4) -- lower hamming distance is better:")
    baseline_model = train_baseline(copy.deepcopy(base_config), train_dataset, total_steps=args.steps)
    baseline_metrics = average_metrics(evaluate_hierarchy(baseline_model, eval_activations))
    print(f"  {'Baseline':<16s} hamming={baseline_metrics.hamming_distance:.3f}  "
          f"P(parent|child)={baseline_metrics.parent_given_child:.3f}  "
          f"P(child|parent)={baseline_metrics.child_given_parent:.3f}")
    run_variant("w/o Constraint", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps,
                use_constraint=False, use_perturbation=True)
    run_variant("w/o Perturbation", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps,
                use_constraint=True, use_perturbation=False)
    run_variant("HSAE (full)", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps,
                use_constraint=True, use_perturbation=True)

    print("\nTree topology ablation (Appendix F.3 / Figure 9) -- lower hamming distance is better:")
    run_variant("Fix Tree", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps,
                max_hierarchy_updates=1)
    full_tree_config = copy.deepcopy(base_config)
    full_tree_config.hierarchy.exclusion_quantile = 0.0
    run_variant("Full Tree", full_tree_config, train_dataset, eval_activations, args.steps)
    run_variant("Binary Tree", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps,
                max_children_per_parent=2)
    run_variant("HSAE (full)", copy.deepcopy(base_config), train_dataset, eval_activations, args.steps)


if __name__ == "__main__":
    main()
