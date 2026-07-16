"""Ablation of the orthogonality and L1 sparsity losses (Appendix B.2 spirit),
comparing dead-latent rate and reconstruction quality across configurations,
on synthetic activation data (the paper's own ablations run on the Gemma
2-2B unembedding matrix, which is out of scope here).

Usage:
    uv run python experiments/run_ablation.py --steps 300
"""

from __future__ import annotations

import argparse
import copy

import torch

from h_sae.config import HSAEConfig
from h_sae.data import SyntheticParentChildActivations, SyntheticTreeSpec
from h_sae.evaluation import dead_latent_rate, one_minus_explained_variance
from h_sae.training import HSAETrainer
from h_sae.utils import set_seed


def run_variant(name: str, config: HSAEConfig, train_dataset, eval_x, steps) -> None:
    trainer = HSAETrainer(config)
    trainer.train(train_dataset, total_steps=steps)
    with torch.no_grad():
        x_hat, z, _, _, _ = trainer.model(eval_x)
    ev = one_minus_explained_variance(eval_x, x_hat)
    dead = dead_latent_rate(z)
    print(f"  {name:<24s} 1-explained-var={ev:.4f}  dead-latent-rate={dead:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate H-SAE orthogonality/L1 losses")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_config = HSAEConfig.small_synthetic()
    base_config.seed = args.seed
    set_seed(base_config.seed)

    spec = SyntheticTreeSpec(
        d_model=base_config.model.d_model, m_top=base_config.model.m_top,
        m_low=base_config.model.m_low, subspace_dim=base_config.model.subspace_dim,
    )
    train_dataset = SyntheticParentChildActivations(
        spec, num_samples=max(base_config.data.training_vectors, base_config.optim.batch_size), seed=base_config.seed
    )
    eval_dataset = SyntheticParentChildActivations(spec, num_samples=500, seed=base_config.seed + 1)
    eval_x = eval_dataset.activations

    print("H-SAE loss ablation (Appendix B.2 spirit):")

    full_config = copy.deepcopy(base_config)
    run_variant("Baseline (full)", full_config, train_dataset, eval_x, args.steps)

    no_ortho = copy.deepcopy(base_config)
    no_ortho.loss.lambda_ortho = 0.0
    run_variant("No Orthogonality", no_ortho, train_dataset, eval_x, args.steps)

    no_l1 = copy.deepcopy(base_config)
    no_l1.loss.lambda_sparse = 0.0
    run_variant("No L1", no_l1, train_dataset, eval_x, args.steps)

    neither = copy.deepcopy(base_config)
    neither.loss.lambda_ortho = 0.0
    neither.loss.lambda_sparse = 0.0
    run_variant("No Orthogonality or L1", neither, train_dataset, eval_x, args.steps)


if __name__ == "__main__":
    main()
