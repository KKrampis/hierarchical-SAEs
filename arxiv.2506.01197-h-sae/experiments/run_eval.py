"""Train H-SAE and the baseline TopK SAE at matched top-level dictionary
size, and compare reconstruction quality, dead-latent rate, and a
simplified absorption/splitting score (Section 4, 4.1).

Usage:
    uv run python experiments/run_eval.py --steps 300
"""

from __future__ import annotations

import argparse

import torch

from h_sae.config import HSAEConfig
from h_sae.data import SyntheticParentChildActivations, SyntheticTreeSpec
from h_sae.evaluation import dead_latent_rate, one_minus_explained_variance, simplified_absorption_score
from h_sae.training import HSAETrainer, TopKSAETrainer
from h_sae.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate H-SAE vs. baseline TopK SAE")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = HSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    spec = SyntheticTreeSpec(
        d_model=config.model.d_model, m_top=config.model.m_top,
        m_low=config.model.m_low, subspace_dim=config.model.subspace_dim,
    )
    train_dataset = SyntheticParentChildActivations(
        spec, num_samples=max(config.data.training_vectors, config.optim.batch_size), seed=config.seed
    )
    eval_dataset = SyntheticParentChildActivations(spec, num_samples=500, seed=config.seed + 1)
    eval_x = eval_dataset.activations

    hsae_trainer = HSAETrainer(config)
    hsae_trainer.train(train_dataset, total_steps=args.steps)
    with torch.no_grad():
        x_hat, z, _, _, _ = hsae_trainer.model(eval_x)
    hsae_ev = one_minus_explained_variance(eval_x, x_hat)
    hsae_dead = dead_latent_rate(z)
    hsae_absorption = simplified_absorption_score(eval_x, z, eval_dataset.parent_labels, target_label=0, decoder=hsae_trainer.model.D)

    baseline_trainer = TopKSAETrainer(config)
    baseline_trainer.train(train_dataset, total_steps=args.steps)
    with torch.no_grad():
        x_hat_baseline, z_baseline = baseline_trainer.model(eval_x)
    baseline_ev = one_minus_explained_variance(eval_x, x_hat_baseline)
    baseline_dead = dead_latent_rate(z_baseline)
    baseline_absorption = simplified_absorption_score(eval_x, z_baseline, eval_dataset.parent_labels, target_label=0, decoder=baseline_trainer.model.D)

    print("Reconstruction (1 - explained variance, lower is better):")
    print(f"  H-SAE:    {hsae_ev:.4f}")
    print(f"  Baseline: {baseline_ev:.4f}")
    print("Dead top-level latent rate:")
    print(f"  H-SAE:    {hsae_dead:.3f}")
    print(f"  Baseline: {baseline_dead:.3f}")
    print("Simplified absorption score (lower = less splitting/absorption):")
    print(f"  H-SAE:    {hsae_absorption:.3f}")
    print(f"  Baseline: {baseline_absorption:.3f}")


if __name__ == "__main__":
    main()
