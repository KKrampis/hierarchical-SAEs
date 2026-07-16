"""Train an H-SAE on synthetic activations.

Usage:
    uv run python experiments/run_train.py --steps 300
"""

from __future__ import annotations

import argparse

from h_sae.config import HSAEConfig
from h_sae.data import SyntheticParentChildActivations, SyntheticTreeSpec
from h_sae.training import HSAETrainer
from h_sae.utils import set_seed


def build_dataset(config: HSAEConfig):
    spec = SyntheticTreeSpec(
        d_model=config.model.d_model,
        m_top=config.model.m_top,
        m_low=config.model.m_low,
        subspace_dim=config.model.subspace_dim,
    )
    num_samples = max(config.data.training_vectors, config.optim.batch_size)
    return SyntheticParentChildActivations(spec, num_samples=num_samples, seed=config.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H-SAE on synthetic activations")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = HSAEConfig.small_synthetic()
    config.seed = args.seed
    set_seed(config.seed)

    dataset = build_dataset(config)
    trainer = HSAETrainer(config)
    logs = trainer.train(dataset, total_steps=args.steps)

    for log in logs[-3:]:
        print(f"step={log.step} total_loss={log.total_loss:.4f} recon_loss={log.recon_loss:.4f}")


if __name__ == "__main__":
    main()
