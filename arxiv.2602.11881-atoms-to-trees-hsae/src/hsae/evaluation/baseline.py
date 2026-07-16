"""Post-hoc Baseline construction (Section 4.2, "Baselines").

Trains a series of independent JumpReLU SAEs (no parent-children constraint
loss, no feature perturbation, no periodic hierarchy update during
training), matching the corresponding HSAE dictionary sizes. After training,
a *single* Similarity-Based Hierarchy Update + Partial Tree filter is applied
post-hoc to build a comparison hierarchy. This isolates the effect of joint
training vs. inferring structure after the fact from independently trained
SAEs.
"""

from __future__ import annotations

from hsae.config import HSAEConfig
from hsae.models.hsae import HSAE
from hsae.training.trainer import HSAETrainer


def train_baseline(config: HSAEConfig, dataset, total_steps: int | None = None) -> HSAE:
    trainer = HSAETrainer(
        config=config,
        use_constraint=False,
        use_perturbation=False,
        train_hierarchy=False,
    )
    trainer.train(dataset, total_steps=total_steps)

    model = trainer.model
    model.update_hierarchy(
        metric=config.hierarchy.similarity_metric,
        exclusion_quantile=config.hierarchy.exclusion_quantile,
    )
    return model
