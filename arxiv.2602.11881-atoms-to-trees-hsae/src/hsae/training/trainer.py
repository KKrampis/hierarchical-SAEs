"""Alternating-optimization trainer for HSAE (Section 3.3).

Each training step performs a Parameter Optimization update (gradient
descent on SAE weights with the hierarchical structure fixed). Every
`hierarchy_update_period_steps` steps, a Hierarchy Update is performed
instead (structure recomputed with weights fixed), so that model parameters
and the hierarchical assignment co-evolve throughout training.

Passing `use_constraint=False` and/or `use_perturbation=False` reproduces
the paper's ablations ("w/o Constraint", "w/o Perturbation", Section 4.5).
Passing `train_hierarchy=False` disables periodic hierarchy updates
entirely, which is how the post-hoc "Baseline" (Section 4.2 "Baselines")
independently-trained SAEs are produced before a single post-hoc hierarchy
assignment is applied (see evaluation/baseline.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from hsae.algorithms.sparsity_control import DynamicSparsityController
from hsae.config import HSAEConfig
from hsae.models.hsae import HSAE
from hsae.training.losses import compute_hsae_loss


def _cosine_warmup_lr_lambda(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class StepLog:
    step: int
    total_loss: float
    mse_per_level: list[float]
    l0_per_level: list[float]


class HSAETrainer:
    def __init__(
        self,
        config: HSAEConfig,
        use_constraint: bool = True,
        use_perturbation: bool = True,
        train_hierarchy: bool = True,
        max_hierarchy_updates: int | None = None,
        max_children_per_parent: int | None = None,
    ):
        self.config = config
        self.use_constraint = use_constraint
        self.use_perturbation = use_perturbation
        self.train_hierarchy = train_hierarchy
        # max_hierarchy_updates=1 reproduces the "Fix Tree" ablation (Appendix F.3): the
        # hierarchy is determined once early in training and then frozen.
        self.max_hierarchy_updates = max_hierarchy_updates
        # max_children_per_parent=2 reproduces the "Binary Tree" ablation (Appendix F.3).
        self.max_children_per_parent = max_children_per_parent
        self._hierarchy_update_count = 0

        self.model = HSAE(config.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.optim.learning_rate,
            betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_warmup_lr_lambda(
                step, config.optim.total_steps, config.optim.warmup_fraction
            ),
        )
        self.sparsity_controller = DynamicSparsityController(config.model.num_levels, config.sparsity)

    def _dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.optim.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def train(self, dataset, total_steps: int | None = None, log_every: int = 50) -> list[StepLog]:
        total_steps = total_steps or self.config.optim.total_steps
        rho = self.config.hierarchy.parent_child_constraint_weight_rho if self.use_constraint else 0.0
        perturbation_rate = self.config.hierarchy.perturbation_rate_r if self.use_perturbation else 0.0

        logs: list[StepLog] = []
        step = 0
        dataloader = self._dataloader(dataset)
        data_iter = iter(dataloader)

        while step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            should_update_hierarchy = (
                self.train_hierarchy
                and step > 0
                and step % self.config.hierarchy.hierarchy_update_period_steps == 0
                and (self.max_hierarchy_updates is None or self._hierarchy_update_count < self.max_hierarchy_updates)
            )
            if should_update_hierarchy:
                self.model.update_hierarchy(
                    metric=self.config.hierarchy.similarity_metric,
                    exclusion_quantile=self.config.hierarchy.exclusion_quantile,
                    max_children_per_parent=self.max_children_per_parent,
                )
                self._hierarchy_update_count += 1

            feature_acts, reconstructions, feature_outputs = self.model(
                batch, perturbation_rate=perturbation_rate if self.train_hierarchy else 0.0
            )

            observed_l0 = torch.stack([acts.gt(0).float().sum(dim=-1).mean() for acts in feature_acts])
            lambdas = self.sparsity_controller.step(observed_l0)

            loss_out = compute_hsae_loss(
                model=self.model,
                x=batch,
                feature_acts=feature_acts,
                reconstructions=reconstructions,
                feature_outputs=feature_outputs,
                lambdas=lambdas,
                rho=rho,
            )

            ghost_loss = sum(
                sae.ghost_grad_loss(batch, batch - reconstructions[i].detach(), self.config.optim.ghost_grad_dead_steps)
                for i, sae in enumerate(self.model.levels)
            )
            total = loss_out.total + 1e-2 * ghost_loss

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.model.post_step()

            with torch.no_grad():
                for sae, acts in zip(self.model.levels, feature_acts):
                    sae.update_dead_feature_tracker(acts)

            if step % log_every == 0:
                logs.append(
                    StepLog(
                        step=step,
                        total_loss=float(total.detach()),
                        mse_per_level=[float(m) for m in loss_out.mse_per_level],
                        l0_per_level=[float(l) for l in loss_out.l0_per_level],
                    )
                )
            step += 1

        return logs
