"""Training loops for H-SAE and the baseline TopK SAE (Appendix A.2-A.3).

Both models share the same optimizer/schedule (Adam, global-norm clipping,
linear-warmup-then-cosine-decay learning rate); H-SAE additionally warms up
its loss coefficients (beta, lambda_ortho, lambda_sparse) over the same
warmup window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from h_sae.config import HSAEConfig
from h_sae.models.hsae import HSAE
from h_sae.models.topk_sae import TopKSAE
from h_sae.training.losses import compute_hsae_loss, warmup_coefficient


def _lr_lambda(step: int, total_steps: int, warmup_steps: int, initial_lr: float, peak_lr: float) -> float:
    """Returns a multiplier on `peak_lr` (for use with LambdaLR, base_lr=1.0 is not used here;
    instead the trainer sets `optimizer.param_groups[0]['lr']` directly each step)."""
    if step < warmup_steps:
        progress = step / max(1, warmup_steps)
        lr = initial_lr + progress * (peak_lr - initial_lr)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr


@dataclass
class StepLog:
    step: int
    total_loss: float
    recon_loss: float


class HSAETrainer:
    def __init__(self, config: HSAEConfig):
        self.config = config
        self.model = HSAE(config.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.optim.initial_learning_rate,
            betas=(config.optim.adam_b1, config.optim.adam_b2),
        )

    def _set_lr(self, step: int, total_steps: int) -> None:
        lr = _lr_lambda(
            step, total_steps, self.config.optim.warmup_steps,
            self.config.optim.initial_learning_rate, self.config.optim.peak_learning_rate,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self, dataset, total_steps: int | None = None, log_every: int = 50) -> list[StepLog]:
        total_steps = total_steps or self.config.optim.total_steps
        dataloader = DataLoader(dataset, batch_size=self.config.optim.batch_size, shuffle=True, drop_last=True)
        data_iter = iter(dataloader)

        logs: list[StepLog] = []
        for step in range(total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            self._set_lr(step, total_steps)

            x_hat, z, x_hat_high, z_low, _ = self.model(batch)
            beta = warmup_coefficient(step, self.config.loss.warmup_steps, self.config.loss.beta)
            lambda_ortho = warmup_coefficient(step, self.config.loss.warmup_steps, self.config.loss.lambda_ortho)
            lambda_sparse = warmup_coefficient(step, self.config.loss.warmup_steps, self.config.loss.lambda_sparse)

            loss_out = compute_hsae_loss(self.model, batch, x_hat, z, x_hat_high, z_low, beta, lambda_ortho, lambda_sparse)

            self.optimizer.zero_grad()
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip_norm)
            self.optimizer.step()
            self.model.post_step()

            if step % log_every == 0:
                logs.append(StepLog(step=step, total_loss=float(loss_out.total.detach()), recon_loss=float(loss_out.recon_loss.detach())))

        return logs


class TopKSAETrainer:
    """Trains the baseline TopK SAE (Eq. 2.1-2.2) for comparison purposes.

    Simplification: the paper notes standard TopK SAEs need an auxiliary
    dead-latent loss (coefficient 1/30); this is omitted here since dead
    latents are not a significant concern at the small synthetic scale used
    for validation in this repository, and the baseline's exact dead-latent
    handling is not the subject of this paper's contribution (see
    pipeline/03_implementation_plan.yaml, Phase 3).
    """

    def __init__(self, config: HSAEConfig):
        self.config = config
        self.model = TopKSAE.from_config(config.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.optim.initial_learning_rate,
            betas=(config.optim.adam_b1, config.optim.adam_b2),
        )

    def _set_lr(self, step: int, total_steps: int) -> None:
        lr = _lr_lambda(
            step, total_steps, self.config.optim.warmup_steps,
            self.config.optim.initial_learning_rate, self.config.optim.peak_learning_rate,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self, dataset, total_steps: int | None = None, log_every: int = 50) -> list[StepLog]:
        total_steps = total_steps or self.config.optim.total_steps
        dataloader = DataLoader(dataset, batch_size=self.config.optim.batch_size, shuffle=True, drop_last=True)
        data_iter = iter(dataloader)

        logs: list[StepLog] = []
        for step in range(total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            self._set_lr(step, total_steps)

            x_hat, _ = self.model(batch)
            loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip_norm)
            self.optimizer.step()

            if step % log_every == 0:
                logs.append(StepLog(step=step, total_loss=float(loss.detach()), recon_loss=float(loss.detach())))

        return logs
