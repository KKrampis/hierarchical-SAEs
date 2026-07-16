"""Training loop for Tree SAE (Section 4, Appendix C, Appendix G).

Each step performs a standard gradient-descent update of the shared
encoder/decoder using the multi-level reconstruction + auxiliary loss
(Eq. 7-9). On a periodic, growing schedule (Appendix G), Algorithm 2 is
invoked per layer to reallocate dead features toward under-served parents.
At approximately half of training, all remaining dead features are moved
to the root node so they can activate freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from tree_sae.algorithms.allocation import dynamic_reallocation, reassign_dead_to_root
from tree_sae.config import TreeSAEConfig
from tree_sae.models.tree_sae import TreeSAE
from tree_sae.training.losses import compute_tree_sae_loss


@dataclass
class StepLog:
    step: int
    total_loss: float
    recons_loss: float


class TreeSAETrainer:
    def __init__(self, config: TreeSAEConfig, enable_dynamic_allocation: bool = True):
        self.config = config
        self.enable_dynamic_allocation = enable_dynamic_allocation
        self.model = TreeSAE(config.model)
        self._init_random_allocation()
        self.model.register_buffer("activation_count", torch.zeros(self.model.dict_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.optim.learning_rate)
        self.register_buffer_capacity = torch.zeros(self.model.dict_size)
        self.total_tokens_seen = 0
        self._next_reallocation_step = config.allocation.reallocation_initial_interval_steps
        self._current_interval = config.allocation.reallocation_initial_interval_steps
        self._dead_to_root_done = False

    def _init_random_allocation(self) -> None:
        """INFERRED default (see pipeline/01_algorithm_extraction.yaml
        missing_but_critical): initialize each non-layer-1 feature's parent
        uniformly at random among all features at strictly earlier layers."""
        generator = torch.Generator().manual_seed(self.config.seed)
        for level in range(1, self.model.num_levels):
            block = self.model.layer_slice(level)
            m_l = self.model.offsets[level]
            random_parents = torch.randint(0, m_l, (self.model.layer_sizes[level],), generator=generator)
            self.model.parent_idx[block] = random_parents

    def _dataloader(self, dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.config.optim.batch_size, shuffle=True, drop_last=True)

    def _dead_masks(self) -> list[torch.Tensor]:
        threshold = self.config.aux.dead_feature_window_tokens
        dead = self.model.is_dead(threshold)
        return [dead[self.model.layer_slice(level)] for level in range(self.model.num_levels)]

    def _eligible_mask(self, m_l: int) -> torch.Tensor:
        if self.total_tokens_seen == 0:
            return torch.zeros(m_l, dtype=torch.bool)
        rate = self.model.activation_count[:m_l].float() / self.total_tokens_seen
        return rate >= self.config.allocation.parent_eligibility_min_rate

    def _maybe_reallocate(self, step: int) -> None:
        if not self.enable_dynamic_allocation:
            return
        if step < self._next_reallocation_step:
            return
        dead_masks = self._dead_masks()
        for level in range(1, self.model.num_levels):
            m_l = self.model.offsets[level]
            capacities = self.register_buffer_capacity[:m_l]
            eligible = self._eligible_mask(m_l)
            dynamic_reallocation(self.model, level, capacities, eligible, dead_masks[level])
        self.register_buffer_capacity.zero_()

        alloc_cfg = self.config.allocation
        self._current_interval = min(
            self._current_interval + alloc_cfg.reallocation_interval_increment_steps,
            alloc_cfg.reallocation_interval_cap_steps,
        )
        self._next_reallocation_step = step + self._current_interval

    def _maybe_dead_to_root(self, step: int, total_steps: int) -> None:
        if not self.enable_dynamic_allocation:
            return
        if self._dead_to_root_done:
            return
        if step >= int(total_steps * self.config.allocation.dead_to_root_step_fraction):
            reassign_dead_to_root(self.model, self.config.aux.dead_feature_window_tokens)
            self._dead_to_root_done = True

    def train(self, dataset, total_steps: int | None = None, log_every: int = 50) -> list[StepLog]:
        total_steps = total_steps or self.config.optim.total_steps
        aux_topk = self.config.aux.aux_topk
        layer_alphas = self.config.aux.layer_alphas

        logs: list[StepLog] = []
        dataloader = self._dataloader(dataset)
        data_iter = iter(dataloader)

        for step in range(total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pre_activation = self.model.encode_pre_activation(batch)
            raw = self.model.topk_raw(pre_activation)
            gated = self.model.gate(raw)
            cumulative = self.model.cumulative_reconstructions(gated)

            dead_masks = self._dead_masks()
            loss_out = compute_tree_sae_loss(
                self.model, batch, pre_activation, cumulative, dead_masks, aux_topk, layer_alphas
            )

            self.optimizer.zero_grad()
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip_norm)
            self.optimizer.step()
            self.model.post_step()

            with torch.no_grad():
                self.model.update_dead_feature_tracker(raw)
                active_counts = (raw > 0).sum(dim=0).float()
                self.model.activation_count += active_counts
                self.register_buffer_capacity += active_counts * float(loss_out.total.detach())
                self.total_tokens_seen += batch.shape[0]

            self._maybe_reallocate(step)
            self._maybe_dead_to_root(step, total_steps)

            if step % log_every == 0:
                logs.append(
                    StepLog(step=step, total_loss=float(loss_out.total.detach()), recons_loss=float(loss_out.recons_loss.detach()))
                )

        return logs

    def dead_feature_rate(self) -> float:
        dead = self.model.is_dead(self.config.aux.dead_feature_window_tokens)
        return dead.float().mean().item()
