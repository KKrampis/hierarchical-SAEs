"""Dynamic (Adaptive) Sparsity Control (Appendix A.2).

Maintains an EMA estimate of each level's average L0 and adjusts the
sparsity penalty weight lambda_l so the tracked L0 follows a first-order
exponential decay trajectory toward a target sparsity:

    Delta_L0_target = -eta * (L0_ema - L0_hat)

The paper describes the adjustment rule only qualitatively ("if the observed
reduction in sparsity is more rapid than the target trajectory, lambda_l is
decreased; if the model remains denser than expected, lambda_l is
increased") without giving the exact update magnitude. This is implemented
here as a small proportional/velocity feedback controller (see
pipeline/01_algorithm_extraction.yaml, "missing_but_critical", and
pipeline/03_implementation_plan.yaml Section 2.4) with a tunable gain
`kappa` and a per-step multiplicative clamp, both exposed via
SparsityConfig and clearly documented as an inferred approximation.
"""

from __future__ import annotations

import torch

from hsae.config import SparsityConfig


class DynamicSparsityController:
    def __init__(self, num_levels: int, config: SparsityConfig):
        self.config = config
        self.l0_ema = torch.full((num_levels,), config.target_l0)
        self.lambdas = torch.full((num_levels,), config.initial_lambda)
        self._prev_l0_ema = self.l0_ema.clone()

    @torch.no_grad()
    def step(self, observed_l0: torch.Tensor) -> torch.Tensor:
        """observed_l0: [num_levels] batch-average L0 per level for the current step.

        Updates the internal EMA and lambda_l values in place, and returns the
        updated lambda tensor (one value per level) to use for the *next*
        training step's L_SAE,l computation.
        """
        cfg = self.config
        self._prev_l0_ema = self.l0_ema.clone()
        self.l0_ema = cfg.ema_momentum * self.l0_ema + (1 - cfg.ema_momentum) * observed_l0

        delta_target = -cfg.decay_rate_eta * (self.l0_ema - cfg.target_l0)
        delta_observed = self.l0_ema - self._prev_l0_ema

        # If sparsity is falling faster than the target trajectory (delta_observed more
        # negative than delta_target), lambda_l should decrease; if the model is denser
        # than expected (delta_observed less negative / more positive than delta_target),
        # lambda_l should increase.
        error = delta_target - delta_observed
        multiplier = torch.exp(cfg.control_kappa * error)
        multiplier = multiplier.clamp(*cfg.lambda_step_clip)
        self.lambdas = (self.lambdas * multiplier).clamp_min(1e-8)
        return self.lambdas
