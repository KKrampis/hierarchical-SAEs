"""Single-level JumpReLU sparse autoencoder.

Implements SAE_l(x) = sum_i d_{l,i} * sigma(e_{l,i}^T x) from
pipeline/01_algorithm_extraction.yaml ("Per-level JumpReLU SAE"), following
the JumpReLU SAE architecture of Rajamanoharan et al. (2024b) that the paper
builds on (Section 3, unnumbered intro).

sigma is JumpReLU: sigma(z) = z * H(z - theta), where theta is a learnable
per-feature threshold and H is the Heaviside step function. Because H is
non-differentiable, we use a straight-through estimator with a rectangular
kernel around the threshold, as is standard for JumpReLU SAEs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _JumpReLUSTE(torch.autograd.Function):
    """Straight-through estimator for z * H(z - theta)."""

    @staticmethod
    def forward(ctx, pre_activation: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_activation, threshold)
        ctx.bandwidth = bandwidth
        gate = (pre_activation > threshold).to(pre_activation.dtype)
        return pre_activation * gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_activation, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        gate = (pre_activation > threshold).to(pre_activation.dtype)
        # Straight-through: treat d(gate)/d(pre_activation) as 0 (pass grad only where active).
        grad_pre_activation = grad_output * gate
        # Rectangular-kernel pseudo-derivative of the Heaviside step w.r.t. threshold.
        kernel = ((pre_activation - threshold).abs() < (bandwidth / 2)).to(pre_activation.dtype)
        kernel = kernel / bandwidth
        grad_threshold = -(grad_output * pre_activation * kernel).sum(dim=0)
        return grad_pre_activation, grad_threshold, None


class _StepSTE(torch.autograd.Function):
    """Straight-through estimator for the bare indicator H(z - theta), used for the L0 penalty."""

    @staticmethod
    def forward(ctx, pre_activation: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_activation, threshold)
        ctx.bandwidth = bandwidth
        return (pre_activation > threshold).to(pre_activation.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_activation, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        kernel = ((pre_activation - threshold).abs() < (bandwidth / 2)).to(pre_activation.dtype)
        kernel = kernel / bandwidth
        grad_threshold = -(grad_output * kernel).sum(dim=0)
        return None, grad_threshold, None


class JumpReLUSAE(nn.Module):
    """A single hierarchy level's SAE: encoder, learnable threshold, decoder."""

    def __init__(self, d_model: int, dict_size: int, bandwidth: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.bandwidth = bandwidth

        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # log-threshold parameterization keeps thresholds positive.
        self.log_threshold = nn.Parameter(torch.full((dict_size,), fill_value=-1.0))

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._renormalize_decoder()

        self.register_buffer("steps_since_fired", torch.zeros(dict_size, dtype=torch.long))

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def encode_pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigma(e^T x): the sparse, thresholded feature activations."""
        pre_activation = self.encode_pre_activation(x)
        return _JumpReLUSTE.apply(pre_activation, self.threshold, self.bandwidth)

    def active_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the (straight-through-differentiable) L0 indicator 1{sigma(e^T x) > 0}."""
        pre_activation = self.encode_pre_activation(x)
        return _StepSTE.apply(pre_activation, self.threshold, self.bandwidth)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard (un-perturbed) forward pass: returns (reconstruction, feature_acts)."""
        feature_acts = self.encode(x)
        recon = self.decode(feature_acts)
        return recon, feature_acts

    def feature_outputs(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Per-feature decoder contribution d_i * sigma(e_i^T x), shape [B, dict_size, d_model].

        This materializes f_i(x) = d_i * sigma(e_i^T x) for every feature, which is what the
        parent-children constraint loss and perturbation mechanism operate on (they need each
        feature's individual contribution, not just the summed reconstruction).
        """
        return feature_acts.unsqueeze(-1) * self.W_dec.unsqueeze(0)

    @torch.no_grad()
    def _renormalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def post_step(self) -> None:
        """Call after every optimizer step: fixes decoder norms (Appendix A.1)."""
        self._renormalize_decoder()

    @torch.no_grad()
    def update_dead_feature_tracker(self, feature_acts: torch.Tensor) -> None:
        fired = feature_acts.abs().sum(dim=0) > 0
        self.steps_since_fired[fired] = 0
        self.steps_since_fired[~fired] += 1

    def ghost_grad_loss(self, x: torch.Tensor, residual: torch.Tensor, dead_steps_threshold: int) -> torch.Tensor:
        """Ghost-gradients auxiliary loss (Jermyn & Templeton, 2024) to revive dead features.

        Dead features (inactive for `dead_steps_threshold` steps) are given an auxiliary
        reconstruction target of the current residual `x - recon.detach()`, using an
        exponential of their pre-activation as a soft, always-differentiable gate instead of
        the hard JumpReLU gate. This nudges dead encoder/decoder directions back toward
        under-reconstructed activations without interfering with the main loss.
        """
        dead = self.steps_since_fired > dead_steps_threshold
        if not torch.any(dead):
            return x.new_zeros(())
        pre_activation = self.encode_pre_activation(x)
        ghost_acts = torch.exp(pre_activation[:, dead])
        ghost_recon = ghost_acts @ self.W_dec[dead]
        return (ghost_recon - residual).pow(2).sum(dim=-1).mean()
