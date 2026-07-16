"""H-SAE: mixture-of-experts Hierarchical Sparse Autoencoder (Eq. 3.1, Algorithm 1).

Top-level TopK_k routing selects k active experts (NO bias subtraction, per
Algorithm 1's ForwardPass); only those k experts' projection and low-level
SAE parameters are gathered and used per batch element, matching the
paper's conditional-compute efficiency claim (Section 3.1, "Computational
Efficiency").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from h_sae.config import ModelConfig


class HSAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d, m_top, k, m_low, s = (
            config.d_model,
            config.m_top,
            config.k,
            config.m_low,
            config.subspace_dim,
        )
        self.d_model = d
        self.m_top = m_top
        self.k = min(k, m_top)
        self.m_low = m_low
        self.subspace_dim = s
        self.leaky_relu_slope = config.leaky_relu_slope

        # Top-level SAE (E, D) -- no bias, per Algorithm 1.
        self.E = nn.Parameter(torch.empty(d, m_top))
        self.D = nn.Parameter(torch.empty(m_top, d))
        nn.init.kaiming_uniform_(self.E)
        with torch.no_grad():
            self.D.copy_(self.E.T)

        # Per-expert parameters, batched over the expert dimension (m_top).
        self.Pi_down = nn.Parameter(torch.empty(m_top, d, s))
        self.Pi_up = nn.Parameter(torch.empty(m_top, s, d))
        self.E_low = nn.Parameter(torch.empty(m_top, s, m_low))
        self.D_low = nn.Parameter(torch.empty(m_top, m_low, s))
        for param in (self.Pi_down, self.Pi_up, self.E_low, self.D_low):
            nn.init.kaiming_uniform_(param.view(m_top, -1))

    def forward(self, x: torch.Tensor):
        """Returns (x_hat, z, x_hat_high, z_low, topk_indices).

        x_hat: [B, d]                final reconstruction
        z: [B, m_top]                 top-level sparse code
        x_hat_high: [B, d]            top-level-only reconstruction (= z @ D)
        z_low: [B, k, m_low]          low-level codes for the k active experts (exactly
                                       one nonzero entry along the last dim, per expert)
        topk_indices: [B, k]          indices of the k active experts per batch element
        """
        pre_top = F.leaky_relu(x @ self.E, negative_slope=self.leaky_relu_slope)  # no bias (Algorithm 1)
        top_values, topk_indices = torch.topk(pre_top, k=self.k, dim=-1)
        z = torch.zeros_like(pre_top)
        z.scatter_(1, topk_indices, top_values)

        x_hat_high = z @ self.D

        pi_down = self.Pi_down[topk_indices]  # [B, k, d, s]
        pi_up = self.Pi_up[topk_indices]  # [B, k, s, d]
        e_low = self.E_low[topk_indices]  # [B, k, s, m_low]
        d_low = self.D_low[topk_indices]  # [B, k, m_low, s]

        x_sub = torch.einsum("bd,bkds->bks", x, pi_down)  # [B, k, s]
        pre_low = F.leaky_relu(torch.einsum("bks,bksm->bkm", x_sub, e_low), negative_slope=self.leaky_relu_slope)

        low_values, low_indices = torch.topk(pre_low, k=1, dim=-1)  # TopK_1 per expert
        z_low = torch.zeros_like(pre_low)
        z_low.scatter_(-1, low_indices, low_values)  # [B, k, m_low]

        x_hat_sub = torch.einsum("bkm,bkms->bks", z_low, d_low)  # [B, k, s]
        x_hat_low_per_expert = torch.einsum("bks,bksd->bkd", x_hat_sub, pi_up)  # [B, k, d]
        x_hat_low = x_hat_low_per_expert.sum(dim=1)  # [B, d]

        x_hat = x_hat_high + x_hat_low
        return x_hat, z, x_hat_high, z_low, topk_indices

    @torch.no_grad()
    def post_step(self) -> None:
        if not self.config.decoder_renormalize:
            return
        self.D.div_(self.D.norm(dim=1, keepdim=True).clamp_min(1e-8))
