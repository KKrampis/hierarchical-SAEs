"""Baseline TopK SAE (Eq. 2.1-2.2, Gao et al. 2024), used as the comparison
baseline throughout the paper.

    SAE_k(x) = D TopK_k( LeakyReLU_alpha( E(x - b) ) )
    L_recon  = || x - SAE_k(x) ||_2^2

Implementation note: the paper describes LeakyReLU_alpha as having threshold
alpha = 1/sqrt(d) with "some small negative slope below the threshold" --
this thresholded-leaky variant is inherited from Gao et al. (2024) and not
fully re-specified in this paper. We approximate it here with a standard
LeakyReLU applied at zero (`negative_slope` configurable), which preserves
the essential sparsity-promoting shape without over-committing to an
unstated exact threshold mechanism; see
pipeline/01_algorithm_extraction.yaml missing_but_critical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from h_sae.config import ModelConfig


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, dict_size: int, k: int, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k
        self.leaky_relu_slope = leaky_relu_slope

        self.E = nn.Parameter(torch.empty(d_model, dict_size))
        self.D = nn.Parameter(torch.empty(dict_size, d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.E)
        with torch.no_grad():
            self.D.copy_(self.E.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_activation = F.leaky_relu((x - self.b) @ self.E, negative_slope=self.leaky_relu_slope)
        k = min(self.k, self.dict_size)
        values, indices = torch.topk(pre_activation, k=k, dim=-1)
        z = torch.zeros_like(pre_activation)
        z.scatter_(1, indices, values)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = z @ self.D
        return x_hat, z

    @classmethod
    def from_config(cls, config: ModelConfig) -> "TopKSAE":
        return cls(config.d_model, config.m_top, config.k, config.leaky_relu_slope)
