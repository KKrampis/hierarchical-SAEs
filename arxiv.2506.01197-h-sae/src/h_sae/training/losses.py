"""H-SAE loss functions (Eq. 3.2-3.4).

    L_recon = || x - H-SAE(x) ||_2^2 + beta * || x - x_hat_high ||_2^2
    L_ortho = || E_paper @ D_paper - diag(E_paper @ D_paper) ||_F / (m_top^2 - m_top)
    L_sparse = ||z||_1 + sum_{j in K} ||z_j||_1
    L = L_recon + lambda_1 * L_ortho + lambda_2 * L_sparse

Parameter-convention note: this implementation stores the top-level encoder
as `E` with shape [d, m_top] (so `x @ E` gives the pre-activation) and the
decoder as `D` with shape [m_top, d] (so `z @ D` gives the reconstruction).
The paper's convention is the transpose of this (E_paper: [m_top, d],
D_paper: [d, m_top], with pre-activation E_paper(x) and reconstruction
D_paper @ z). Since ||M - diag(M)||_F is invariant under transposing M (a
matrix and its transpose share the same diagonal, and the Frobenius norm is
transpose-invariant), `D @ E` (shape [m_top, m_top]) here is mathematically
equivalent to the paper's `E_paper @ D_paper` for the purposes of Eq. 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from h_sae.models.hsae import HSAE


@dataclass
class HSAELossOutput:
    total: torch.Tensor
    recon_loss: torch.Tensor
    top_recon_loss: torch.Tensor
    ortho_loss: torch.Tensor
    sparse_loss: torch.Tensor


def warmup_coefficient(step: int, warmup_steps: int, target: float) -> float:
    """Linear warmup from 0 to `target` over `warmup_steps` (Appendix A.2:
    "the regularizers also warmup from 0 over the first 1,000 steps")."""
    if warmup_steps <= 0:
        return target
    return target * min(1.0, step / warmup_steps)


def orthogonality_loss(E: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    m_top = E.shape[1]
    gram = D @ E  # [m_top, m_top]; equivalent to E_paper @ D_paper up to transpose (see module docstring)
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    return torch.linalg.norm(off_diagonal, ord="fro") / (m_top**2 - m_top)


def compute_hsae_loss(
    model: HSAE,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    x_hat_high: torch.Tensor,
    z_low: torch.Tensor,
    beta: float,
    lambda_ortho: float,
    lambda_sparse: float,
) -> HSAELossOutput:
    recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
    top_recon_loss = (x - x_hat_high).pow(2).sum(dim=-1).mean()
    total_recon = recon_loss + beta * top_recon_loss

    ortho_loss = orthogonality_loss(model.E, model.D)
    sparse_loss = (z.abs().sum(dim=-1) + z_low.abs().sum(dim=(-2, -1))).mean()

    total = total_recon + lambda_ortho * ortho_loss + lambda_sparse * sparse_loss
    return HSAELossOutput(
        total=total,
        recon_loss=recon_loss,
        top_recon_loss=top_recon_loss,
        ortho_loss=ortho_loss,
        sparse_loss=sparse_loss,
    )
