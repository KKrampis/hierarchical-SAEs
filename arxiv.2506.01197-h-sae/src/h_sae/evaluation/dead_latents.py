"""Dead top-level latent rate tracking (Appendix A.3).

The paper tracks an EMA over 300 batches (~1M tokens) of each latent's
activation frequency; here we provide a simple batch-level dead-rate
computation suitable for the small synthetic evaluation runs in this repo.
"""

from __future__ import annotations

import torch


def dead_latent_rate(z_batch: torch.Tensor) -> float:
    """Fraction of top-level latents (columns of z_batch) that never fire
    across the given batch of codes."""
    ever_active = (z_batch != 0).any(dim=0)
    return 1.0 - ever_active.float().mean().item()
