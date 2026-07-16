"""Tree SAE: a flat multi-block TopK SAE with a recursive activation-coverage gate.

Implements Eq. 1-2 (base SAE), Eq. 6 (recursive activation-coverage gate) from
pipeline/01_algorithm_extraction.yaml, i.e. Section 2.1 and Section 4 of the
paper.

The dictionary is a single flat encoder/decoder of size d_f = sum(layer_sizes),
partitioned into contiguous per-privilege-layer blocks. A separate
(non-gradient) allocation tensor `parent_idx` gives each feature's parent
index among ALL features at strictly earlier layers (layer-1 features are
always parented by the imaginary root and are never gated).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tree_sae.config import ModelConfig

ROOT = -1


class TreeSAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layer_sizes = list(config.layer_sizes)
        self.d_model = config.d_model
        self.l0 = config.l0

        offsets = [0]
        for size in self.layer_sizes:
            offsets.append(offsets[-1] + size)
        self.offsets = offsets  # length num_levels + 1
        dict_size = offsets[-1]

        self.W_enc = nn.Parameter(torch.empty(self.d_model, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._renormalize_decoder()

        # parent_idx[i] = global feature index of feature i's parent (ROOT for layer-1 features).
        self.register_buffer("parent_idx", torch.full((dict_size,), ROOT, dtype=torch.long))
        self.register_buffer("steps_since_fired", torch.zeros(dict_size, dtype=torch.long))

    @property
    def num_levels(self) -> int:
        return len(self.layer_sizes)

    @property
    def dict_size(self) -> int:
        return self.offsets[-1]

    def layer_slice(self, level: int) -> slice:
        return slice(self.offsets[level], self.offsets[level + 1])

    def encode_pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk_raw(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """Raw (un-gated) TopK activation over the FULL flat dictionary (Eq. 1)."""
        relu = F.relu(pre_activation)
        k = min(self.l0, relu.shape[-1])
        values, indices = torch.topk(relu, k=k, dim=-1)
        raw = torch.zeros_like(relu)
        raw.scatter_(1, indices, values)
        return raw

    def gate(self, raw: torch.Tensor) -> torch.Tensor:
        """Recursive activation-coverage gate (Eq. 6): zeroes out any feature whose
        assigned parent's already-gated activation is inactive. Layer 1 is always
        parented by the root and is therefore never gated."""
        f_star = torch.zeros_like(raw)
        f_star[:, self.layer_slice(0)] = raw[:, self.layer_slice(0)]

        for level in range(1, self.num_levels):
            block = self.layer_slice(level)
            parents = self.parent_idx[block]  # [layer_size], indices < offsets[level], or ROOT (-1)
            is_root = parents.eq(ROOT)
            safe_parents = parents.clamp(min=0)  # dummy index for ROOT entries; overridden below
            prefix = f_star[:, : self.offsets[level]]
            gathered_active = torch.gather(prefix, 1, safe_parents.unsqueeze(0).expand(raw.shape[0], -1)) > 0
            # ROOT-parented features (the default before allocation, or after
            # reassign_dead_to_root) are always active/ungated, matching Appendix G.
            parent_active = is_root.unsqueeze(0) | gathered_active
            f_star[:, block] = raw[:, block] * parent_active.to(raw.dtype)

        return f_star

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw, gated) feature activations, both [B, dict_size]."""
        pre_activation = self.encode_pre_activation(x)
        raw = self.topk_raw(pre_activation)
        gated = self.gate(raw)
        return raw, gated

    def decode_block(self, level: int, feature_acts: torch.Tensor) -> torch.Tensor:
        block = self.layer_slice(level)
        return feature_acts[:, block] @ self.W_dec[block]

    def cumulative_reconstructions(self, feature_acts: torch.Tensor) -> list[torch.Tensor]:
        """sum_{t=1}^{l} x_hat_t for every l (Eq. 7's inner sum), bias added once per level."""
        cumulative = []
        running = torch.zeros(feature_acts.shape[0], self.d_model, device=feature_acts.device)
        for level in range(self.num_levels):
            running = running + self.decode_block(level, feature_acts)
            cumulative.append(running + self.b_dec)
        return cumulative

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        raw, gated = self.encode(x)
        cumulative = self.cumulative_reconstructions(gated)
        return raw, gated, cumulative

    @torch.no_grad()
    def _renormalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def post_step(self) -> None:
        self._renormalize_decoder()

    @torch.no_grad()
    def update_dead_feature_tracker(self, raw: torch.Tensor) -> None:
        fired = raw.abs().sum(dim=0) > 0
        self.steps_since_fired[fired] = 0
        self.steps_since_fired[~fired] += 1

    def is_dead(self, dead_steps_threshold: int) -> torch.Tensor:
        return self.steps_since_fired > dead_steps_threshold
