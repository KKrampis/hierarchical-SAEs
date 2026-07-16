"""Multi-level HSAE wrapper.

Owns L JumpReLUSAE instances (Section 3) of increasing dictionary size,
plus the hierarchical assignment state {C_(l,i)} between every pair of
adjacent levels, represented as one parent_idx tensor per level boundary
(child -> parent index, UNASSIGNED if the child has no parent).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hsae.algorithms.hierarchy import UNASSIGNED, children_sets, update_level_hierarchy
from hsae.algorithms.perturbation import children_sum_outputs, perturbed_reconstruction
from hsae.config import ModelConfig
from hsae.models.jumprelu_sae import JumpReLUSAE


class HSAE(nn.Module):
    def __init__(self, config: ModelConfig, bandwidth: float = 1e-3):
        super().__init__()
        self.config = config
        self.levels = nn.ModuleList(
            [JumpReLUSAE(config.d_model, dict_size, bandwidth=bandwidth) for dict_size in config.dict_sizes]
        )
        # parent_idx[l] maps level (l+1) children -> level l parent index; length L-1.
        self.parent_idx: list[torch.Tensor] = [
            torch.full((self.levels[l + 1].dict_size,), UNASSIGNED, dtype=torch.long)
            for l in range(self.num_levels - 1)
        ]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def children_of(self, level: int) -> list[list[int]]:
        """C_(l,i) for every parent feature i at `level` (0-indexed)."""
        return children_sets(self.parent_idx[level], self.levels[level].dict_size)

    @torch.no_grad()
    def update_hierarchy(
        self,
        metric: str = "encoder",
        exclusion_quantile: float = 0.20,
        coactivation_tracker=None,
        max_children_per_parent: int | None = None,
    ) -> None:
        """Hierarchy Update stage (Section 3.3): recompute {C_(l,i)} for every level
        boundary from the current model parameters, with structure held fixed
        (this method performs no gradient step)."""
        for level in range(self.num_levels - 1):
            self.parent_idx[level] = update_level_hierarchy(
                child_sae=self.levels[level + 1],
                parent_sae=self.levels[level],
                metric=metric,
                exclusion_quantile=exclusion_quantile,
                coactivation_tracker=coactivation_tracker,
                level=level,
                max_children_per_parent=max_children_per_parent,
            )

    def forward(self, x: torch.Tensor, perturbation_rate: float = 0.0, generator: torch.Generator | None = None):
        """Runs every level's encoder and returns, per level, the feature activations
        and the reconstruction used for the loss:
          - the last level (finest) always uses its own (un-perturbed) reconstruction,
            since it has no children.
          - every other level uses the perturbed reconstruction SAE_hat_l(x) from
            Section 3.2, mixing its own output with its children's summed output.

        Returns:
            feature_acts: list[Tensor[B, n_l]] per level
            reconstructions: list[Tensor[B, d_model]] per level (perturbed where applicable)
            feature_outputs: list[Tensor[B, n_l, d_model]] per level (needed for L_PC)
        """
        feature_acts = [sae.encode(x) for sae in self.levels]
        feature_outputs = [sae.feature_outputs(acts) for sae, acts in zip(self.levels, feature_acts)]

        reconstructions: list[torch.Tensor] = []
        for level in range(self.num_levels):
            if level == self.num_levels - 1:
                reconstructions.append(self.levels[level].decode(feature_acts[level]))
                continue
            if perturbation_rate > 0.0:
                recon = perturbed_reconstruction(
                    parent_feature_outputs=feature_outputs[level],
                    child_feature_outputs=feature_outputs[level + 1],
                    parent_idx=self.parent_idx[level],
                    perturbation_rate=perturbation_rate,
                    generator=generator,
                )
            else:
                recon = self.levels[level].decode(feature_acts[level])
            reconstructions.append(recon)

        return feature_acts, reconstructions, feature_outputs

    def children_sum_feature_outputs(self, level: int, feature_outputs: list[torch.Tensor]) -> torch.Tensor:
        """sum_{j in C_(l,i)} f_{l+1,j}(x) for every parent i at `level`, shape [B, n_l, d_model]."""
        return children_sum_outputs(
            child_feature_outputs=feature_outputs[level + 1],
            parent_idx=self.parent_idx[level],
            n_parents=self.levels[level].dict_size,
        )

    @torch.no_grad()
    def post_step(self) -> None:
        for sae in self.levels:
            sae.post_step()
