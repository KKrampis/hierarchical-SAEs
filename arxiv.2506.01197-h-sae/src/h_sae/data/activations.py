"""Activation datasets.

`SyntheticParentChildActivations` generates data directly from Park et al.
(2024)'s categorical-concept geometry that H-SAE is designed to exploit:
each sample is (a parent-concept direction) + (a child-in-context vector
drawn from that parent's low-rank subspace) + noise, i.e. exactly the
"child = parent + child-in-context" structure described in Section 3.1.

`collect_activations` is the real-data integration point (Gemma 2-2B layer
20 + multilingual Wikipedia), not exercised by this repository's tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticTreeSpec:
    d_model: int
    m_top: int
    m_low: int
    subspace_dim: int
    child_scale: float = 0.6
    noise_std: float = 0.05

    def build(self, generator: torch.Generator):
        parent_directions = torch.randn(self.m_top, self.d_model, generator=generator)
        parent_directions = parent_directions / parent_directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        subspace_bases = torch.randn(self.m_top, self.d_model, self.subspace_dim, generator=generator)
        child_coords = torch.randn(self.m_top, self.m_low, self.subspace_dim, generator=generator)
        # child_vectors[p, c] = subspace_bases[p] @ child_coords[p, c]
        child_vectors = torch.einsum("pds,pcs->pcd", subspace_bases, child_coords)
        child_vectors = child_vectors * self.child_scale

        return parent_directions, child_vectors


class SyntheticParentChildActivations(Dataset):
    """CPU-friendly synthetic activations with a known ground-truth
    (parent, child) generative structure, one sample = one (parent, child) pair."""

    def __init__(self, spec: SyntheticTreeSpec, num_samples: int, seed: int = 0):
        self.spec = spec
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.parent_directions, self.child_vectors = spec.build(generator)

        parent_idx = torch.randint(0, spec.m_top, (num_samples,), generator=generator)
        child_idx = torch.randint(0, spec.m_low, (num_samples,), generator=generator)
        noise = torch.randn(num_samples, spec.d_model, generator=generator) * spec.noise_std

        self.parent_labels = parent_idx
        self.child_labels = child_idx
        self.activations = self.parent_directions[parent_idx] + self.child_vectors[parent_idx, child_idx] + noise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def collect_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    d_model: int,
) -> torch.Tensor:
    """Collects real residual-stream activations from Gemma 2-2B layer 20
    (Appendix A.1). Requires a HuggingFace `transformers` model exposing
    hidden states; not exercised by this repository's tests, which use
    `SyntheticParentChildActivations` instead because downloading Gemma 2-2B
    and multilingual Wikipedia and running large-batch inference over 1B
    tokens is outside this development environment's network/compute budget.

    Returns unit-normalized activations of shape [N, d_model] (Appendix A.1:
    "we normalize the vectors to unit norm ... we do not subtract any mean
    vector nor include a bias term").
    """
    all_activations = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        outputs = model(tokens, output_hidden_states=True)
        all_activations.append(outputs.hidden_states[layer][0])

    activations = torch.cat(all_activations, dim=0)
    assert activations.shape[-1] == d_model
    norms = activations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return activations / norms
