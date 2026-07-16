"""Activation datasets.

Two paths, per pipeline/03_implementation_plan.yaml Section 5.1 (mirroring
the design of the companion HSAE repo,
arxiv.2602.11881-atoms-to-trees-hsae/src/hsae/data/activations.py):

1. `SyntheticHierarchicalActivations`: an in-memory generator with a known
   ground-truth tree structure, used to validate the Tree SAE pipeline on
   CPU (this environment has no budget for the paper's full-scale GPT2-small
   + 500M-token-of-The-Pile real run).
2. `collect_activations`: an interface for collecting real GPT2-small
   layer-5 residual-stream activations over The Pile, matching Appendix C's
   "SAE Training" setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticTreeSpec:
    """Ground-truth multi-level tree used to generate synthetic activations.
    Each finer-level cluster center is a small perturbation of a parent center
    chosen from the immediately preceding level (a valid special case of Tree
    SAE's more general "parent at any earlier layer" structure)."""

    d_model: int
    layer_sizes: list[int]
    cluster_spread: float = 0.15
    noise_std: float = 0.05

    def build(self, generator: torch.Generator) -> list[torch.Tensor]:
        centers = [torch.randn(self.layer_sizes[0], self.d_model, generator=generator)]
        for level in range(1, len(self.layer_sizes)):
            n_children = self.layer_sizes[level]
            n_parents = self.layer_sizes[level - 1]
            parent_of = torch.randint(0, n_parents, (n_children,), generator=generator)
            noise = torch.randn(n_children, self.d_model, generator=generator) * self.cluster_spread
            centers.append(centers[level - 1][parent_of] + noise)
        return centers


class SyntheticHierarchicalActivations(Dataset):
    """CPU-friendly synthetic activations with a known ground-truth hierarchy.
    Each sample activates one randomly chosen finest-level cluster center plus noise."""

    def __init__(self, spec: SyntheticTreeSpec, num_samples: int, seed: int = 0):
        self.spec = spec
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.centers = spec.build(generator)
        finest = self.centers[-1]
        chosen = torch.randint(0, finest.shape[0], (num_samples,), generator=generator)
        noise = torch.randn(num_samples, spec.d_model, generator=generator) * spec.noise_std
        self.activations = finest[chosen] + noise
        self.labels = chosen

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
    """Collects real residual-stream activations from GPT2-small (or a similar
    HF/TransformerLens model), matching Appendix C's "SAE Training" setup
    (layer 5 of GPT2-small, trained on The Pile). Not exercised by this
    repository's tests, which use `SyntheticHierarchicalActivations` instead
    because downloading GPT2-small + The Pile and running large-batch
    inference is outside this development environment's network/compute
    budget.

    Returns activations of shape [N, d_model].
    """
    if not hasattr(model, "run_with_cache"):
        raise NotImplementedError(
            "collect_activations expects a TransformerLens-style model exposing "
            "`run_with_cache`. Wrap your HuggingFace model with TransformerLens "
            "(transformer_lens.HookedTransformer.from_pretrained('gpt2')) to use this path."
        )

    all_activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        all_activations.append(cache[hook_name][0])

    activations = torch.cat(all_activations, dim=0)
    assert activations.shape[-1] == d_model
    return activations
