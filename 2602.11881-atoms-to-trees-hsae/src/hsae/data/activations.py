"""Activation datasets.

Two paths, per pipeline/03_implementation_plan.yaml Section 5.1:

1. `SyntheticHierarchicalActivations`: an in-memory generator with a *known*
   ground-truth tree structure, used to validate the HSAE pipeline quickly on
   CPU (this environment has no budget for the paper's full-scale ~100M
   activations / ~6h-on-A800 real run).
2. `collect_activations`: an interface for collecting real residual-stream
   activations from a HuggingFace/TransformerLens-loadable LLM (e.g.
   gemma2-2b, qwen3-4b) over a text corpus (e.g. mini-PILE), matching
   Appendix A.1's "Data Preparation": sequence length 1024, BOS excluded,
   RMS-normalized to a constant scaler with expectation sqrt(d).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def rms_normalize(activations: torch.Tensor) -> torch.Tensor:
    """Rescales activations by a single constant scaler so E[||x||_2] = sqrt(d)
    (Appendix A.1: "normalized with a constant scaler to have a RMS norm
    expectation of sqrt(d)")."""
    d = activations.shape[-1]
    norm = activations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = math.sqrt(d) / norm.mean()
    return activations * scale


@dataclass
class SyntheticTreeSpec:
    """Ground-truth tree used to generate synthetic activations for validation.

    A chain of levels with sizes matching `dict_sizes`; each finer-level cluster
    center is a small perturbation of its assigned coarser-level parent center,
    so the "correct" hierarchy is recoverable and known for testing.
    """

    d_model: int
    dict_sizes: list[int]
    cluster_spread: float = 0.15
    noise_std: float = 0.05

    def build(self, generator: torch.Generator) -> list[torch.Tensor]:
        """Returns a list of [n_l, d_model] cluster-center tensors, one per level."""
        centers = [torch.randn(self.dict_sizes[0], self.d_model, generator=generator)]
        for level in range(1, len(self.dict_sizes)):
            n_children = self.dict_sizes[level]
            n_parents = self.dict_sizes[level - 1]
            parent_of = torch.randint(0, n_parents, (n_children,), generator=generator)
            noise = torch.randn(n_children, self.d_model, generator=generator) * self.cluster_spread
            centers.append(centers[level - 1][parent_of] + noise)
        return centers


class SyntheticHierarchicalActivations(Dataset):
    """A CPU-friendly synthetic activation dataset with a known ground-truth hierarchy.

    Each sample activates one randomly chosen finest-level cluster center (plus noise),
    mimicking a single monosemantic concept firing per token, which is the regime HSAE
    is designed to discover structure within.
    """

    def __init__(self, spec: SyntheticTreeSpec, num_samples: int, seed: int = 0):
        self.spec = spec
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.centers = spec.build(generator)
        finest = self.centers[-1]
        chosen = torch.randint(0, finest.shape[0], (num_samples,), generator=generator)
        noise = torch.randn(num_samples, spec.d_model, generator=generator) * spec.noise_std
        self.activations = rms_normalize(finest[chosen] + noise)
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
    site: str,
    d_model: int,
    sequence_length: int = 1024,
    exclude_bos: bool = True,
) -> torch.Tensor:
    """Collects real residual-stream activations from an HF/TransformerLens model.

    This is the "real path" referenced in pipeline/03_implementation_plan.yaml; it
    requires the caller to supply an already-loaded model/tokenizer (e.g. gemma2-2b
    via TransformerLens, as used in the paper) and is not exercised by the tests in
    this repository, which use `SyntheticHierarchicalActivations` instead because
    downloading gemma2-2b weights and running large-batch inference is outside this
    development environment's network/compute budget.

    Returns activations of shape [N, d_model], RMS-normalized per Appendix A.1.
    """
    if not hasattr(model, "run_with_cache"):
        raise NotImplementedError(
            "collect_activations expects a TransformerLens-style model exposing "
            "`run_with_cache`. Wrap your HuggingFace model with TransformerLens "
            "(transformer_lens.HookedTransformer.from_pretrained) to use this path."
        )

    all_activations = []
    hook_name = f"blocks.{layer}.hook_resid_post" if site == "resid_post" else site
    for text in texts:
        tokens = tokenizer(text, truncation=True, max_length=sequence_length, return_tensors="pt")["input_ids"]
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        activations = cache[hook_name][0]  # [seq_len, d_model]
        if exclude_bos:
            activations = activations[1:]
        all_activations.append(activations)

    activations = torch.cat(all_activations, dim=0)
    assert activations.shape[-1] == d_model
    return rms_normalize(activations)
