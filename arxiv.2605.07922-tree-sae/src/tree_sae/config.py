"""Configuration dataclasses for Tree SAE.

Paper: Tree SAE: Learning Hierarchical Feature Structures in Sparse
       Autoencoders (arXiv:2605.07922v2)

Default values are taken directly from the paper (Section 4, Appendix C,
Appendix G) where stated. Values the paper does not give a concrete number
for are marked INFERRED below and are exposed as tunable defaults (see
pipeline/01_algorithm_extraction.yaml "missing_but_critical").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 768  # GPT2-small hidden dim (Appendix C)
    layer_sizes: list[int] = field(default_factory=lambda: [1536, 3072, 9216, 10752])  # 4-layer, Appendix C Table 1
    l0: int = 32  # TopK k, IS the L0 sparsity level per Appendix C

    @property
    def num_levels(self) -> int:
        return len(self.layer_sizes)

    @property
    def dict_size(self) -> int:
        return sum(self.layer_sizes)


@dataclass
class AuxLossConfig:
    aux_topk: int = 256  # Appendix C
    # alpha_1 = 1/32, alpha_{2..L} = 0 in the main experiment (Appendix C)
    layer_alphas: list[float] = field(default_factory=lambda: [1.0 / 32, 0.0, 0.0, 0.0])
    dead_feature_window_tokens: int = 10_000_000  # Appendix G


@dataclass
class AllocationConfig:
    reallocation_initial_interval_steps: int = 3000  # Appendix G
    reallocation_interval_increment_steps: int = 2000  # INFERRED unit for "increase by two"
    reallocation_interval_cap_steps: int = 10_000  # Appendix G
    dead_to_root_step_fraction: float = 0.5  # "around 50,000 steps" of ~100k total (Appendix G)
    parent_eligibility_min_rate: float = 1.0 / 50_000  # activations per token (Appendix G)
    capacity_window_steps: int | None = None  # INFERRED: defaults to the current reallocation interval


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4  # Appendix C
    batch_size: int = 5120  # Appendix C
    grad_clip_norm: float = 1.0  # INFERRED: paper states normalization without a numeric value
    total_steps: int = 20_000  # INFERRED: paper reports token budget (500M) not a step count


@dataclass
class DataConfig:
    training_tokens: int = 500_000_000  # Appendix C


@dataclass
class TreeSAEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 0

    @classmethod
    def small_synthetic(cls) -> "TreeSAEConfig":
        """A CPU-friendly configuration used for smoke tests and synthetic-data
        validation. Not a paper-reported setting."""
        return cls(
            model=ModelConfig(d_model=16, layer_sizes=[4, 8, 16], l0=3),
            aux=AuxLossConfig(aux_topk=4, layer_alphas=[1.0 / 32, 0.0, 0.0], dead_feature_window_tokens=200),
            allocation=AllocationConfig(
                reallocation_initial_interval_steps=25,
                reallocation_interval_increment_steps=25,
                reallocation_interval_cap_steps=100,
                dead_to_root_step_fraction=0.5,
                parent_eligibility_min_rate=0.0,
                capacity_window_steps=25,
            ),
            optim=OptimizerConfig(learning_rate=1e-3, batch_size=64, total_steps=300, grad_clip_norm=1.0),
            data=DataConfig(training_tokens=64 * 300),
        )
