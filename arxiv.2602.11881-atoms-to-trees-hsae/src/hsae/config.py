"""Configuration dataclasses for HSAE.

Paper: From Atoms to Trees: Building a Structured Feature Forest with
       Hierarchical Sparse Autoencoders (arXiv:2602.11881v1)

Default values are taken directly from the paper (Section 3, 4, Appendix A)
where stated. Values the paper does not give a concrete number for are
marked INFERRED below and are exposed as tunable defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 2304  # gemma2-2b residual stream width (Appendix A.1)
    dict_sizes: list[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384])  # Section 4

    @property
    def num_levels(self) -> int:
        return len(self.dict_sizes)


@dataclass
class SparsityConfig:
    target_l0: float = 50.0  # Section 4
    initial_lambda: float = 5e-2  # INFERRED: paper does not state an initial value
    ema_momentum: float = 0.999  # Appendix A.2
    decay_rate_eta: float = 0.001  # Appendix A.2
    control_kappa: float = 1.0  # INFERRED: proportional-control gain, paper gives no exact formula
    lambda_step_clip: tuple[float, float] = (0.98, 1.02)  # INFERRED: per-step multiplicative clamp


@dataclass
class HierarchyConfig:
    parent_child_constraint_weight_rho: float = 0.01  # Section 3.1/3.2
    perturbation_rate_r: float = 0.05  # Section 3.2
    hierarchy_update_period_steps: int = 5000  # Section 3.3
    exclusion_quantile: float = 0.20  # Section 3.3 ("Partial Tree Structure")
    similarity_metric: str = "encoder"  # Section 3.3 default; alternatives: "decoder", "coactivation"
    coactivation_ema_momentum: float = 0.99  # INFERRED: not numerically specified for this alt metric


@dataclass
class OptimizerConfig:
    learning_rate: float = 4e-4  # INFERRED: paper gives only Adam betas + schedule shape, not peak LR
    adam_beta1: float = 0.0  # Appendix A.1
    adam_beta2: float = 0.995  # Appendix A.1
    warmup_fraction: float = 0.10  # Appendix A.1
    batch_size: int = 1024  # Appendix A.1
    total_steps: int = 20_000  # INFERRED: paper reports wall-clock (~6h on A800) not a step count
    ghost_grad_dead_steps: int = 200  # INFERRED: window of inactivity before a feature is "dead"


@dataclass
class DataConfig:
    dataset_size: int = 100_000_000  # Section 4 / Appendix A.1 (full-scale target)
    sequence_length: int = 1024  # Appendix A.1
    exclude_bos: bool = True  # Appendix A.1


@dataclass
class HSAEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 0

    @classmethod
    def small_synthetic(cls) -> "HSAEConfig":
        """A CPU-friendly configuration used for smoke tests and synthetic-data
        validation, matching the "reproduction_roadmap" in
        pipeline/02_concept_analysis.yaml and the "Validation Approach" in
        pipeline/03_implementation_plan.yaml. Not a paper-reported setting."""
        return cls(
            model=ModelConfig(d_model=16, dict_sizes=[8, 16, 32]),
            sparsity=SparsityConfig(target_l0=3.0, initial_lambda=1e-2),
            hierarchy=HierarchyConfig(hierarchy_update_period_steps=25),
            optim=OptimizerConfig(
                learning_rate=1e-3,
                batch_size=64,
                total_steps=300,
                ghost_grad_dead_steps=50,
            ),
            data=DataConfig(dataset_size=2000, sequence_length=1, exclude_bos=False),
        )
