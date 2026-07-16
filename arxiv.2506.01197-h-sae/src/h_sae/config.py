"""Configuration dataclasses for H-SAE.

Paper: Incorporating Hierarchical Semantics in Sparse Autoencoder
       Architectures (arXiv:2506.01197v1)

Default values are taken directly from the paper (Section 4, Appendix A)
where stated. Values the paper does not give a concrete number for are
marked INFERRED below and are exposed as tunable defaults (see
pipeline/01_algorithm_extraction.yaml "missing_but_critical").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 2304  # Gemma 2-2B hidden dim (not stated in this paper; matches gemma2-2b, [INFERRED] consistent with the companion HSAE repo's value for the same model family)
    m_top: int = 16000  # top-level dictionary size (Section 4: 8k/16k/32k tested)
    k: int = 32  # top-level TopK sparsity (Appendix A.2)
    m_low: int = 16  # sublatents per expert (Section 4: 16 or 64 tested)
    subspace_dim: int = 4  # s: 4 when m_low=16, 8 otherwise (Appendix A.2)
    leaky_relu_slope: float = 0.01  # INFERRED: not numerically specified in the paper
    decoder_renormalize: bool = False  # INFERRED: not mentioned for H-SAE; default off (see missing_but_critical)


@dataclass
class LossConfig:
    beta: float = 0.1  # top-level-only reconstruction weight (Appendix A.2)
    lambda_ortho: float = 0.1  # orthogonality penalty coefficient (Appendix A.2)
    lambda_sparse: float = 0.001  # L1 sparsity coefficient (Appendix A.2)
    warmup_steps: int = 1000  # loss-coefficient linear warmup (Appendix A.2)


@dataclass
class OptimizerConfig:
    peak_learning_rate: float = 5e-4  # Appendix A.2
    initial_learning_rate: float = 1e-11  # Appendix A.2
    warmup_steps: int = 1000  # Appendix A.2
    adam_b1: float = 0.9  # Appendix A.2
    adam_b2: float = 0.999  # INFERRED: not specified in paper
    grad_clip_norm: float = 0.75  # Appendix A.2
    batch_size: int = 32512  # Appendix A.2
    total_steps: int = 20_000  # INFERRED: paper reports 1B vectors / 4 epochs / batch size (~123k steps), not directly a step count for small-scale use


@dataclass
class DataConfig:
    training_vectors: int = 1_000_000_000  # Appendix A.1
    epochs: int = 4  # Section 4


@dataclass
class HSAEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
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
            model=ModelConfig(d_model=16, m_top=12, k=3, m_low=4, subspace_dim=3),
            loss=LossConfig(beta=0.1, lambda_ortho=0.1, lambda_sparse=0.001, warmup_steps=20),
            optim=OptimizerConfig(
                peak_learning_rate=1e-3,
                initial_learning_rate=1e-6,
                warmup_steps=20,
                grad_clip_norm=0.75,
                batch_size=64,
                total_steps=300,
            ),
            data=DataConfig(training_vectors=64 * 300, epochs=1),
        )
