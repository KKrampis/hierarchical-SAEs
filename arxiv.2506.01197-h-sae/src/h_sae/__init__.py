"""H-SAE: Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures.

Reference implementation of arXiv:2506.01197v1, built via the paper2code
Claude Code skill (.claude/skills/paper2code).
"""

from h_sae.config import HSAEConfig
from h_sae.models.hsae import HSAE

__all__ = ["HSAEConfig", "HSAE"]
