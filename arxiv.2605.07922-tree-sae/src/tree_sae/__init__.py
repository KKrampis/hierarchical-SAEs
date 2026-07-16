"""Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders.

Reference implementation of arXiv:2605.07922v2, built via the paper2code
Claude Code skill (.claude/skills/paper2code).
"""

from tree_sae.config import TreeSAEConfig
from tree_sae.models.tree_sae import TreeSAE

__all__ = ["TreeSAEConfig", "TreeSAE"]
