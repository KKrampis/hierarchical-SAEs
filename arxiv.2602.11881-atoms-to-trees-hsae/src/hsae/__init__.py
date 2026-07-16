"""HSAE: Hierarchical Sparse Autoencoders.

Reference implementation of "From Atoms to Trees: Building a Structured
Feature Forest with Hierarchical Sparse Autoencoders" (arXiv:2602.11881v1),
built via the paper2code Claude Code skill (.claude/skills/paper2code).
"""

from hsae.config import HSAEConfig
from hsae.models.hsae import HSAE

__all__ = ["HSAEConfig", "HSAE"]
