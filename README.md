# hierarchical-SAEs

Reference implementations of three research papers on **hierarchical sparse
autoencoders (SAEs)** for LLM interpretability, each built end-to-end with
the [`paper2code`](.claude/skills/paper2code/SKILL.md) Claude Code skill: the
paper is read, distilled into a structured YAML pipeline
(algorithm extraction → concept analysis → implementation plan), and then
implemented, tested, and validated from that plan.

All three papers tackle the same underlying question — standard SAEs learn a
flat, unstructured set of features, but real-world concepts are organized
hierarchically (e.g. "dog" → "corgi"/"greyhound") — with three genuinely
different architectural answers.

## Subdirectories

### `arxiv.2602.11881-atoms-to-trees-hsae/`
**"From Atoms to Trees: Building a Structured Feature Forest with
Hierarchical Sparse Autoencoders"** ([arXiv:2602.11881](https://arxiv.org/abs/2602.11881))

Jointly trains a series of JumpReLU SAEs of increasing dictionary size
("levels"), each level's features assigned a parent at the **immediately
preceding** level. Hierarchy is enforced during training via two explicit
mechanisms: a parent-children **structural constraint loss** (pulls a
parent's output toward the sum of its children's outputs) and a **random
feature-perturbation** mechanism (stochastically substitutes a feature's own
activation with its children's summed activation). Parent-child assignment
is periodically recomputed via similarity-based matching with a partial-tree
quantile filter, alternating with gradient-descent parameter updates.

### `arxiv.2605.07922-tree-sae/`
**"Tree SAE: Learning Hierarchical Feature Structures in Sparse
Autoencoders"** ([arXiv:2605.07922](https://arxiv.org/abs/2605.07922))

Also a multi-level flat dictionary, but a child feature may be parented by
**any** feature at any strictly earlier level (not just the adjacent one).
Hierarchy is enforced **architecturally** rather than by a soft loss: a
recursive activation-coverage gate zeroes out a feature whenever its
assigned parent is inactive, combined with a multi-level cumulative
reconstruction loss that (provably, per the paper's Appendix H) drives
semantic alignment as a byproduct. A capacity-aware dynamic
feature-reallocation mechanism (a greedy max-heap allocator plus periodic
dead-feature reassignment) combats the dead-feature problem this rigid tree
structure would otherwise cause.

### `arxiv.2506.01197-h-sae/`
**"Incorporating Hierarchical Semantics in Sparse Autoencoder
Architectures"** ([arXiv:2506.01197](https://arxiv.org/abs/2506.01197))

A structurally different approach: a **mixture-of-experts** SAE, exactly two
levels deep. A sparse top-level SAE selects `k` active "expert" concepts;
each activated expert runs its own small, low-rank-subspace-restricted,
`TopK=1` low-level SAE. Inactive experts are never computed at all (true
conditional compute, not masking), which both instantiates a specific
geometric theory of categorical concepts (Park et al. 2024: a concept is a
parent direction plus a low-rank subspace of child directions) and yields
large efficiency gains over an equivalently-sized flat SAE.

## Common structure

Each subdirectory is a self-contained project following the same layout:

```
arxiv.<id>-<name>/
├── pipeline/                  # paper2code artifacts: algorithm extraction,
│                               # concept analysis, and implementation plan
│                               # (YAML), including hyperparameters, equation-
│                               # by-equation citations, and explicitly
│                               # flagged assumptions where the paper is
│                               # underspecified
├── src/<package>/              # the implementation (models, training,
│                               # algorithms, data, evaluation)
├── experiments/                # train / eval / ablate CLI scripts
├── tests/                      # pytest smoke tests (all passing)
├── main.py                     # CLI entry point
├── pyproject.toml
└── README.md                   # paper-specific details, scope, and limitations
```

## Scope and limitations

None of the three papers' full-scale experiments (which require production
LLM activations — Gemma 2-2B or GPT2-small — over hundreds of millions to a
billion tokens, on multi-GPU hardware) were run in this environment; that is
outside its compute and network budget. Each implementation is instead:

- **complete and faithful to its paper's algorithm** — every equation,
  loss term, and named algorithm (including verbatim pseudocode boxes where
  the paper provides one) is implemented, not stubbed;
- **validated end-to-end on synthetic, CPU-scale data** with a known
  ground-truth generative structure matching that paper's theory, via a
  passing pytest smoke-test suite in each subdirectory;
- **wired up for real-scale use** via a `collect_activations`-style
  integration point, for anyone who wants to point it at a real model and
  run it on their own hardware.

Each subdirectory's own README documents its paper's specific hyperparameters,
reproduction targets, and any places where the paper itself was ambiguous or
underspecified (and how that gap was resolved).
