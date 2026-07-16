# Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders

Reference implementation of **"Tree SAE: Learning Hierarchical Feature
Structures in Sparse Autoencoders"** (arXiv:[2605.07922v2](https://arxiv.org/abs/2605.07922)),
built end-to-end with the [`paper2code`](../.claude/skills/paper2code/SKILL.md)
Claude Code skill.

The paper shows that the standard "activation coverage" criterion used by
prior work to identify parent-child feature pairs post-hoc (Masked Cosine
Similarity / MCS) is insufficient and produces semantically incoherent
pairs. It introduces a stronger "reconstruction condition" and a Tree SAE
architecture that encodes tree structure directly within the feature set
via a recursive activation-coverage gate (Eq. 6) plus a multi-level
reconstruction loss (Eq. 7-9), paired with a dynamic, capacity-aware
feature-reallocation mechanism (Algorithms 1-2) that combats the
dead-feature problem the rigid tree otherwise causes.

## Pipeline artifacts

The `pipeline/` directory holds the paper2code intermediate representations
this codebase was built from:

- `01_algorithm_extraction.yaml` — every algorithm, equation, and
  hyperparameter extracted from the paper, including the verbatim Algorithm
  1 (greedy allocation) and Algorithm 2 (dynamic reallocation) pseudocode,
  with source citations.
- `02_concept_analysis.yaml` — the paper's structure, method decomposition,
  and experiment/success-criteria mapping.
- `03_implementation_plan.yaml` — the file structure, component-by-component
  implementation plan, validation approach, environment setup, and
  implementation strategy this code follows.

## Scope and limitations

The paper's main experiments train Tree SAE on **GPT2-small layer-5
residual-stream activations**, over **500M tokens of The Pile**, at
dictionary sizes up to **49,152**. That is outside the network/compute
budget of the environment this code was written in, so:

- The codebase is **complete and faithful to the paper's algorithm**: the
  flat multi-block TopK SAE, the recursive activation-coverage gate
  (Eq. 6), the multi-level reconstruction + per-layer auxiliary loss
  (Eq. 7-9), the greedy allocation algorithm (Algorithm 1, verified against
  its theoretical max-min-fair optimum in `tests/test_smoke.py`), and the
  dynamic dead-feature reallocation algorithm (Algorithm 2, including the
  ~50%-of-training "move dead features to root" rule).
- It is **validated on a synthetic, CPU-sized dataset** with a known
  ground-truth multi-level tree (`tree_sae.data.SyntheticHierarchicalActivations`)
  rather than on real GPT2-small activations.
- A `collect_activations` function is provided as the integration point for
  real activations (HuggingFace / TransformerLens + The Pile), for users who
  want to run the full-scale experiment on their own hardware.
- At this tiny synthetic scale, `experiments/run_eval.py` and
  `experiments/run_ablation.py` are **demonstration/reproduction tools, not
  strict correctness tests** — with only a few features per layer, a few
  hundred training steps, and a purely random initial allocation that is
  only ever repaired for *dead* features (by design — Algorithm 2 never
  moves an already-alive child, matching the paper exactly, including its
  own noted limitation in Appendix A that "Tree SAEs' structure still
  contains pairs that violate the reconstruction condition"), the
  hierarchical-pair metric and dead-feature-rate comparison can legitimately
  come out flat or noisy. `tests/test_smoke.py` is the correctness check
  (8 tests, all passing): it validates the gate invariant, Algorithm 1
  against its closed-form theoretical optimum, that Algorithm 2 only ever
  touches dead features, and that training reduces reconstruction loss.

The paper's reported reproduction targets (Tables 1-4, Figures 3-9, 14) are
recorded in `pipeline/01_algorithm_extraction.yaml` and
`pipeline/02_concept_analysis.yaml` for anyone running the real-scale
configuration.

## Installation

```bash
uv init --no-readme --name tree-sae   # if you haven't already
uv sync                                # installs dependencies from pyproject.toml
uv add --optional real-models transformers transformer-lens datasets  # optional: real activations
```

## Usage

```bash
# Train a Tree SAE on synthetic data (fast, CPU, ~seconds)
uv run python main.py train --steps 300

# Evaluate Tree Structure vs. MCS hierarchy-pair detection, plus S_res/S_cov for a sample pair
uv run python main.py eval --steps 300

# Compare dead-feature rate with vs. without dynamic allocation (Figure 14 direction)
uv run python main.py ablate --steps 300
```

## Tests

```bash
uv run pytest tests/ -v
```

`tests/test_smoke.py` runs the full pipeline (model construction, gate
invariant checking, Algorithm 1 correctness vs. its theoretical optimum,
Algorithm 2's dead-feature-only invariant, training, evaluation) end-to-end
on synthetic data in well under a minute.

## Package layout

```
src/tree_sae/
├── config.py                        # all hyperparameters (paper defaults + a small_synthetic() preset)
├── models/
│   └── tree_sae.py                  # flat multi-block TopK SAE + allocation state + Eq.6 gate
├── algorithms/
│   └── allocation.py                # Algorithm 1 (greedy) + Algorithm 2 (dynamic reallocation)
├── training/
│   ├── losses.py                    # L_recons, L_aux,l, L_tree (Eq. 7-9)
│   └── trainer.py                   # training loop incl. reallocation schedule
├── data/
│   └── activations.py               # synthetic multi-level-tree dataset + real-model collector
└── evaluation/
    ├── coverage.py                  # S_cov (Eq.4) + MCS baseline (non-scaling binary, Appendix I)
    ├── reconstruction_score.py      # S_res (Eq.5) via linear probes
    └── hierarchy_metric.py          # Section 5.2 top-5 co-membership metric
```

## Key hyperparameters (from the paper)

| Hyperparameter | Value | Source |
|---|---|---|
| Dictionary size (main) | 24576 | Appendix C |
| L0 / TopK levels | 32, 48, 64, 80 | Section 5, Appendix C |
| Batch size | 5120 | Appendix C |
| Learning rate | 1e-4 (Adam) | Appendix C |
| Training tokens | 500M (The Pile) | Appendix C |
| Aux top-k | 256 | Appendix C |
| Aux coefficient (layer 1 / others) | 1/32 / 0 | Appendix C |
| Reallocation schedule | 3000 steps, +2000/event, cap 10000 | Appendix G |
| Dead-feature window | 10M tokens | Appendix G |
| Parent eligibility rate | 1 activation / 50,000 tokens | Appendix G |
| Dead-to-root reassignment | ~50% of training | Appendix G |

See `pipeline/01_algorithm_extraction.yaml` for the complete list, including
values the paper does not specify numerically (marked `[INFERRED]`, e.g. the
exact reallocation-interval growth unit and the gradient-normalization
scheme).

## Relationship to the companion HSAE implementation

This repository's structure mirrors `../arxiv.2602.11881-atoms-to-trees-hsae/`
(the companion Hierarchical Sparse Autoencoder paper). The two methods make
an interesting contrast: HSAE adds an *explicit* pairwise structural
constraint loss between adjacent levels, while Tree SAE relies on an
*architectural* activation-coverage gate plus a multi-level reconstruction
loss, provably (Appendix H) driving semantic alignment as a byproduct
without a direct pairwise loss term, and allows a child to be parented by
*any* earlier layer rather than only the immediately preceding one.
