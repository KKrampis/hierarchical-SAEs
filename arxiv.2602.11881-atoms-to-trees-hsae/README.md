# HSAE: Hierarchical Sparse Autoencoders

Reference implementation of **"From Atoms to Trees: Building a Structured
Feature Forest with Hierarchical Sparse Autoencoders"** (arXiv:[2602.11881v1](https://arxiv.org/abs/2602.11881v1)),
built end-to-end with the [`paper2code`](../.claude/skills/paper2code/SKILL.md)
Claude Code skill.

HSAE jointly trains a series of JumpReLU sparse autoencoders of increasing
dictionary size while explicitly learning parent-child relationships between
their features, turning the "feature splitting" phenomenon into a
deliberate hierarchical indexing structure (a "conceptual forest") instead
of treating it as a nuisance.

## Pipeline artifacts

The `pipeline/` directory holds the paper2code intermediate representations
this codebase was built from:

- `01_algorithm_extraction.yaml` — every algorithm, equation, and
  hyperparameter extracted from the paper, with source citations.
- `02_concept_analysis.yaml` — the paper's structure, method decomposition,
  and experiment/success-criteria mapping.
- `03_implementation_plan.yaml` — the file structure, component-by-component
  implementation plan, validation approach, environment setup, and
  implementation strategy this code follows.

## Scope and limitations

The paper's full-scale experiments train HSAE on **100M residual-stream
activations** collected from **gemma2-2b** (and additionally qwen3-4b), on
an **NVIDIA A800 GPU for ~6 hours per run**. That is outside the compute and
network budget of the environment this code was written in, so:

- The codebase is **complete and faithful to the paper's algorithm** (every
  mechanism in Sections 3.1–3.3 and Appendix A.2 is implemented: the
  parent-children constraint loss, the random feature perturbation, the
  similarity-based hierarchy update with partial-tree quantile filtering,
  the alternating-optimization trainer, and the dynamic sparsity
  controller).
- It is **validated on a synthetic, CPU-sized dataset** with a known
  ground-truth hierarchy (`hsae.data.SyntheticHierarchicalActivations`)
  rather than on a real LLM's activations.
- A `collect_activations` function is provided as the integration point for
  real activations (HuggingFace / TransformerLens), for users who want to
  run the full-scale experiment on their own hardware.
- The AutoInterp LLM-judge pipeline (`hsae.evaluation.autointerp`) is fully
  implemented (sampling, example retrieval, prompt construction exactly as
  given in Appendix C) but requires the caller to supply their own
  `judge_fn` — no LLM API credentials are configured in this repository.

The paper's reported reproduction targets (e.g. Hamming distance 21.3 for
full HSAE vs. 29.7 without the constraint loss) are recorded in
`pipeline/01_algorithm_extraction.yaml` for anyone running the real-scale
configuration.

## Installation

```bash
uv init --no-readme --name hsae   # if you haven't already
uv sync                            # installs dependencies from pyproject.toml
uv add --optional viz umap-learn matplotlib          # optional: geometric visualization
uv add --optional real-models transformers transformer-lens  # optional: real-model activations
```

## Usage

```bash
# Train an HSAE on synthetic data (fast, CPU, ~seconds)
uv run python main.py train --steps 300

# Train the post-hoc Baseline (independent SAEs + post-hoc hierarchy) instead
uv run python main.py train --baseline --steps 300

# Compare HSAE vs. Baseline hierarchy-consistency metrics
uv run python main.py eval --steps 300

# Reproduce the qualitative ablation orderings (Figure 4 mechanisms, Figure 9 tree topology)
uv run python main.py ablate --steps 300
```

## Tests

```bash
uv run pytest tests/ -v
```

`tests/test_smoke.py` runs the full pipeline (model construction, hierarchy
update, training, evaluation, baseline construction) end-to-end on
synthetic data in well under a minute.

## Package layout

```
src/hsae/
├── config.py                    # all hyperparameters (paper defaults + a small_synthetic() preset)
├── models/
│   ├── jumprelu_sae.py          # single-level JumpReLU SAE (straight-through, ghost grads)
│   └── hsae.py                  # multi-level HSAE wrapper + hierarchy state
├── algorithms/
│   ├── hierarchy.py             # similarity-based hierarchy update + partial-tree filter
│   ├── perturbation.py          # parent-children feature perturbation (Section 3.2)
│   └── sparsity_control.py      # dynamic sparsity control (Appendix A.2)
├── training/
│   ├── losses.py                # L_SAE, L_PC, L_HSAE
│   └── trainer.py                # alternating-optimization trainer
├── data/
│   └── activations.py           # synthetic dataset + real-model activation collector
└── evaluation/
    ├── hierarchy_metrics.py     # Hamming distance, conditional co-activation probabilities
    ├── baseline.py               # post-hoc Baseline construction
    └── autointerp.py             # LLM-judge hierarchy evaluation (Appendix C)
```

## Key hyperparameters (from the paper)

| Hyperparameter | Value | Source |
|---|---|---|
| Dictionary sizes | 2048, 4096, 8192, 16384 | Section 4 |
| Target L0 sparsity | 50 | Section 4 |
| Parent-child constraint weight `rho` | 0.01 | Section 3.1/3.2 |
| Perturbation rate `r` | 5% | Section 3.2 |
| Hierarchy update period | every 5000 steps | Section 3.3 |
| Partial-tree exclusion quantile | 20% | Section 3.3 |
| Batch size | 1024 | Appendix A.1 |
| Adam betas | (0, 0.995) | Appendix A.1 |
| Sparsity-control decay rate `eta` | 0.001 | Appendix A.2 |
| Sparsity-control EMA momentum | 0.999 | Appendix A.2 |

See `pipeline/01_algorithm_extraction.yaml` for the complete list, including
values the paper does not specify numerically (marked `[INFERRED]`, e.g. the
exact Adam learning rate and the dynamic-sparsity-control step size).
