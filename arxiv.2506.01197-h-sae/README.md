# H-SAE: Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures

Reference implementation of **"Incorporating Hierarchical Semantics in
Sparse Autoencoder Architectures"** (arXiv:[2506.01197v1](https://arxiv.org/abs/2506.01197)),
built end-to-end with the [`paper2code`](../.claude/skills/paper2code/SKILL.md)
Claude Code skill.

The paper translates Park et al. (2024)'s geometric theory of categorical
concepts — every concept has a binary "parent" feature plus a low-rank
subspace of "child" values — directly into a **mixture-of-experts SAE
architecture**: a sparse top-level SAE selects `k` active "expert" concepts,
and each activated expert runs its own small, low-dimensional-subspace
-restricted, `TopK=1` low-level SAE. Because inactive experts are never
computed at all (not just masked), this gives both a very large effective
dictionary size and large compute/memory savings versus a flat SAE, while
improving reconstruction and reducing feature splitting/absorption and
cross-lingual feature redundancy.

## Pipeline artifacts

The `pipeline/` directory holds the paper2code intermediate representations
this codebase was built from:

- `01_algorithm_extraction.yaml` — every algorithm, equation, and
  hyperparameter extracted from the paper, including the verbatim
  Algorithm 1 ("Hierarchical SAE Forward Pass and Loss Computation")
  pseudocode, with source citations and an explicit note on a real
  notational inconsistency found in the paper (Algorithm 1's `DE` vs.
  Eq. 3.4's `ED` in the orthogonality penalty).
- `02_concept_analysis.yaml` — the paper's structure, method decomposition,
  and experiment/success-criteria mapping.
- `03_implementation_plan.yaml` — the file structure, component-by-component
  implementation plan, validation approach, environment setup, and
  implementation strategy this code follows.

## Scope and limitations

The paper's main experiments train on **1 billion Gemma 2-2B layer-20
residual-stream activation vectors** collected from multilingual Wikipedia,
on an **8xA100 node**, at top-level dictionary sizes up to **32,000**. That
is outside the network/compute budget of the environment this code was
written in, so:

- The codebase is **complete and faithful to the paper's architecture**:
  the top-level SAE, the per-expert down/up subspace projections, the
  per-expert `TopK=1` low-level SAE, and the full multi-term loss (combined
  + top-level-only reconstruction, top-level-only bi-orthogonality penalty,
  and L1 sparsity), all exactly matching Algorithm 1 and Eq. 3.1-3.4. The
  forward pass genuinely **gathers only the `k` active experts'
  parameters** per batch element (via advanced indexing), matching the
  paper's conditional-compute efficiency claim rather than computing all
  experts and masking.
- It is **validated on a synthetic, CPU-sized dataset** whose generative
  structure directly mirrors the paper's own theoretical motivation:
  `x = parent_direction + child_in_subspace_direction + noise`
  (`h_sae.data.SyntheticParentChildActivations`) — rather than on real
  Gemma 2-2B activations.
- A `collect_activations` function is provided as the integration point for
  real activations (HuggingFace `transformers` + Wikipedia), for users who
  want to run the full-scale experiment on their own hardware.
- `experiments/run_eval.py` and `experiments/run_ablation.py` are
  **demonstration/reproduction tools**, not strict correctness tests, and
  at this tiny scale their numbers can be noisy — e.g. in one representative
  run, H-SAE beat the baseline TopK SAE on reconstruction (1-explained-var
  1.128 vs. 1.318, matching the paper's qualitative claim) but the
  simplified absorption proxy metric (built for a 12-parent-concept toy
  dataset, not real first-letter-classification tokens) did not
  consistently favor H-SAE the way the paper's real SAEBench-based
  benchmark does. `tests/test_smoke.py` is the correctness check (7 tests,
  all passing): it validates the `TopK_1` low-level architectural
  invariant, the orthogonality-loss formula against hand-constructed
  orthogonal/non-orthogonal matrices, the warmup schedule, and that both
  H-SAE and the baseline reduce their reconstruction loss during training.

The paper's reported reproduction targets (Figure 3, Figure 5 and its exact
cross-lingual set-difference table) are recorded in
`pipeline/01_algorithm_extraction.yaml` and `pipeline/02_concept_analysis.yaml`
for anyone running the real-scale configuration.

## Installation

```bash
uv init --no-readme --name h-sae   # if you haven't already
uv sync                             # installs dependencies from pyproject.toml
uv add --optional real-models transformers datasets  # optional: real activations
```

## Usage

```bash
# Train an H-SAE on synthetic data (fast, CPU, seconds)
uv run python main.py train --steps 300

# Compare H-SAE vs. the baseline TopK SAE: reconstruction, dead-latent rate, absorption proxy
uv run python main.py eval --steps 300

# Ablate the orthogonality and L1 losses (Appendix B.2 spirit)
uv run python main.py ablate --steps 300
```

## Tests

```bash
uv run pytest tests/ -v
```

`tests/test_smoke.py` runs the full pipeline (model construction, the
`TopK_1` invariant, the orthogonality-loss formula, the warmup schedule,
training for both H-SAE and the baseline, and evaluation) end-to-end on
synthetic data in a couple of minutes.

## Package layout

```
src/h_sae/
├── config.py                        # all hyperparameters (paper defaults + a small_synthetic() preset)
├── models/
│   ├── topk_sae.py                  # baseline TopK SAE (Eq. 2.1-2.2, Gao et al. 2024)
│   └── hsae.py                      # mixture-of-experts H-SAE (Eq. 3.1, Algorithm 1)
├── training/
│   ├── losses.py                    # L_recon, L_ortho, L_sparse, L (Eq. 3.2-3.4) + warmup
│   └── trainer.py                   # training loops (HSAETrainer, TopKSAETrainer)
├── data/
│   └── activations.py               # synthetic parent+child-subspace dataset + real-model collector
└── evaluation/
    ├── reconstruction.py            # 1 - explained variance
    ├── absorption.py                # simplified first-letter-classification-style absorption probe
    └── dead_latents.py              # dead top-level latent rate
```

## Key hyperparameters (from the paper)

| Hyperparameter | Value | Source |
|---|---|---|
| Top-level dictionary size | 8k / 16k / 32k | Section 4 |
| Top-level TopK `k` | 32 | Appendix A.2 |
| Sublatents per expert | 16 or 64 | Section 4 |
| Subspace dimension `s` | 4 (16 sublatents) / 8 (64 sublatents) | Appendix A.2 |
| Low-level TopK | 1 (architecturally fixed) | Eq. 3.1 |
| Orthogonality coefficient `lambda_1` | 0.1 | Appendix A.2 |
| Top-level-recon coefficient `beta` | 0.1 | Appendix A.2 |
| L1 coefficient `lambda_2` | 0.001 | Appendix A.2 |
| Batch size | 32,512 | Appendix A.2 |
| Peak / initial learning rate | 5e-4 / 1e-11 | Appendix A.2 |
| Warmup steps | 1000 | Appendix A.2 |
| Adam `b1` / grad-clip norm | 0.9 / 0.75 | Appendix A.2 |

See `pipeline/01_algorithm_extraction.yaml` for the complete list, including
values the paper does not specify numerically (marked `[INFERRED]`, e.g.
Adam's `b2` and the LeakyReLU negative slope).

## Relationship to the companion HSAE / Tree SAE implementations

This repository's structure mirrors `../arxiv.2602.11881-atoms-to-trees-hsae/`
and `../arxiv.2605.07922-tree-sae/`. All three papers tackle "hierarchy in
SAEs" but with genuinely different architectures: the other two are
multi-level *flat* dictionaries with soft/hard gating between levels
(adjacent-only for HSAE, any-earlier-layer for Tree SAE); this paper's H-SAE
is instead a **two-level mixture-of-experts** architecture with an explicit
per-expert low-rank subspace projection, directly implementing a specific
geometric theory (Park et al. 2024) of how categorical concepts are
represented, rather than inferring/enforcing structure between otherwise
flat dictionary features.
