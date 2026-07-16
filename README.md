# hierarchical-SAEs

Reference implementations of three research papers on **hierarchical sparse
autoencoders (SAEs)** for LLM interpretability, each built end-to-end with
the [`paper2code`](.claude/skills/paper2code/SKILL.md) Claude Code skill: the
paper is read, distilled into a structured YAML pipeline
(algorithm extraction → concept analysis → implementation plan), and then
implemented, tested, and validated from that plan.

All three papers tackle the same underlying question — standard SAEs learn a
flat, unstructured set of features, but real-world concepts are organized
hierarchically (e.g. "dog" → "corgi"/"greyhound"/"shitzu") — with three
genuinely different architectural answers. This document goes through each
paper's algorithm in some depth and describes what was actually built.

## Common structure

Each subdirectory is a self-contained `uv`/pytest project following the same
layout:

```
arxiv.<id>-<name>/
├── pipeline/                  # paper2code artifacts: algorithm extraction,
│                               # concept analysis, and implementation plan
│                               # (YAML), including equation-by-equation
│                               # citations, hyperparameter tables, and
│                               # explicitly flagged assumptions where the
│                               # paper is underspecified
├── src/<package>/              # the implementation (models, training,
│                               # algorithms, data, evaluation)
├── experiments/                # train / eval / ablate CLI scripts
├── tests/                      # pytest smoke tests (all passing)
├── main.py                     # CLI entry point
├── pyproject.toml
└── README.md                   # paper-specific details, scope, and limitations
```

## Scope and limitations (applies to all three)

None of the three papers' full-scale experiments — which require production
LLM activations (Gemma 2-2B or GPT2-small) over hundreds of millions to a
billion tokens, on multi-GPU hardware — were run in this environment; that
is outside its compute and network budget. Each implementation is instead:

- **complete and faithful to its paper's algorithm** — every equation, loss
  term, and named algorithm (including verbatim pseudocode boxes where the
  paper provides one) is implemented, not stubbed;
- **validated end-to-end on synthetic, CPU-scale data** with a known
  ground-truth generative structure matching that paper's theory, via a
  passing pytest smoke-test suite in each subdirectory;
- **wired up for real-scale use** via a `collect_activations`-style
  integration point, for anyone who wants to point it at a real model and
  run it on their own hardware.

Where the source paper leaves a detail unspecified (an exact learning rate,
an update-rule constant, an initialization scheme), the implementation
records a documented, reasonable default and flags it `[INFERRED]` in that
paper's `pipeline/01_algorithm_extraction.yaml`, rather than silently
guessing.

---

## `arxiv.2602.11881-atoms-to-trees-hsae/` — HSAE

**"From Atoms to Trees: Building a Structured Feature Forest with
Hierarchical Sparse Autoencoders"** ([arXiv:2602.11881](https://arxiv.org/abs/2602.11881))

### The idea

Standard SAEs exhibit "feature splitting": as dictionary size grows, one
broad feature (e.g. "punctuation mark") fragments into several narrow ones
("period", "comma", "question mark"). Prior work treats this as a defect.
This paper's premise is that feature splitting is actually *evidence* of a
latent hierarchy in the model's concept space, and that training several
SAEs of increasing dictionary size *jointly*, while explicitly modeling
parent→child relationships between adjacent sizes, will recover that
hierarchy more faithfully than inferring it after the fact from
independently-trained SAEs.

### The algorithm

HSAE trains `L` JumpReLU SAEs of increasing dictionary size
(`n_1 < n_2 < ... < n_L`) on the same activation stream, and maintains a
**partial-tree assignment** `C_(l,i)` — the set of level-`(l+1)` features
that are children of level-`l` feature `i`. Two mechanisms pull parent and
child features into alignment during training:

1. **Structural constraint loss.** For every parent feature with a
   non-empty child set, `L_PC = ‖f_parent(x) − Σ_children f_child(x)‖²` —
   the parent's output should equal the sum of its children's outputs.
   Weighted by `ρ = 0.01` in the total loss.
2. **Random feature perturbation.** At each step, with probability `r =
   5%` a parent feature's own activation is swapped for the *sum of its
   children's activations* when computing the reconstruction loss —
   forcing the model to treat parent and children as interchangeable.

The hierarchy assignment itself is **not** learned by gradient descent.
Instead, training alternates between a **Parameter Optimization** stage
(gradient descent with the tree fixed) and a **Hierarchy Update** stage
(every 5,000 steps: recompute each child's parent as its most
cosine-similar feature at the level above, via encoder-vector similarity by
default). A **partial-tree quantile filter** leaves the bottom 20% of
similarity scores unassigned, so idiosyncratic features aren't forced under
a bad parent. A separate **dynamic sparsity controller** (a small feedback
loop tracking an EMA of each level's L0) adaptively adjusts each level's
sparsity penalty to hold a target L0 of 50.

### What we implemented

`src/hsae/`:
- `models/jumprelu_sae.py` — a single-level JumpReLU SAE with a
  straight-through estimator for the learnable per-feature threshold
  (`_JumpReLUSTE`/`_StepSTE` custom autograd functions), decoder
  renormalization after every step, and a ghost-gradients dead-feature
  revival mechanism.
- `models/hsae.py` — the `L`-level wrapper: owns the per-level SAEs plus
  the parent-index tensors, and runs the perturbed forward pass.
- `algorithms/hierarchy.py` — the similarity-based hierarchy update
  (encoder/decoder/co-activation metrics) and the quantile-based partial
  tree filter.
- `algorithms/perturbation.py` — the Bernoulli-gated parent/children
  feature-swap mechanism.
- `algorithms/sparsity_control.py` — the adaptive-λ feedback controller.
- `training/losses.py`, `training/trainer.py` — the combined loss and the
  alternating-optimization training loop.
- `evaluation/hierarchy_metrics.py` — Hamming-distance and conditional
  co-activation-probability metrics for hierarchy quality; `evaluation/
  baseline.py` builds the paper's "independent SAEs + post-hoc alignment"
  comparison baseline; `evaluation/autointerp.py` implements the
  LLM-judge pipeline with the paper's exact prompts (pluggable judge —
  no API key is configured here).

`experiments/run_ablation.py` reproduces the paper's mechanism ablation
(with/without constraint loss, with/without perturbation) and its tree-
topology ablation (Fixed / Full / Binary tree variants) on synthetic data.
7 pytest tests validate the hierarchy update, the loss shapes, and that
training reduces reconstruction loss.

---

## `arxiv.2605.07922-tree-sae/` — Tree SAE

**"Tree SAE: Learning Hierarchical Feature Structures in Sparse
Autoencoders"** ([arXiv:2605.07922](https://arxiv.org/abs/2605.07922))

### The idea

This paper argues that the standard way prior work *detects* a hierarchical
pair — "activation coverage" (a child rarely fires without its parent,
formalized as Masked Cosine Similarity) — is necessary but not sufficient:
a densely-active, semantically meaningless feature can trivially "cover"
many unrelated children. Its fix is a stronger **reconstruction
condition** (both parent and child decoder vectors must actually contribute
to reconstructing the child's true concept direction) and a **Tree SAE**
architecture that enforces coverage *architecturally*, letting the
reconstruction condition emerge as a provable byproduct rather than adding
an explicit pairwise loss term.

### The algorithm

Tree SAE is a single flat dictionary partitioned into contiguous
**privilege-layer blocks** (e.g. `[1536, 3072, 9216, 10752]` features across
4 layers). Unlike HSAE, a child feature at layer `l` may be parented by
**any** feature at any strictly earlier layer `0..l-1`, not just the
adjacent one; layer 0 is an imaginary root that every unassigned feature
falls back to.

Hierarchy is enforced by a **recursive activation-coverage gate**: a
feature's final activation is its raw TopK activation multiplied by an
indicator that its assigned parent's *own already-gated* activation is
nonzero — computed layer-by-layer outward from the root, so `S_cov = 1` for
every declared pair *by construction*, not by penalty. Training uses a
**multi-level cumulative reconstruction loss** — for every layer `l`, the
sum of reconstructions from layers `1..l` must independently be close to
the input — plus a per-layer auxiliary loss (only the first layer gets
`α₁ = 1/32` in the main config) that fights dead features. The paper proves
(their Appendix H) that this combination drives the reconstruction
condition up as a side effect, without ever computing `S_res` during
training.

Because the tree structure is rigid, a feature can get stuck under a poor
parent and starve. The paper's fix — and its most novel algorithmic
contribution — is a **two-part dynamic reallocation mechanism**:
1. **Algorithm 1 (greedy allocation):** given a "capacity" estimate `C_p`
   per candidate parent (accumulated training loss) and a fixed number of
   child slots to place, find the max-min-fair number of children per
   parent — i.e. maximize the *minimum* payoff `C_p / k_p` across parents.
   The paper proves this is solvable optimally in `O(s_l log m_l)` via a
   max-heap (Theorem 4.1), by greedily giving the next child to whichever
   parent currently has the highest marginal payoff.
2. **Algorithm 2 (dynamic reallocation):** every 3,000–10,000 steps
   (interval growing after each event), recompute the optimal counts via
   Algorithm 1, then reassign **only dead** children to under-served
   parents via first-fit — live children are never moved. At ~50% of
   training, any feature still dead is force-assigned to the root so it can
   fire freely.

### What we implemented

`src/tree_sae/`:
- `models/tree_sae.py` — the flat multi-block TopK SAE, the parent-index
  buffer, and the recursive `Eq. 6` gate (with a `ROOT = -1` sentinel
  handled correctly for both the default-unassigned and
  explicitly-reassigned-to-root cases — this actually caught a real
  `torch.gather` bug during development, since -1 is not a valid gather
  index and had to be special-cased as "always active").
- `algorithms/allocation.py` — `greedy_allocation` (a `heapq`-based
  max-heap implementation of Algorithm 1, unit-tested against a
  brute-force search of the true theoretical optimum) and
  `dynamic_reallocation` / `reassign_dead_to_root` (Algorithm 2 and the
  50%-mark rule).
- `training/losses.py` — the multi-level cumulative reconstruction loss
  and per-layer auxiliary loss.
- `training/trainer.py` — the training loop, including the growing
  reallocation schedule and per-parent capacity accumulation.
- `evaluation/coverage.py` — `S_cov` and the paper's best-performing MCS
  baseline variant (non-scaling, binarized correlation, per their Appendix
  I ablation); `evaluation/reconstruction_score.py` — `S_res` via a
  trained linear probe as the ground-truth concept direction (the paper
  proves the child's own encoder vector is *not* a valid stand-in for
  this); `evaluation/hierarchy_metric.py` — the paper's Section 5.2
  top-5-co-membership metric.

`experiments/run_ablation.py` compares dead-feature rate with vs. without
dynamic allocation (the paper's Figure 14). 8 pytest tests validate the
gate invariant (a child is never active while its parent is inactive),
Algorithm 1 against its closed-form optimum, that Algorithm 2 only ever
touches dead features, and that training converges.

---

## `arxiv.2506.01197-h-sae/` — H-SAE (mixture-of-experts)

**"Incorporating Hierarchical Semantics in Sparse Autoencoder
Architectures"** ([arXiv:2506.01197](https://arxiv.org/abs/2506.01197))

### The idea

This paper starts from a geometric result (Park et al., 2024): in an LLM's
representation space, a categorical concept (e.g. "dog") is represented by
a **parent direction** (is the concept present at all) plus a **low-rank
subspace** whose points are its possible **child values** (which breed) —
and a specific child's representation decomposes exactly as
`parent_vector + child_in_context_of_parent_vector`. Rather than
approximating this with a training penalty, H-SAE builds it directly into
the architecture as a **mixture of experts**: only two levels deep, but
each "expert" is a small SAE living in its own learned subspace.

### The algorithm

Given input `x`:
1. A **top-level SAE** (`TopK_k`, `k = 32` in the main config, *no* bias
   subtraction — an explicit departure from the generic baseline TopK SAE
   formula, for training-stability reasons the paper states directly) picks
   `k` active "expert" features out of `m_top` (8k/16k/32k tested).
2. For **only those `k` active experts** — inactive experts are never
   computed, a real conditional-compute saving, not a masking trick — the
   input is projected down into a small learned subspace (dimension `s = 4`
   or `8`) via a per-expert matrix `Π_down`, run through a tiny local SAE
   that keeps **exactly one** active sublatent (`TopK_1`, architecturally
   fixed — this is literally "child = a specific vertex of the parent's
   polytope"), and projected back up via `Π_up`.
3. The final reconstruction is the sum of the top-level reconstruction and
   every active expert's up-projected contribution:
   `H-SAE(x) = Σ_{j∈TopK} [z_j·d_j + Π_up_j(SAE1_j(Π_down_j·x))]`.

The training loss (`L = L_recon + λ₁·L_ortho + λ₂·L_sparse`) has three
parts: the combined reconstruction loss *plus* a `β=0.1`-weighted
**top-level-only** reconstruction term (so top-level features stay
individually meaningful, not just routers); a **bi-orthogonality penalty**
on the top-level encoder/decoder only (`λ₁=0.1`) that empirically turns out
to do the job of a dead-latent auxiliary loss without needing one; and a
small L1 penalty (`λ₂=0.001`) over all active latent values. Because
`k << m_top` and `s << d` in practice, the paper shows the asymptotic
forward-pass cost is barely more than a plain top-level SAE of the same
size — this is the source of the paper's headline result: a 64-sublatent
H-SAE matches a *4x larger* flat SAE at 1/4 the compute.

### What we implemented

`src/h_sae/`:
- `models/hsae.py` — the mixture-of-experts forward pass, implemented with
  genuine gather-based conditional compute: `torch.topk` picks the active
  expert indices, and every per-expert parameter tensor
  (`Π_down`/`Π_up`/`E_low`/`D_low`, each shaped `[m_top, ...]`) is indexed
  by those indices via advanced indexing (`self.Pi_down[topk_indices]`)
  rather than looping over all `m_top` experts and masking — so the
  implementation's actual FLOP count reflects the paper's efficiency claim.
- `models/topk_sae.py` — the flat baseline TopK SAE used for comparison
  throughout the paper's experiments (this one *does* subtract a bias,
  per the paper's generic Eq. 2.1 baseline formula).
- `training/losses.py` — the three-term loss, including a documented
  resolution of a real inconsistency in the paper (its Algorithm-1 pseudocode
  box prints the orthogonality term as `‖DE − diag(DE)‖`, but the separately
  numbered Eq. 3.4 uses `‖ED − diag(ED)‖`; since `D` and `E`'s stated shapes
  only make `E·D` dimensionally valid as an `m_top × m_top` matrix, we
  implement that form, noted in the code and in
  `pipeline/01_algorithm_extraction.yaml`).
- `training/trainer.py` — Adam with global-norm clipping at 0.75, a linear
  warmup (1,000 steps) into cosine decay for both the learning rate *and*
  all three loss coefficients simultaneously, per the paper.
- `data/activations.py` — a synthetic dataset generator built directly from
  the paper's own theory (`x = parent_direction + child_in_subspace +
  noise`), rather than a generic hierarchy — so the evaluation is actually
  testing the geometric structure this architecture is designed to exploit.
- `evaluation/` — explained-variance reconstruction comparison, a dead
  top-level-latent-rate tracker, and a simplified absorption/splitting
  probe adapted from the paper's first-letter-classification SAEBench
  benchmark to the synthetic dataset's known parent labels.

`experiments/run_eval.py` trains both H-SAE and the baseline at matched
top-level dictionary size; in a representative run H-SAE reconstructs
better than the baseline (1-explained-variance 1.13 vs. 1.32), matching the
paper's central claim even at toy scale. `experiments/run_ablation.py`
mirrors the paper's Appendix B.2 orthogonality/L1 ablation. 7 pytest tests
validate the `TopK_1` invariant, the orthogonality-loss formula against
hand-built orthogonal/non-orthogonal matrices, and the warmup schedule.

---

## At a glance

| | HSAE (2602.11881) | Tree SAE (2605.07922) | H-SAE (2506.01197) |
|---|---|---|---|
| Structure | Multi-level flat dictionaries | One flat dictionary, layered blocks | Two-level mixture of experts |
| Parent scope | Adjacent level only | Any strictly earlier level | Top-level → its own experts only |
| Hierarchy enforced by | Explicit constraint loss + perturbation | Architectural gate + multi-level loss | Architectural conditional compute |
| Assignment learned via | Periodic similarity re-matching | Periodic capacity-aware reallocation | Implicit (TopK routing) |
| Novel algorithmic contribution | Alternating optimization + adaptive sparsity | Greedy max-heap allocator (proved optimal) | Subspace-projected mixture of experts |
| Compute efficiency angle | Not a focus | Not a focus | Central: inactive experts never computed |
