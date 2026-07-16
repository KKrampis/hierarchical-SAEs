"""AutoInterp Hierarchy Judge pipeline (Section 4.3, Appendix C).

Samples parent-child feature pairs, retrieves example activating sequences
for each feature (10 highest-activation + 5 sampled proportional to
activation value, per Appendix C's "Feature Pairs Sampling and Data
Collection"), formats them with <<token>> highlighting, and queries an LLM
judge with the exact system/user prompt templates quoted in Appendix C to
classify whether a valid parent-child relationship holds.

The LLM call itself is injected by the caller via `judge_fn` (this repo has
no configured LLM API credentials); this module implements everything
around that call — sampling, formatting, prompt construction, response
parsing — faithfully to the paper.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from hsae.algorithms.hierarchy import UNASSIGNED
from hsae.models.hsae import HSAE

SYSTEM_PROMPT = (
    "We're studying neurons in a sparse autoencoder (SAE) within a neural network. "
    "Each neuron activates on specific words, substrings, or concepts in short "
    "documents, with activating words indicated by << ... >>. You will be given "
    "two sets of documents where two different neurons activate. Your task is to "
    "compare the activation patterns of these two neurons and determine if there is "
    "a parent-child relationship between them. A parent-child relationship means "
    "that one neuron's activating concept (the child) is a subset or a more specific "
    "version of the other neuron's activating concept (the parent). Analyze the "
    "provided examples and output your judgment in the following format:\n\n"
    "HaveRelationship: [Yes/No]\n"
    "Confidence: [High/Medium/Low]\n\n"
    "Do not include any additional text, explanations, or formatting."
)

USER_PROMPT_TEMPLATE = (
    "Here are the activating documents for Neuron A:\n"
    "{examples_a}\n\n"
    "And for Neuron B:\n"
    "{examples_b}\n\n"
    "Based on these documents, determine if Neuron A and Neuron B have a parent-child "
    "relationship."
)


@dataclass
class FeaturePair:
    parent_level: int
    parent_index: int
    child_level: int
    child_index: int


@dataclass
class JudgeResult:
    has_relationship: bool
    confidence: str


def sample_feature_pairs(model: HSAE, num_pairs: int, seed: int = 0) -> list[FeaturePair]:
    """Randomly samples up to `num_pairs` assigned parent-child feature pairs
    from the model's hierarchy (Appendix C: "randomly sample 5000 parent-child
    pairs")."""
    rng = random.Random(seed)
    all_pairs: list[FeaturePair] = []
    for level in range(model.num_levels - 1):
        parent_idx = model.parent_idx[level]
        for child_index in range(parent_idx.shape[0]):
            parent_index = int(parent_idx[child_index])
            if parent_index != UNASSIGNED:
                all_pairs.append(FeaturePair(level, parent_index, level + 1, child_index))
    rng.shuffle(all_pairs)
    return all_pairs[:num_pairs]


def retrieve_examples(
    feature_activations: torch.Tensor,
    texts: Sequence[str],
    num_top: int = 10,
    num_sampled: int = 5,
    seed: int = 0,
) -> list[str]:
    """Retrieves the `num_top` highest-activation examples plus `num_sampled`
    examples drawn with probability proportional to activation value, matching
    Appendix C's sampling strategy. `texts` must already contain <<token>>
    highlighting around the activating span for each example."""
    if feature_activations.shape[0] != len(texts):
        raise ValueError("feature_activations and texts must be the same length")

    active_mask = feature_activations > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    if active_indices.numel() == 0:
        return []

    active_values = feature_activations[active_indices]
    order = torch.argsort(active_values, descending=True)
    top_indices = active_indices[order[:num_top]].tolist()

    remaining = [i for i in active_indices.tolist() if i not in set(top_indices)]
    if remaining and num_sampled > 0:
        weights = feature_activations[remaining].clamp_min(1e-8)
        probs = (weights / weights.sum()).tolist()
        rng = random.Random(seed)
        sampled = rng.choices(remaining, weights=probs, k=min(num_sampled, len(remaining)))
    else:
        sampled = []

    return [texts[i] for i in top_indices + sampled]


def format_examples(examples: list[str]) -> str:
    return "\n".join(f"- {example}" for example in examples)


def build_prompts(examples_a: list[str], examples_b: list[str]) -> tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        examples_a=format_examples(examples_a),
        examples_b=format_examples(examples_b),
    )
    return SYSTEM_PROMPT, user_prompt


def parse_judge_response(response: str) -> JudgeResult:
    has_relationship = False
    confidence = "Low"
    for line in response.strip().splitlines():
        if line.strip().lower().startswith("haverelationship"):
            has_relationship = line.split(":", 1)[1].strip().lower().startswith("yes")
        elif line.strip().lower().startswith("confidence"):
            confidence = line.split(":", 1)[1].strip()
    return JudgeResult(has_relationship=has_relationship, confidence=confidence)


JudgeFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> raw model response


def default_judge_fn(system_prompt: str, user_prompt: str) -> str:
    raise NotImplementedError(
        "No LLM judge is configured. Pass a `judge_fn(system_prompt, user_prompt) -> str` "
        "callable that calls your LLM of choice (the paper uses Qwen3-Max, Appendix C) to "
        "evaluate_autointerp(...). This repository has no configured API credentials."
    )


def evaluate_autointerp(
    model: HSAE,
    activations: torch.Tensor,
    texts: Sequence[str],
    num_pairs: int = 5000,
    judge_fn: JudgeFn = default_judge_fn,
    seed: int = 0,
) -> float:
    """Runs the full AutoInterp hierarchy-judge pipeline and returns the fraction
    of sampled pairs the judge classifies as a valid hierarchical relationship."""
    with torch.no_grad():
        feature_acts = [sae.encode(activations) for sae in model.levels]

    pairs = sample_feature_pairs(model, num_pairs, seed=seed)
    if not pairs:
        return float("nan")

    results = []
    for pair in pairs:
        examples_a = retrieve_examples(feature_acts[pair.parent_level][:, pair.parent_index], texts, seed=seed)
        examples_b = retrieve_examples(feature_acts[pair.child_level][:, pair.child_index], texts, seed=seed)
        if not examples_a or not examples_b:
            continue
        system_prompt, user_prompt = build_prompts(examples_a, examples_b)
        response = judge_fn(system_prompt, user_prompt)
        results.append(parse_judge_response(response).has_relationship)

    if not results:
        return float("nan")
    return sum(results) / len(results)
