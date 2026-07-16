"""Fast CPU smoke test: exercises the full H-SAE pipeline end-to-end on
synthetic data, per the "Validation Approach" in
pipeline/03_implementation_plan.yaml.
"""

from __future__ import annotations

import torch

from h_sae.config import HSAEConfig
from h_sae.data import SyntheticParentChildActivations, SyntheticTreeSpec
from h_sae.evaluation import dead_latent_rate, one_minus_explained_variance
from h_sae.models.hsae import HSAE
from h_sae.models.topk_sae import TopKSAE
from h_sae.training import HSAETrainer, TopKSAETrainer, orthogonality_loss, warmup_coefficient
from h_sae.utils import set_seed


def test_config_loads():
    config = HSAEConfig.small_synthetic()
    assert config.model.m_top == 12
    assert config.model.k <= config.model.m_top


def test_hsae_forward_shapes_and_topk1_invariant():
    config = HSAEConfig.small_synthetic()
    model = HSAE(config.model)
    x = torch.randn(6, config.model.d_model)
    x_hat, z, x_hat_high, z_low, topk_indices = model(x)

    assert x_hat.shape == (6, config.model.d_model)
    assert z.shape == (6, config.model.m_top)
    assert x_hat_high.shape == (6, config.model.d_model)
    assert z_low.shape == (6, config.model.k, config.model.m_low)
    assert topk_indices.shape == (6, config.model.k)

    # Top-level: exactly k nonzero entries per sample.
    assert torch.all((z != 0).sum(dim=-1) == config.model.k)
    # Low-level: exactly one nonzero sublatent per active expert (TopK_1 invariant).
    assert torch.all((z_low != 0).sum(dim=-1) == 1)


def test_orthogonality_loss():
    m_top = 4
    E = torch.eye(m_top, m_top)
    D = torch.eye(m_top, m_top)
    loss_orthogonal = orthogonality_loss(E, D)
    assert loss_orthogonal.item() < 1e-6

    D_non_orthogonal = torch.ones(m_top, m_top)
    loss_non_orthogonal = orthogonality_loss(E, D_non_orthogonal)
    assert loss_non_orthogonal.item() > loss_orthogonal.item()


def test_warmup_coefficient():
    assert warmup_coefficient(0, 100, 0.5) == 0.0
    assert warmup_coefficient(50, 100, 0.5) == 0.25
    assert warmup_coefficient(100, 100, 0.5) == 0.5
    assert warmup_coefficient(200, 100, 0.5) == 0.5


def test_hsae_training_reduces_loss():
    set_seed(0)
    config = HSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(
        d_model=config.model.d_model, m_top=config.model.m_top,
        m_low=config.model.m_low, subspace_dim=config.model.subspace_dim,
    )
    dataset = SyntheticParentChildActivations(spec, num_samples=max(config.data.training_vectors, config.optim.batch_size), seed=0)

    trainer = HSAETrainer(config)
    logs = trainer.train(dataset, total_steps=config.optim.total_steps, log_every=25)

    assert len(logs) >= 2
    assert logs[-1].recon_loss < logs[0].recon_loss


def test_baseline_topk_sae_training_reduces_loss():
    set_seed(0)
    config = HSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(
        d_model=config.model.d_model, m_top=config.model.m_top,
        m_low=config.model.m_low, subspace_dim=config.model.subspace_dim,
    )
    dataset = SyntheticParentChildActivations(spec, num_samples=max(config.data.training_vectors, config.optim.batch_size), seed=0)

    trainer = TopKSAETrainer(config)
    logs = trainer.train(dataset, total_steps=config.optim.total_steps, log_every=25)

    assert len(logs) >= 2
    assert logs[-1].recon_loss < logs[0].recon_loss

    model = TopKSAE.from_config(config.model)
    x = torch.randn(6, config.model.d_model)
    x_hat, z = model(x)
    assert x_hat.shape == (6, config.model.d_model)
    assert torch.all((z != 0).sum(dim=-1) == config.model.k)


def test_evaluation_suite_runs_end_to_end():
    set_seed(0)
    config = HSAEConfig.small_synthetic()
    spec = SyntheticTreeSpec(
        d_model=config.model.d_model, m_top=config.model.m_top,
        m_low=config.model.m_low, subspace_dim=config.model.subspace_dim,
    )
    dataset = SyntheticParentChildActivations(spec, num_samples=max(config.data.training_vectors, config.optim.batch_size), seed=0)
    eval_dataset = SyntheticParentChildActivations(spec, num_samples=200, seed=1)

    trainer = HSAETrainer(config)
    trainer.train(dataset, total_steps=config.optim.total_steps)

    with torch.no_grad():
        x_hat, z, _, _, _ = trainer.model(eval_dataset.activations)

    ev = one_minus_explained_variance(eval_dataset.activations, x_hat)
    dead = dead_latent_rate(z)
    assert 0.0 <= ev
    assert 0.0 <= dead <= 1.0
