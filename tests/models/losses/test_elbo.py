"""Tests for tnbbeta_vae.models.losses.elbo."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo


def _make_posterior_and_prior(
    batch_size: int, dim: int
) -> tuple[TNBBetaSpherical, TNBBetaSpherical, torch.Tensor]:
    mean_direction = torch.randn(batch_size, dim)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    posterior = TNBBetaSpherical(
        mean_direction, p=torch.full((batch_size,), 0.8), q=0.7, epsilon=1.0
    )

    prior_direction = torch.zeros(dim)
    prior_direction[0] = 1.0
    prior = TNBBetaSpherical(prior_direction, p=0.9, q=0.8, epsilon=1.0)

    z = posterior.rsample()
    return posterior, prior, z


def test_elbo_terms_are_finite_and_arithmetically_consistent() -> None:
    torch.manual_seed(0)
    batch_size, dim = 4, 5
    posterior, prior, z = _make_posterior_and_prior(batch_size, dim)
    x = torch.rand(batch_size, 3, 8, 8)
    reconstruction = torch.rand(batch_size, 3, 8, 8)

    terms = monte_carlo_elbo(x, reconstruction, z, posterior, prior)

    for key in ("elbo", "log_likelihood", "kl"):
        assert terms[key].shape == (batch_size,)
        assert torch.isfinite(terms[key]).all()

    assert torch.allclose(terms["elbo"], terms["log_likelihood"] - terms["kl"])
    assert torch.allclose(
        terms["kl"], posterior.log_prob(z) - prior.log_prob(z), atol=1e-5
    )


def test_elbo_penalizes_reconstruction_error() -> None:
    """A perfect reconstruction should score a higher log_likelihood than a bad one."""
    torch.manual_seed(1)
    batch_size, dim = 4, 5
    posterior, prior, z = _make_posterior_and_prior(batch_size, dim)
    x = torch.rand(batch_size, 3, 8, 8)

    perfect_terms = monte_carlo_elbo(x, x, z, posterior, prior)
    bad_terms = monte_carlo_elbo(
        x, torch.rand(batch_size, 3, 8, 8), z, posterior, prior
    )

    assert (perfect_terms["log_likelihood"] > bad_terms["log_likelihood"]).all()


def test_elbo_is_differentiable_through_z() -> None:
    torch.manual_seed(2)
    batch_size, dim = 4, 5
    posterior, prior, z = _make_posterior_and_prior(batch_size, dim)
    z.requires_grad_(True)
    x = torch.rand(batch_size, 3, 8, 8)
    reconstruction = torch.rand(batch_size, 3, 8, 8, requires_grad=True)

    terms = monte_carlo_elbo(x, reconstruction, z, posterior, prior)
    terms["elbo"].sum().backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
