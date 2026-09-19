"""Tests for tnbbeta_vae.models.losses.elbo."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo


def test_elbo_terms_are_finite_and_arithmetically_consistent() -> None:
    torch.manual_seed(0)
    batch_size, dim = 4, 5
    posterior, prior = _make_posterior_and_prior(batch_size, dim)
    x = torch.rand(batch_size, 3, 8, 8)
    captured_z: dict[str, torch.Tensor] = {}

    def decode(z: torch.Tensor) -> torch.Tensor:
        captured_z["z"] = z
        return torch.rand(batch_size, 3, 8, 8)

    terms = monte_carlo_elbo(x, posterior, prior, decode)

    for key in ("elbo", "log_likelihood", "kl"):
        assert terms[key].shape == (batch_size,)
        assert torch.isfinite(terms[key]).all()

    z = captured_z["z"]
    assert torch.allclose(terms["elbo"], terms["log_likelihood"] - terms["kl"])
    assert torch.allclose(
        terms["kl"], posterior.log_prob(z) - prior.log_prob(z), atol=1e-5
    )


def test_elbo_penalizes_reconstruction_error() -> None:
    """A perfect reconstruction should score a higher log_likelihood than a bad one."""
    torch.manual_seed(1)
    batch_size, dim = 4, 5
    posterior, prior = _make_posterior_and_prior(batch_size, dim)
    x = torch.rand(batch_size, 3, 8, 8)
    bad_reconstruction = torch.rand(batch_size, 3, 8, 8)

    perfect_terms = monte_carlo_elbo(x, posterior, prior, decoder=lambda _z: x)
    bad_terms = monte_carlo_elbo(
        x, posterior, prior, decoder=lambda _z: bad_reconstruction
    )

    assert (perfect_terms["log_likelihood"] > bad_terms["log_likelihood"]).all()


def test_elbo_is_differentiable_wrt_posterior_parameters() -> None:
    """Checks the real gradient path: posterior params -> rsample -> decoder -> elbo."""
    torch.manual_seed(2)
    batch_size, dim = 4, 5
    raw_direction = torch.randn(batch_size, dim, requires_grad=True)
    mean_direction = raw_direction / raw_direction.norm(dim=-1, keepdim=True)
    p = torch.full((batch_size,), 0.8, requires_grad=True)
    posterior = TNBBetaSpherical(mean_direction, p=p, q=0.7, epsilon=1.0)
    prior = _canonical_prior(dim)

    x = torch.rand(batch_size, 3, 8, 8)
    decoder = nn.Linear(dim, 3 * 8 * 8)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return decoder(z).reshape(batch_size, 3, 8, 8)

    terms = monte_carlo_elbo(x, posterior, prior, decode)
    terms["elbo"].sum().backward()

    assert raw_direction.grad is not None
    assert torch.isfinite(raw_direction.grad).all()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()


def test_num_samples_matches_manual_average() -> None:
    """num_samples=N must equal the mean of N independent single-sample calls."""
    batch_size, dim = 3, 4
    posterior, prior = _make_posterior_and_prior(batch_size, dim)
    x = torch.rand(batch_size, 3, 8, 8)
    decoder = nn.Linear(dim, 3 * 8 * 8)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return decoder(z).reshape(z.shape[0], 3, 8, 8)

    torch.manual_seed(5)
    multi_sample_terms = monte_carlo_elbo(x, posterior, prior, decode, num_samples=4)

    torch.manual_seed(5)
    single_sample_terms = [
        monte_carlo_elbo(x, posterior, prior, decode) for _ in range(4)
    ]
    manual_mean = {
        key: torch.stack([t[key] for t in single_sample_terms]).mean(0)
        for key in ("elbo", "log_likelihood", "kl")
    }

    for key, expected in manual_mean.items():
        assert torch.allclose(multi_sample_terms[key], expected, atol=1e-5)


def test_analytic_kl_matches_closed_form_gaussian_formula() -> None:
    """KL(N(mu, s^2) || N(0, 1)) = 0.5 * sum(mu^2 + s^2 - 1 - log s^2)."""
    torch.manual_seed(6)
    posterior, prior, mu, sigma = _gaussian_posterior_and_prior(batch_size=5, dim=4)
    x = torch.rand(5, 3, 8, 8)
    decoder = nn.Linear(4, 3 * 8 * 8)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return decoder(z).reshape(z.shape[0], 3, 8, 8)

    terms = monte_carlo_elbo(x, posterior, prior, decode, analytic_kl=True)

    expected = 0.5 * (mu**2 + sigma**2 - 1 - 2 * sigma.log()).sum(-1)
    assert torch.allclose(terms["kl"], expected, atol=1e-5)
    assert torch.allclose(terms["elbo"], terms["log_likelihood"] - terms["kl"])


def test_analytic_kl_is_deterministic_and_nonnegative() -> None:
    """Unlike the MC estimate, exact KL doesn't depend on which z was drawn."""
    posterior, prior, _, _ = _gaussian_posterior_and_prior(batch_size=6, dim=4)
    x = torch.rand(6, 3, 8, 8)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 3, 8, 8)

    torch.manual_seed(0)
    first = monte_carlo_elbo(x, posterior, prior, decode, analytic_kl=True)
    torch.manual_seed(1)
    second = monte_carlo_elbo(x, posterior, prior, decode, analytic_kl=True)

    assert torch.equal(first["kl"], second["kl"])
    assert (first["kl"] >= 0).all()


def test_analytic_kl_agrees_with_monte_carlo_kl_in_expectation() -> None:
    """Sanity check that the two KL paths estimate the same quantity."""
    torch.manual_seed(7)
    posterior, prior, _, _ = _gaussian_posterior_and_prior(batch_size=3, dim=4)
    x = torch.rand(3, 3, 8, 8)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 3, 8, 8)

    exact = monte_carlo_elbo(x, posterior, prior, decode, analytic_kl=True)["kl"]
    mc = monte_carlo_elbo(x, posterior, prior, decode, num_samples=20_000)["kl"]

    assert torch.allclose(exact, mc, atol=0.05)


def test_analytic_kl_raises_for_pairs_without_a_closed_form() -> None:
    """TNBBetaSpherical has no registered KL -- fail loudly, don't silently MC."""
    posterior, prior = _make_posterior_and_prior(batch_size=3, dim=4)
    x = torch.rand(3, 3, 8, 8)

    with pytest.raises(NotImplementedError):
        monte_carlo_elbo(
            x,
            posterior,
            prior,
            decoder=lambda z: torch.zeros(z.shape[0], 3, 8, 8),
            analytic_kl=True,
        )


def _gaussian_posterior_and_prior(
    batch_size: int, dim: int
) -> tuple[Independent, Independent, torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch_size, dim)
    sigma = torch.rand(batch_size, dim) * 1.5 + 0.2
    posterior = Independent(Normal(mu, sigma), 1)
    prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1)
    return posterior, prior, mu, sigma


def _canonical_prior(dim: int) -> TNBBetaSpherical:
    prior_direction = torch.zeros(dim)
    prior_direction[0] = 1.0
    return TNBBetaSpherical(prior_direction, p=0.9, q=0.8, epsilon=1.0)


def _make_posterior_and_prior(
    batch_size: int, dim: int
) -> tuple[TNBBetaSpherical, TNBBetaSpherical]:
    mean_direction = torch.randn(batch_size, dim)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    posterior = TNBBetaSpherical(
        mean_direction, p=torch.full((batch_size,), 0.8), q=0.7, epsilon=1.0
    )
    return posterior, _canonical_prior(dim)
