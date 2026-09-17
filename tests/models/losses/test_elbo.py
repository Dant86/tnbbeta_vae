"""Tests for tnbbeta_vae.models.losses.elbo."""

from __future__ import annotations

import torch
from torch import nn

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
