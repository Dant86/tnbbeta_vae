"""Tests for tnbbeta_vae.distributions.tnbbeta_univariate."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import TNBBetaUnivariate


def test_median_matches_p_parameter() -> None:
    """Theorem 3.1: median(Y) = p exactly, so the sample median should too."""
    torch.manual_seed(0)
    dist = TNBBetaUnivariate(p=0.3, q=0.7, epsilon=1.0)

    samples = dist.sample((20_000,))

    assert torch.abs(dist.median - 0.3) < 1e-6
    assert torch.abs(samples.median() - 0.3) < 0.02


def test_samples_are_in_unit_interval() -> None:
    """All draws must lie in the open unit interval."""
    torch.manual_seed(1)
    dist = TNBBetaUnivariate(
        p=torch.tensor([0.1, 0.5, 0.9]),
        q=torch.tensor([0.2, 0.5, 0.8]),
        epsilon=torch.tensor([0.5, 1.0, 2.0]),
    )

    samples = dist.sample((1_000,))

    assert samples.shape == (1_000, 3)
    assert torch.all(samples > 0.0)
    assert torch.all(samples < 1.0)


def test_log_prob_integrates_to_one() -> None:
    """The density should integrate to ~1 over (0, 1) (numerically, via a grid).

    Uses epsilon=1 and epsilon=2 (finite/vanishing boundary regimes, per
    Proposition 3.1) since epsilon < 1 gives boundary divergences that a
    plain trapezoidal grid can't resolve.
    """
    y = torch.linspace(1e-4, 1 - 1e-4, 200_000)

    for epsilon in (1.0, 2.0):
        dist = TNBBetaUnivariate(p=0.4, q=0.6, epsilon=epsilon)
        density = dist.log_prob(y).exp()
        integral = torch.trapz(density, y)
        assert torch.abs(integral - 1.0) < 0.02, f"epsilon={epsilon}"


def test_log_prob_is_maximized_near_median() -> None:
    """gamma(y, p) is maximized at y=p, so density should peak near p."""
    dist = TNBBetaUnivariate(p=0.5, q=0.9, epsilon=1.0)
    y = torch.linspace(0.01, 0.99, 1_000)

    log_probs = dist.log_prob(y)
    peak = y[torch.argmax(log_probs)]

    assert torch.abs(peak - 0.5) < 0.02


def test_broadcasts_batch_shape() -> None:
    """Batched parameters produce a matching batch_shape."""
    dist = TNBBetaUnivariate(
        p=torch.tensor([0.2, 0.5]), q=torch.tensor([0.3, 0.6]), epsilon=1.0
    )

    assert dist.batch_shape == torch.Size([2])
    assert dist.log_prob(torch.tensor([0.2, 0.5])).shape == torch.Size([2])
