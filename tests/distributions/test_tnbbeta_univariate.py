"""Tests for tnbbeta_vae.distributions.tnbbeta_univariate."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Beta, NegativeBinomial

from tnbbeta_vae.distributions import TNBBetaUnivariate

# Truncation for the c/a/b sums in _log_prob_via_auxiliary_construction. Large
# enough that truncation error is ~1e-6 or smaller for every case in
# test_log_prob_matches_auxiliary_construction (checked empirically).
_AUXILIARY_CONSTRUCTION_N_MAX = 150


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


@pytest.mark.parametrize(
    ("p", "q", "epsilon", "y"),
    [
        (0.3, 0.1, 0.5, 0.3),
        (0.5, 0.4, 1.0, 0.5),
        (0.7, 0.7, 2.0, 0.8),
        (0.2, 0.6, 1.0, 0.2),
        (0.5, 0.0, 1.0, 0.5),
        (0.4, 0.3, 0.7, 0.6),
    ],
)
def test_log_prob_matches_auxiliary_construction(
    p: float, q: float, epsilon: float, y: float
) -> None:
    """log_prob (Eq. 13) should agree with Theorem 4.1's construction (Eq. 17).

    These are two independent characterizations of the same distribution:
    Eq. 13 is the closed-form density log_prob implements directly, while
    Eq. 17 defines Y as the marginal of a beta distribution randomized by
    three auxiliary negative-binomial draws. Agreement here isn't implied
    by any single-path test (e.g. test_log_prob_integrates_to_one) -- it
    specifically catches a closed-form density that's self-consistent but
    doesn't match the paper's actual construction.
    """
    dist = TNBBetaUnivariate(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        epsilon=torch.tensor(epsilon, dtype=torch.float64),
    )

    closed_form = dist.log_prob(torch.tensor(y, dtype=torch.float64))
    via_construction = _log_prob_via_auxiliary_construction(y, p, q, epsilon)

    assert torch.abs(closed_form - via_construction) < 1e-3


def test_rsample_matches_analytic_mean() -> None:
    """Cross-checks the rsample() transform against log_prob via numerical integration.

    This independently validates the rsample() derivation: if the
    transform were wrong (e.g. sign error, wrong q-tilt), the empirical
    mean of its draws would drift from the mean implied by log_prob,
    which is validated separately (test_log_prob_integrates_to_one).
    """
    torch.manual_seed(2)
    p, q, epsilon = 0.35, 0.8, 1.0
    dist = TNBBetaUnivariate(p=p, q=q, epsilon=epsilon)

    y = torch.linspace(1e-4, 1 - 1e-4, 200_000)
    analytic_mean = torch.trapz(y * dist.log_prob(y).exp(), y)

    samples = dist.rsample((100_000,))

    assert torch.abs(samples.mean() - analytic_mean) < 0.01


def test_rsample_is_differentiable_wrt_all_parameters() -> None:
    """rsample() must be reparameterized: gradients should flow to p, q, epsilon."""
    torch.manual_seed(3)
    p = torch.tensor(0.4, requires_grad=True)
    q = torch.tensor(0.6, requires_grad=True)
    epsilon = torch.tensor(1.2, requires_grad=True)
    dist = TNBBetaUnivariate(p=p, q=q, epsilon=epsilon)

    y = dist.rsample((1_000,))
    y.sum().backward()

    assert dist.has_rsample
    for param, grad in ((p, p.grad), (q, q.grad), (epsilon, epsilon.grad)):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, param


def _log_prob_via_auxiliary_construction(
    y: float,
    p: float,
    q: float,
    epsilon: float,
    n_max: int = _AUXILIARY_CONSTRUCTION_N_MAX,
) -> torch.Tensor:
    """Computes log TNBbeta(y; p, q, eps) from Theorem 4.1's construction (Eq. 17).

    This is a second, independent characterization of the same density as
    log_prob's Eq. 13: C ~ NB(eps, 1-q), A|C ~ NB(eps+C, 1-p),
    B|C ~ NB(eps+C, p), Y|A,B,C ~ Beta(eps+C+A, eps+C+B). The marginal
    log-density is a triple sum over the (discrete) auxiliary variables,
    truncated here to `n_max` per variable and evaluated via logsumexp.
    """
    dtype = torch.float64
    y_t = torch.as_tensor(y, dtype=dtype)
    p_t = torch.as_tensor(p, dtype=dtype)
    q_t = torch.as_tensor(q, dtype=dtype)
    eps_t = torch.as_tensor(epsilon, dtype=dtype)

    counts = torch.arange(n_max + 1, dtype=dtype)
    c = counts.view(-1, 1, 1)
    a = counts.view(1, -1, 1)
    b = counts.view(1, 1, -1)

    log_p_c = NegativeBinomial(total_count=eps_t, probs=q_t).log_prob(c)
    log_p_a = NegativeBinomial(total_count=eps_t + c, probs=p_t).log_prob(a)
    log_p_b = NegativeBinomial(total_count=eps_t + c, probs=1 - p_t).log_prob(b)
    log_p_y = Beta(eps_t + c + a, eps_t + c + b).log_prob(y_t)

    joint_log_prob = log_p_c + log_p_a + log_p_b + log_p_y
    return torch.logsumexp(joint_log_prob.reshape(-1), dim=0)
