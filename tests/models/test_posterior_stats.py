"""Tests for tnbbeta_vae.models.posterior_stats."""

from __future__ import annotations

import math

import pytest
import torch

from tnbbeta_vae.distributions import (
    PowerSpherical,
    TNBBetaSpherical,
    VonMisesFisher,
)
from tnbbeta_vae.models.posterior_stats import posterior_stats


def _normalized(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True)


def test_posterior_stats_power_spherical_entropy_matches_direct_computation() -> None:
    """Regression test: a previous version wrapped posterior in
    Independent(posterior, 1) before calling entropy(), which reinterprets the
    per-node batch dimension into the event and collapses entropy() to a 0-d scalar,
    crashing on entropy[mask] ("too many indices for tensor of dimension 0"). Power
    Spherical has a real entropy() (unlike TNBBetaSpherical), so it hit this directly.
    """
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    posterior = PowerSpherical(mean_direction, kappa)
    mask = torch.tensor([True, True, False, False, False])

    stats = posterior_stats(posterior, mask, {"family": "power_spherical"})

    expected = posterior.entropy()[mask].mean().item()
    assert math.isfinite(stats["entropy_mean"])
    assert stats["entropy_mean"] == pytest.approx(expected)
    assert stats["num_nodes"] == 2


def test_posterior_stats_von_mises_fisher_entropy_matches_direct_computation() -> None:
    """Same regression as above, for vMF -- it also implements entropy() directly."""
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.tensor([False, True, True, False, False])

    stats = posterior_stats(posterior, mask, {"family": "vmf"})

    expected = posterior.entropy()[mask].mean().item()
    assert math.isfinite(stats["entropy_mean"])
    assert stats["entropy_mean"] == pytest.approx(expected)
    assert stats["num_nodes"] == 2


def test_posterior_stats_von_mises_fisher_r_bar_does_not_crash() -> None:
    """Regression test: VonMisesFisher.rsample only accepts torch.Size or a bare int
    (unlike TNBBetaSpherical/PowerSpherical, which accept a plain tuple too) -- a
    previous version called rsample((100,)), a plain tuple, which VonMisesFisher
    mishandles as a single non-int "size" and raises TypeError at torch.Size
    construction, uncaught by posterior_stats's (AttributeError, RuntimeError)
    except clause."""
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.full((5, 1), 3.0)
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.ones(5, dtype=torch.bool)

    stats = posterior_stats(posterior, mask, {"family": "vmf"})

    assert math.isfinite(stats["r_bar"])
    assert stats["r_bar"] >= 0.0


def test_posterior_stats_tnbbeta_spherical_entropy_is_nan_not_a_crash() -> None:
    """TNBBetaSpherical has no closed-form entropy -- confirms this stays a graceful
    NaN (the except branch) after the fix, not a regression."""
    mean_direction = _normalized(torch.randn(5, 3))
    p = torch.full((5,), 0.7)
    q = torch.full((5,), 0.1)
    epsilon = torch.full((5,), 1.0)
    posterior = TNBBetaSpherical(mean_direction, p, q, epsilon)
    mask = torch.ones(5, dtype=torch.bool)

    stats = posterior_stats(posterior, mask, {"family": "tnbbeta", "latent_dim": 3})

    assert math.isnan(stats["entropy_mean"])


def test_posterior_stats_r_bar_is_per_node_not_cross_node_alignment() -> None:
    """Regression test for the averaging-order bug: a previous version averaged
    sample means across nodes before taking one norm, so two tightly concentrated
    nodes pointed in OPPOSITE directions cancelled to r_bar~0 -- reporting "diffuse"
    for a subset that is actually two sharply concentrated populations. The fix
    norms each node's own sample mean first, then averages those norms, so this
    case correctly reports high (not low) concentration.
    """
    mean_direction = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    kappa = torch.full((2, 1), 500.0)  # sharply concentrated at each node
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.ones(2, dtype=torch.bool)

    stats = posterior_stats(posterior, mask, {"family": "vmf"})

    assert stats["r_bar"] > 0.9  # each node is individually concentrated


def test_posterior_stats_tnbbeta_reports_parameter_marginals_and_bimodal_fraction() -> (
    None
):
    """TNBBeta's own (p, q, epsilon) and the fraction of nodes past the proven
    bimodal threshold (m = epsilon - (latent_dim - 1) / 2 < 0, expressivity
    write-up's Theorem 5.1/Corollary 5.3) are reported -- the one diagnostic that
    directly tests whether a subset of nodes is actually using TNBBeta's extra
    capacity, not just a proxy for it.
    """
    mean_direction = _normalized(torch.randn(4, 3))  # latent_dim 3 -> threshold 1.0
    p = torch.tensor([0.6, 0.7, 0.8, 0.9])
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    epsilon = torch.tensor([0.5, 0.5, 2.0, 2.0])  # first two bimodal, last two not
    posterior = TNBBetaSpherical(mean_direction, p, q, epsilon)
    mask = torch.ones(4, dtype=torch.bool)

    stats = posterior_stats(posterior, mask, {"family": "tnbbeta", "latent_dim": 3})

    assert stats["p_mean"] == pytest.approx(0.75)
    assert stats["epsilon_mean"] == pytest.approx(1.25)
    assert stats["m_mean"] == pytest.approx(1.25 - 1.0)
    assert stats["frac_bimodal"] == pytest.approx(0.5)


def test_posterior_stats_omits_tnbbeta_fields_for_other_families() -> None:
    """vMF/Power Spherical posteriors have no (p, q, epsilon) -- confirms the
    TNBBeta-only fields are simply absent, not NaN-filled, for other families."""
    mean_direction = _normalized(torch.randn(3, 3))
    kappa = torch.full((3, 1), 2.0)
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.ones(3, dtype=torch.bool)

    stats = posterior_stats(posterior, mask, {"family": "vmf"})

    assert "p_mean" not in stats
    assert "frac_bimodal" not in stats
