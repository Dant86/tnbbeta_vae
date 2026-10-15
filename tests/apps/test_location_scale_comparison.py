"""Tests for apps.distributions.location_scale_comparison."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from apps.distributions import location_scale_comparison as lsc
from tnbbeta_vae.distributions import VonMisesFisher


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "location_scale_comparison.png"

    lsc.main(
        [
            "--output",
            str(output),
            "--kappa",
            "4",
            "--resolution",
            "6",
            "--curve-points",
            "20",
        ]
    )

    assert output.exists() and output.stat().st_size > 0


def test_points_at_land_on_the_sphere_with_the_given_cosine_similarity() -> None:
    mean_direction = torch.tensor([1.0, 1.0, 1.0]) / 3**0.5
    t = torch.tensor([-0.8, 0.0, 0.3, 0.95])

    points = lsc._points_at(t, mean_direction)

    assert torch.allclose(points.norm(dim=-1), torch.ones_like(t), atol=1e-5)
    assert torch.allclose(points @ mean_direction, t, atol=1e-5)


def test_log_sigma_t_matches_the_known_s2_constant() -> None:
    """On S^2 (ambient_dim=3), t is exactly uniform on [-1, 1] (Archimedes'
    hat-box theorem), so sigma_t is the constant 1/2 for every t."""
    t = torch.tensor([-0.9, -0.1, 0.4, 0.99])

    log_sigma = lsc._log_sigma_t(t, ambient_dim=3)

    assert torch.allclose(log_sigma, torch.full_like(t, math.log(0.5)))


def test_higher_kappa_concentrates_more_mass_near_t_equals_one() -> None:
    mean_direction = torch.tensor([1.0, 1.0, 1.0]) / 3**0.5
    t = torch.tensor([0.9])
    points = lsc._points_at(t, mean_direction)

    low = VonMisesFisher(mean_direction.unsqueeze(0), torch.tensor([[2.0]]))
    high = VonMisesFisher(mean_direction.unsqueeze(0), torch.tensor([[20.0]]))

    with torch.no_grad():
        low_density = (low.log_prob(points) + lsc._log_sigma_t(t, 3)).exp()
        high_density = (high.log_prob(points) + lsc._log_sigma_t(t, 3)).exp()

    assert high_density.item() > low_density.item()
