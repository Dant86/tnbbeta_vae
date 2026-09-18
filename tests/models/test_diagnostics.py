"""Tests for tnbbeta_vae.models.diagnostics."""

from __future__ import annotations

import math

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.diagnostics import (
    random_tangent_direction,
    sphere_geodesic_sweep,
    tnbbeta_spherical_posterior_diagnostics,
)


def test_diagnostics_contain_expected_keys_and_are_finite() -> None:
    posterior = _random_posterior(batch_size=6, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    expected_keys = {
        "posterior_p_mean",
        "posterior_p_min",
        "posterior_p_max",
        "posterior_q_mean",
        "posterior_q_min",
        "posterior_q_max",
        "posterior_epsilon_mean",
        "posterior_direction_pairwise_cosine_mean",
    }
    assert set(diagnostics) == expected_keys
    for value in diagnostics.values():
        assert value.dim() == 0
        assert torch.isfinite(value)


def test_min_mean_max_are_ordered() -> None:
    posterior = _random_posterior(batch_size=10, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_p_min"] <= diagnostics["posterior_p_mean"]
    assert diagnostics["posterior_p_mean"] <= diagnostics["posterior_p_max"]
    assert diagnostics["posterior_q_min"] <= diagnostics["posterior_q_mean"]
    assert diagnostics["posterior_q_mean"] <= diagnostics["posterior_q_max"]


def test_pairwise_cosine_key_absent_for_batch_size_one() -> None:
    posterior = _random_posterior(batch_size=1, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert "posterior_direction_pairwise_cosine_mean" not in diagnostics


def test_pairwise_cosine_is_one_for_a_collapsed_posterior() -> None:
    """A posterior collapsed onto a single direction should read cosine ~= 1."""
    mean_direction = torch.zeros(6, 4)
    mean_direction[:, 0] = 1.0
    posterior = TNBBetaSpherical(
        mean_direction, p=torch.full((6,), 0.01), q=torch.full((6,), 0.999), epsilon=1.0
    )

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_direction_pairwise_cosine_mean"] > 0.999


def test_pairwise_cosine_is_near_zero_for_diverse_directions() -> None:
    """Independent random directions in dim=4 should read cosine near 0, not 1."""
    torch.manual_seed(0)
    posterior = _random_posterior(batch_size=200, dim=4)

    diagnostics = tnbbeta_spherical_posterior_diagnostics(posterior)

    assert diagnostics["posterior_direction_pairwise_cosine_mean"].abs() < 0.2


def test_random_tangent_direction_is_orthogonal_and_unit_norm() -> None:
    torch.manual_seed(0)
    base_point = torch.randn(10, 5)
    base_point = base_point / base_point.norm(dim=-1, keepdim=True)

    tangent = random_tangent_direction(base_point)

    assert torch.allclose(tangent.norm(dim=-1), torch.ones(10), atol=1e-5)
    assert torch.allclose((tangent * base_point).sum(-1), torch.zeros(10), atol=1e-5)


def test_geodesic_sweep_stays_on_sphere() -> None:
    torch.manual_seed(1)
    base_point = torch.randn(5)
    base_point = base_point / base_point.norm()
    tangent = random_tangent_direction(base_point)
    angles = torch.linspace(-math.pi, math.pi, 9)

    points = sphere_geodesic_sweep(base_point, tangent, angles)

    assert points.shape == (9, 5)
    assert torch.allclose(points.norm(dim=-1), torch.ones(9), atol=1e-5)


def test_geodesic_sweep_endpoints_match_base_and_tangent() -> None:
    base_point = torch.randn(5)
    base_point = base_point / base_point.norm()
    tangent = random_tangent_direction(base_point)

    at_zero = sphere_geodesic_sweep(base_point, tangent, torch.tensor([0.0]))
    at_quarter_turn = sphere_geodesic_sweep(
        base_point, tangent, torch.tensor([math.pi / 2])
    )

    assert torch.allclose(at_zero[0], base_point, atol=1e-6)
    assert torch.allclose(at_quarter_turn[0], tangent, atol=1e-6)


def test_geodesic_sweep_angular_distance_matches_angle() -> None:
    """The whole point: moving by angle theta should be theta radians away."""
    torch.manual_seed(2)
    base_point = torch.randn(6)
    base_point = base_point / base_point.norm()
    tangent = random_tangent_direction(base_point)
    angles = torch.tensor([0.1, 0.5, 1.0, 2.0])

    points = sphere_geodesic_sweep(base_point, tangent, angles)
    cosine_to_base = (points * base_point).sum(-1).clamp(-1, 1)
    traveled = torch.acos(cosine_to_base)

    assert torch.allclose(traveled, angles, atol=1e-5)


def _random_posterior(batch_size: int, dim: int) -> TNBBetaSpherical:
    mean_direction = torch.randn(batch_size, dim)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    p = torch.rand(batch_size) * 0.4 + 0.3
    q = torch.rand(batch_size) * 0.4 + 0.3
    return TNBBetaSpherical(mean_direction, p=p, q=q, epsilon=1.0)
