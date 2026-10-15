"""Tests for apps.distributions.tangent_normal_decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apps.distributions import tangent_normal_decomposition as tnd


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "tangent_normal_decomposition.png"

    tnd.main(["--output", str(output), "--t", "0.5", "--phi", "0.4"])

    assert output.exists() and output.stat().st_size > 0


def test_decomposition_reconstructs_x_on_the_unit_sphere() -> None:
    """x = t*mu + sqrt(1-t^2)*xi must itself land back on the unit sphere, with
    mu^T x == t exactly -- the identity the figure is illustrating."""
    t, phi = 0.3, 1.1
    mu = np.array([0.0, 0.0, 1.0])
    xi = np.array([np.cos(phi), np.sin(phi), 0.0])
    normal_weight = np.sqrt(1.0 - t**2)
    x = t * mu + normal_weight * xi

    assert np.isclose(np.linalg.norm(x), 1.0)
    assert np.isclose(mu @ x, t)
    assert np.isclose(np.linalg.norm(xi), 1.0)
    assert np.isclose(mu @ xi, 0.0)


def test_build_figure_runs_at_a_boundary_t() -> None:
    """t near +-1 (mu and x nearly coincide/oppose) must not divide by zero."""
    figure = tnd.build_figure(t=0.999, phi=0.0)

    assert figure.data  # built without raising
