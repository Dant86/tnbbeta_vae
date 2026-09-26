"""Tests for apps.distributions.ablation_grid (shared by pq_ablation and
p_epsilon_ablation)."""

from __future__ import annotations

import numpy as np
import torch

from apps.distributions import ablation_grid


def test_build_grid_figure_has_one_surface_and_one_curve_per_cell() -> None:
    figure = ablation_grid.build_grid_figure(
        row_values=[0.2, 0.8],
        row_label="p",
        column_values=[0.0, 0.5],
        column_label="q",
        params_for_cell=lambda p, q: (p, q, 1.0),
        mean_direction=ablation_grid.DEFAULT_MEAN_DIRECTION,
        title="test grid",
        resolution=6,
        curve_points=20,
    )

    surfaces = [
        t
        for t in figure.data
        if t.type == "surface"  # pyright: ignore[reportAttributeAccessIssue]
    ]
    curves = [
        t
        for t in figure.data
        if t.type == "scatter"  # pyright: ignore[reportAttributeAccessIssue]
    ]
    assert len(surfaces) == 4  # 2 rows x 2 columns
    assert len(curves) == 4


def test_color_range_pads_a_near_constant_cell() -> None:
    constant = torch.full((5, 5), -2.531)

    lo, hi = ablation_grid.color_range(constant)

    assert hi - lo >= 0.5
    assert lo < -2.531 < hi


def test_color_range_leaves_a_wide_range_untouched() -> None:
    varied = torch.tensor([-5.0, 0.0, 5.0])

    lo, hi = ablation_grid.color_range(varied)

    assert (lo, hi) == (-5.0, 5.0)


def test_uniform_special_case_is_constant_to_float_precision() -> None:
    # p=0.5, q=0, eps=1 on S^2 is exactly the uniform-sphere special case
    # (uniform_prior_params); this is what motivated color_range's padding.
    mesh_x, mesh_y, mesh_z = ablation_grid.sphere_mesh(resolution=10)
    points = torch.tensor(
        np.stack([mesh_x, mesh_y, mesh_z], axis=-1), dtype=torch.float32
    )

    log_density = ablation_grid.sphere_log_density(
        points, ablation_grid.DEFAULT_MEAN_DIRECTION, p=0.5, q=0.0, epsilon=1.0
    )

    assert (log_density.max() - log_density.min()).item() < 1e-4


def test_univariate_density_is_symmetric_at_p_half() -> None:
    y = torch.linspace(0.01, 0.99, 5)

    density = ablation_grid.univariate_density(y, p=0.5, q=0.7, epsilon=1.0)
    density_reflected = ablation_grid.univariate_density(
        1 - y, p=0.5, q=0.7, epsilon=1.0
    )

    assert torch.allclose(density, density_reflected, atol=1e-5)
