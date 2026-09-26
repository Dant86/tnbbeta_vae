"""Tests for apps.distributions.pq_ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from apps.distributions import pq_ablation


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "pq_ablation.png"

    pq_ablation.main(
        ["--output", str(output), "--resolution", "6", "--curve-points", "20"]
    )

    assert output.exists() and output.stat().st_size > 0


def test_figure_has_one_surface_and_one_curve_per_pq_cell() -> None:
    figure = pq_ablation._figure(resolution=6, curve_points=20)

    n_cells = len(pq_ablation._P_VALUES) * len(pq_ablation._Q_VALUES)
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
    assert len(surfaces) == n_cells
    assert len(curves) == n_cells


def test_color_range_pads_a_near_constant_cell() -> None:
    constant = torch.full((5, 5), -2.531)

    lo, hi = pq_ablation._color_range(constant)

    assert hi - lo >= 0.5
    assert lo < -2.531 < hi


def test_color_range_leaves_a_wide_range_untouched() -> None:
    varied = torch.tensor([-5.0, 0.0, 5.0])

    lo, hi = pq_ablation._color_range(varied)

    assert (lo, hi) == (-5.0, 5.0)


def test_uniform_special_case_is_constant_to_float_precision() -> None:
    # p=0.5, q=0, eps=1 on S^2 is exactly the uniform-sphere special case
    # (uniform_prior_params); this is what motivated _color_range's padding.
    mesh_x, mesh_y, mesh_z = pq_ablation._sphere_mesh(resolution=10)
    points = torch.tensor(
        np.stack([mesh_x, mesh_y, mesh_z], axis=-1), dtype=torch.float32
    )

    log_density = pq_ablation._sphere_log_density(points, p=0.5, q=0.0)

    assert (log_density.max() - log_density.min()).item() < 1e-4
