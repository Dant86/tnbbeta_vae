"""Shared grid-plotting core for TNBBeta parameter-ablation figures.

Builds a grid of (row value) x (column value) cells, each showing a 3D S^2
heatmap of ``TNBBetaSpherical``'s log-density directly above the corresponding
``TNBBetaUnivariate`` density curve on (0, 1) -- used by ``pq_ablation`` (rows
p, columns q, epsilon fixed) and ``p_epsilon_ablation`` (rows p, columns
epsilon, q fixed) so the grid/layout logic only needs to live in one place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from tnbbeta_vae.distributions import TNBBetaSpherical, TNBBetaUnivariate
from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = [
    "DEFAULT_MEAN_DIRECTION",
    "build_grid_figure",
    "color_range",
    "sphere_log_density",
    "sphere_mesh",
    "univariate_density",
]

DEFAULT_MEAN_DIRECTION = torch.tensor([0.0, 0.0, 1.0])
_CURVE_COLOR = "#e7298a"


def build_grid_figure(
    row_values: Sequence[float],
    row_label: str,
    column_values: Sequence[float],
    column_label: str,
    params_for_cell: Callable[[float, float], tuple[float, float, float]],
    mean_direction: Tensor,
    title: str,
    resolution: int,
    curve_points: int,
) -> go.Figure:
    """Builds the full (sphere-over-curve) x (row value, column value) grid.

    Args:
        row_values: Values placed down the rows (e.g. p).
        row_label: Row parameter's name, for its "``label``=value" annotation.
        column_values: Values placed across the columns (e.g. q or epsilon).
        column_label: Column parameter's name, for its header.
        params_for_cell: Maps ``(row_value, column_value)`` to the ``(p, q,
            epsilon)`` triple to evaluate at that cell.
        mean_direction: Fixed mean direction shared by every cell.
        title: Figure title.
        resolution: Sphere mesh resolution (points per side).
        curve_points: Number of points along the univariate curve's (0, 1) grid.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    n_rows, n_cols = len(row_values), len(column_values)
    specs, row_heights = [], []
    for _ in range(n_rows):
        specs += [[{"type": "scene"}] * n_cols, [{"type": "xy"}] * n_cols]
        row_heights += [3.0, 1.0]

    figure = make_subplots(
        rows=2 * n_rows,
        cols=n_cols,
        specs=specs,
        row_heights=row_heights,
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        column_titles=[f"{column_label}={v:g}" for v in column_values],
    )

    mesh_x, mesh_y, mesh_z = sphere_mesh(resolution)
    points = torch.tensor(
        np.stack([mesh_x, mesh_y, mesh_z], axis=-1), dtype=torch.float32
    )
    curve_y = torch.linspace(0.005, 0.995, curve_points)

    for row_index, row_value in enumerate(row_values):
        sphere_row, curve_row = 2 * row_index + 1, 2 * row_index + 2
        for col_index, column_value in enumerate(column_values, start=1):
            p, q, epsilon = params_for_cell(row_value, column_value)
            log_density = sphere_log_density(points, mean_direction, p, q, epsilon)
            log_density = log_density.reshape(mesh_x.shape)
            cmin, cmax = color_range(log_density)
            figure.add_trace(
                go.Surface(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    surfacecolor=log_density,
                    colorscale="Inferno",
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False,
                ),
                row=sphere_row,
                col=col_index,
            )
            density = univariate_density(curve_y, p, q, epsilon)
            figure.add_trace(
                go.Scatter(
                    x=curve_y.numpy(),
                    y=density.numpy(),
                    mode="lines",
                    fill="tozeroy",
                    line={"color": _CURVE_COLOR, "width": 1.5},
                    showlegend=False,
                ),
                row=curve_row,
                col=col_index,
            )
            figure.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="cube",
                camera={"eye": {"x": 1.4, "y": 1.4, "z": 1.0}},
                row=sphere_row,
                col=col_index,
            )
            figure.update_xaxes(
                range=[0, 1],
                showticklabels=row_index == n_rows - 1,
                ticks="outside" if row_index == n_rows - 1 else "",
                tickvals=[0, 0.5, 1],
                row=curve_row,
                col=col_index,
            )
            figure.update_yaxes(
                showticklabels=False, ticks="", row=curve_row, col=col_index
            )
        _add_row_label(figure, f"{row_label}={row_value:g}", sphere_row, curve_row)

    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=title,
        showlegend=False,
        margin={"l": 50, "r": 10, "t": 60, "b": 30},
    )
    return figure


def color_range(log_density: Tensor) -> tuple[float, float]:
    """Returns a (cmin, cmax) color range, padded if ``log_density`` is near-constant.

    (p=0.5, q=0) at eps=1 on S^2 is exactly the uniform-sphere special case (see
    ``uniform_prior_params``): every point has the same density up to float32
    noise (~1e-6). Auto-scaling a colorscale to that tiny a range turns the noise
    into what looks like structured rings, misrepresenting a uniform density as a
    non-uniform one; padding the range out to a fixed minimum span instead keeps
    the cell reading as the flat, single color it actually is.
    """
    lo, hi = float(log_density.min()), float(log_density.max())
    min_span = 0.5
    if hi - lo < min_span:
        centre = (lo + hi) / 2
        return centre - min_span / 2, centre + min_span / 2
    return lo, hi


def sphere_mesh(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a unit-sphere (x, y, z) mesh, each of shape (resolution, resolution)."""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    return np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)


def sphere_log_density(
    points: Tensor, mean_direction: Tensor, p: float, q: float, epsilon: float
) -> Tensor:
    """Returns TNBBetaSpherical's log-density at a flattened batch of points."""
    dist = TNBBetaSpherical(mean_direction, p, q, epsilon)
    with torch.no_grad():
        return dist.log_prob(points)


def univariate_density(y: Tensor, p: float, q: float, epsilon: float) -> Tensor:
    """Returns TNBBetaUnivariate's density (not log-density) at ``y``."""
    dist = TNBBetaUnivariate(torch.tensor(p), torch.tensor(q), torch.tensor(epsilon))
    with torch.no_grad():
        return dist.log_prob(y).exp()


def _add_row_label(
    figure: go.Figure, text: str, sphere_row: int, curve_row: int
) -> None:
    """Adds a vertical "<label>=..." annotation spanning one row pair.

    ``make_subplots``' own ``row_titles`` labels every grid row, but each
    label here needs to span a *pair* of rows (one sphere row, one curve row)
    instead, so it's added directly from the pair's actual rendered domain.
    """
    sphere_scene = figure.get_subplot(sphere_row, 1)
    curve_axes = figure.get_subplot(curve_row, 1)
    top = sphere_scene.domain.y[1]  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]
    bottom = curve_axes.yaxis.domain[0]  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]
    figure.add_annotation(
        text=text,
        x=-0.03,
        y=(top + bottom) / 2,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font={"size": 12},
    )
