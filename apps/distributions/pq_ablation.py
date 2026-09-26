"""TNBBeta p x q ablation grid: S^2 heatmaps with univariate densities below.

Usage:
    uv run python -m apps.distributions.pq_ablation [--output PATH] \\
        [--resolution 60] [--curve-points 400]

Ported from a matplotlib prototype in the sibling ``hypergen`` project, rebuilt
with plotly (this project's plotting library, styled via the shared
``tnbbeta_vae.plotting`` theme) and this project's own validated
``TNBBetaSpherical``/``TNBBetaUnivariate`` rather than reimplementing the
spherical density: the sphere heatmap uses ``TNBBetaSpherical.log_prob``
directly (including its Jacobian and surface-area normalization terms), not
just the univariate log-density evaluated at a cosine-similarity-derived
latitude, which is not a properly normalized spherical density on its own.

For each (p, q) in a grid (epsilon and the mean direction held fixed), plots:

* a 3D S^2 heatmap of ``TNBBetaSpherical``'s log-density (each sphere is
  colored on its own scale -- density magnitude varies by orders of magnitude
  across the grid, so a single shared color scale would wash out most cells);
* the corresponding ``TNBBetaUnivariate`` density curve on (0, 1) directly
  below it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from tnbbeta_vae.distributions import TNBBetaSpherical, TNBBetaUnivariate
from tnbbeta_vae.plotting import TEMPLATE_NAME

_MEAN_DIRECTION = torch.tensor([0.0, 0.0, 1.0])
_EPSILON = 1.0
_P_VALUES = [0.05, 0.2, 0.5, 0.8, 0.95]
_Q_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
_CURVE_COLOR = "#e7298a"


def main(argv: list[str] | None = None) -> None:
    """Builds the p x q ablation grid and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="pq_ablation.png")
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--curve-points", type=int, default=400)
    args = parser.parse_args(argv)

    figure = _figure(args.resolution, args.curve_points)
    figure.write_image(
        args.output,
        width=210 * len(_Q_VALUES),
        height=230 * len(_P_VALUES),
        scale=2,
    )
    print(f"Wrote {args.output}")


def _figure(resolution: int, curve_points: int) -> go.Figure:
    """Builds the full (sphere-over-curve) x (p, q) grid figure."""
    n_rows, n_cols = len(_P_VALUES), len(_Q_VALUES)
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
        column_titles=[f"q={q:g}" for q in _Q_VALUES],
    )

    mesh_x, mesh_y, mesh_z = _sphere_mesh(resolution)
    points = torch.tensor(
        np.stack([mesh_x, mesh_y, mesh_z], axis=-1), dtype=torch.float32
    )
    curve_y = torch.linspace(0.005, 0.995, curve_points)

    for row_index, p in enumerate(_P_VALUES):
        sphere_row, curve_row = 2 * row_index + 1, 2 * row_index + 2
        for col_index, q in enumerate(_Q_VALUES, start=1):
            log_density = _sphere_log_density(points, p, q).reshape(mesh_x.shape)
            cmin, cmax = _color_range(log_density)
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
            density = _univariate_density(curve_y, p, q)
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
        _add_row_label(figure, f"p={p:g}", sphere_row, curve_row)

    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=(
            f"TNBBeta p x q ablation on S^2 "
            f"(epsilon={_EPSILON:g}, mu={_MEAN_DIRECTION.tolist()})"
        ),
        showlegend=False,
        margin={"l": 50, "r": 10, "t": 60, "b": 30},
    )
    return figure


def _add_row_label(
    figure: go.Figure, text: str, sphere_row: int, curve_row: int
) -> None:
    """Adds a vertical "p=..." label spanning one sphere-and-curve row pair.

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


def _color_range(log_density: Tensor) -> tuple[float, float]:
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


def _sphere_mesh(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a unit-sphere (x, y, z) mesh, each of shape (resolution, resolution)."""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    return np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)


def _sphere_log_density(points: Tensor, p: float, q: float) -> Tensor:
    """Returns TNBBetaSpherical's log-density at a flattened batch of points."""
    dist = TNBBetaSpherical(_MEAN_DIRECTION, p, q, _EPSILON)
    with torch.no_grad():
        return dist.log_prob(points)


def _univariate_density(y: Tensor, p: float, q: float) -> Tensor:
    """Returns TNBBetaUnivariate's density (not log-density) at ``y``."""
    dist = TNBBetaUnivariate(torch.tensor(p), torch.tensor(q), torch.tensor(_EPSILON))
    with torch.no_grad():
        return dist.log_prob(y).exp()


if __name__ == "__main__":
    main(sys.argv[1:])
