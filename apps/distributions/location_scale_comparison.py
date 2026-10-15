"""Side-by-side von Mises-Fisher vs. Power Spherical comparison: S^2 log-density
heatmaps with their univariate angular densities below, same two-row-per-cell
styling as the TNBBeta p x q ablation grid (``apps.distributions.pq_ablation``).

Usage:
    uv run python -m apps.distributions.location_scale_comparison \\
        [--output PATH] [--kappa 50.0] [--resolution 60] [--curve-points 400]

Illustrates the paper's closing remark in section 2
(``sections/2_unit_interval_to_hypersphere.tex``) that von Mises-Fisher and Power
Spherical are both "location-scale" distributions on the hypersphere: both panels
share the same mean direction and the same concentration ``kappa``, so the only
visible difference is each family's own angular kernel (``exp(kappa*t)`` for vMF,
``(1+t)^kappa`` for Power Spherical). The mean direction is deliberately not a
coordinate axis -- a heatmap centred on ``e_1`` or ``e_3`` could be mistaken for a
special, axis-aligned case, when every point on the sphere is as good as any other.

Each panel's univariate angular density is computed directly from Lemma 2.1's own
relation, ``g(t;Theta) = f(x;mu,Theta) * sigma_t(t)`` -- the already-implemented
``VonMisesFisher``/``PowerSpherical`` ``log_prob`` evaluated at a point with that
cosine similarity, times the geometric correction ``sigma_t`` -- rather than
re-deriving each family's closed-form marginal by hand a second time.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from apps.distributions.ablation_grid import sphere_mesh
from tnbbeta_vae.distributions import PowerSpherical, VonMisesFisher
from tnbbeta_vae.plotting import TEMPLATE_NAME

# Deliberately not e_1, and offset ~72 degrees from the camera's own viewing
# direction (eye=(1.4, 1.4, 1.0) below), so a concentrated cap is actually
# visible against a clearly colder background instead of the camera staring
# straight down the mean direction and seeing only a smooth gradient.
_MEAN_DIRECTION = torch.tensor([0.1, -0.3, 0.95])
_MEAN_DIRECTION = _MEAN_DIRECTION / _MEAN_DIRECTION.norm()
_CURVE_COLOR = "#e7298a"
_PANELS = ("vMF", "Power-Spherical")

__all__ = ["build_figure", "main"]


def main(argv: list[str] | None = None) -> None:
    """Builds the vMF-vs-Power-Spherical comparison figure and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="location_scale_comparison.png")
    parser.add_argument("--kappa", type=float, default=50.0)
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--curve-points", type=int, default=400)
    args = parser.parse_args(argv)

    figure = build_figure(args.kappa, args.resolution, args.curve_points)
    figure.write_image(args.output, width=900, height=560, scale=2)
    print(f"Wrote {args.output}")


def build_figure(kappa: float, resolution: int, curve_points: int) -> go.Figure:
    """Builds the two-column (sphere-over-curve) comparison figure.

    Args:
        kappa: Shared concentration, used for both distributions.
        resolution: Sphere mesh resolution (points per side).
        curve_points: Number of points along the angular density curve's
            ``(-1, 1)`` grid.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    mesh_x, mesh_y, mesh_z = sphere_mesh(resolution)
    mesh_points = torch.tensor(
        np.stack([mesh_x, mesh_y, mesh_z], axis=-1), dtype=torch.float32
    )
    ambient_dim = _MEAN_DIRECTION.shape[-1]
    t_curve = torch.linspace(-0.995, 0.995, curve_points)
    curve_points_x = _points_at(t_curve, _MEAN_DIRECTION)

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}] * 2, [{"type": "xy"}] * 2],
        row_heights=[3.0, 1.0],
        vertical_spacing=0.03,
        horizontal_spacing=0.04,
        column_titles=list(_PANELS),
    )

    # VonMisesFisher needs an explicit leading batch dim (its _log_normalization
    # does `output.view(*output.shape[:-1])`, which breaks for an unbatched,
    # single-example loc/scale -- see tests/distributions/test_von_mises_fisher.py
    # for the same convention).
    distributions: tuple[VonMisesFisher, PowerSpherical] = (
        VonMisesFisher(_MEAN_DIRECTION.unsqueeze(0), torch.tensor([[kappa]])),
        PowerSpherical(_MEAN_DIRECTION, kappa),
    )
    for col, dist in enumerate(distributions, start=1):
        with torch.no_grad():
            mesh_log_density = dist.log_prob(mesh_points).reshape(mesh_x.shape)
        cmin, cmax = _color_range(mesh_log_density)
        figure.add_trace(
            go.Surface(
                x=mesh_x,
                y=mesh_y,
                z=mesh_z,
                surfacecolor=mesh_log_density.numpy(),
                colorscale="Inferno",
                cmin=cmin,
                cmax=cmax,
                showscale=False,
            ),
            row=1,
            col=col,
        )

        with torch.no_grad():
            curve_log_f = dist.log_prob(curve_points_x)
        angular_density = (curve_log_f + _log_sigma_t(t_curve, ambient_dim)).exp()
        figure.add_trace(
            go.Scatter(
                x=t_curve.numpy(),
                y=angular_density.numpy(),
                mode="lines",
                fill="tozeroy",
                line={"color": _CURVE_COLOR, "width": 1.5},
                showlegend=False,
            ),
            row=2,
            col=col,
        )

        figure.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="cube",
            camera={"eye": {"x": 1.4, "y": 1.4, "z": 1.0}},
            row=1,
            col=col,
        )
        figure.update_xaxes(range=[-1, 1], tickvals=[-1, 0, 1], row=2, col=col)
        figure.update_yaxes(showticklabels=False, ticks="", row=2, col=col)

    mu = ", ".join(f"{v:.2f}" for v in _MEAN_DIRECTION.tolist())
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=f"vMF vs. Power-Spherical (kappa={kappa:g}, mu=[{mu}])",
        showlegend=False,
        margin={"l": 20, "r": 10, "t": 70, "b": 30},
    )
    return figure


def _color_range(log_density: Tensor) -> tuple[float, float]:
    """Returns a (cmin, cmax) color range clipped to the 1st-99th percentile.

    Both families' log-density over the *whole* sphere mesh is heavily
    left-skewed: it is relatively flat across most of the mu-facing
    hemisphere and only plunges steeply in a thin sliver right at the
    antipode (most pronounced for Power Spherical's ``(1 + t)^kappa``
    kernel, whose log is nearly linear near ``t=1`` but diverges as
    ``t -> -1``). Unlike :func:`apps.distributions.ablation_grid.color_range`
    (which pads a near-constant range), scaling the color axis to the raw
    min/max here buries nearly all of the *visible*, camera-facing
    hemisphere's real variation under that one occluded tail, washing the
    cap out to a single flat color. Clipping to the 1st/99th percentile
    keeps that extreme tail from dominating the scale.

    Args:
        log_density: Log-density values over the full sphere mesh.

    Returns:
        ``(cmin, cmax)`` for the colorscale.
    """
    values = log_density.flatten().numpy()
    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    min_span = 0.5
    if hi - lo < min_span:
        centre = (lo + hi) / 2
        return centre - min_span / 2, centre + min_span / 2
    return lo, hi


def _points_at(t: Tensor, mean_direction: Tensor) -> Tensor:
    """Points on the sphere with cosine similarity ``t`` to ``mean_direction``.

    By rotational symmetry, every distribution in this module depends on
    ``x`` only through ``t = mean_direction^T x``, so any fixed unit tangent
    direction works here (see :func:`_tangent_direction`).

    Args:
        t: Cosine similarities, shape ``(n,)``.
        mean_direction: Unit mean direction, shape ``(dim,)``.

    Returns:
        Points on the sphere, shape ``(n, dim)``.
    """
    tangent = _tangent_direction(mean_direction)
    normal_weight = torch.sqrt((1 - t**2).clamp_min(0.0))
    return t.unsqueeze(-1) * mean_direction + normal_weight.unsqueeze(-1) * tangent


def _log_sigma_t(t: Tensor, ambient_dim: int) -> Tensor:
    """Lemma 2.1's ``log(sigma_t)``: the log Lebesgue density of ``t`` under a
    uniform direction on ``S^(ambient_dim - 1)``.
    """
    log_beta = (
        math.lgamma((ambient_dim - 1) / 2)
        + math.lgamma(0.5)
        - math.lgamma(ambient_dim / 2)
    )
    return -log_beta + ((ambient_dim - 3) / 2) * torch.log(1 - t**2)


def _tangent_direction(mean_direction: Tensor) -> Tensor:
    """An arbitrary unit vector orthogonal to ``mean_direction``."""
    seed = torch.zeros_like(mean_direction)
    seed[0] = 1.0
    if torch.allclose(mean_direction.abs(), seed.abs(), atol=1e-3):
        seed = torch.zeros_like(mean_direction)
        seed[1] = 1.0
    tangent = seed - (seed @ mean_direction) * mean_direction
    return tangent / tangent.norm()


if __name__ == "__main__":
    main(sys.argv[1:])
