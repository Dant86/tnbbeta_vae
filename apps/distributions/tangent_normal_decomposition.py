"""Illustrates the tangent-normal decomposition x = t*mu + sqrt(1-t^2)*xi on S^2.

Usage:
    uv run python -m apps.distributions.tangent_normal_decomposition \\
        [--output PATH] [--t 0.5] [--phi 0.4]

Static 3-D diagram for the paper's "unit interval to hypersphere" section
(``writeup/weekly_tex_summaries/paper/sections/2_unit_interval_to_hypersphere.tex``):
fixes mu at the north pole of S^2 (ambient R^3) and places a worked x at
``t = mu^T x`` away from it, so the decomposition

    x = t * mu + sqrt(1 - t^2) * xi,   t in [-1, 1]

is a literal right triangle (origin, ``t*mu``, ``x``) sitting on the sphere's
surface. ``xi`` is drawn both as the unit vector the sampling procedure draws it
as (lying in the plane through the origin orthogonal to mu) and, dashed, as the
``sqrt(1-t^2)*xi`` leg of that triangle. The shaded disk and its boundary circle
in that same plane are exactly ``S^{d-2}`` -- here ``S^1`` -- the lower-dimensional
hypersphere ``xi ~ Uniform(S^{d-2})`` is sampled from in every spherical
distribution this paper builds (section 2's three-step procedure). The segment
from the origin to ``t*mu`` is highlighted separately to call out ``t``.

Plotly + kaleido, styled via the shared ``tnbbeta_vae.plotting`` theme (this
project's other 3-D sphere plots -- ``notebooks/plot_sphere_3d.py``,
``apps.eval.hammer_projection`` -- use the same template and a faint
``go.Surface`` sphere mesh).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from tnbbeta_vae.plotting import TEMPLATE_NAME

_MU_COLOR = "#1f77b4"
_X_COLOR = "#d62728"
_XI_COLOR = "#2ca02c"
_T_COLOR = "#ff7f0e"
_HELPER_COLOR = "#7f7f7f"
_DISK_COLOR = _XI_COLOR

__all__ = ["build_figure", "main"]


def main(argv: list[str] | None = None) -> None:
    """Builds the tangent-normal decomposition figure and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str, default="tangent_normal_decomposition.png"
    )
    parser.add_argument("--t", type=float, default=0.5, help="t = mu^T x, in (-1, 1).")
    parser.add_argument(
        "--phi",
        type=float,
        default=0.4,
        help="xi's azimuth within the equatorial plane, in radians.",
    )
    args = parser.parse_args(argv)

    figure = build_figure(args.t, args.phi)
    figure.write_image(args.output, width=750, height=620, scale=2)
    print(f"Wrote {args.output}")


def build_figure(t: float, phi: float) -> go.Figure:
    """Builds the S^2 tangent-normal decomposition diagram.

    Args:
        t: ``t = mu^T x``, in ``(-1, 1)`` (exclusive so ``x``, ``mu`` and the
            equatorial plane stay visually distinct).
        phi: ``xi``'s azimuth within the equatorial plane (the plane through
            the origin orthogonal to ``mu``), in radians.

    Returns:
        A plotly figure with a faint sphere, the shaded ``S^{d-2}`` disk, and
        the ``mu``, ``xi``, ``x`` vectors with their connecting construction
        lines.
    """
    mu = np.array([0.0, 0.0, 1.0])
    xi = np.array([np.cos(phi), np.sin(phi), 0.0])
    normal_weight = float(np.sqrt(max(1.0 - t**2, 0.0)))
    x = t * mu + normal_weight * xi
    foot = t * mu  # x's projection onto the mu-axis.

    figure = go.Figure()
    figure.add_trace(_sphere_mesh())
    figure.add_trace(_equatorial_disk())
    figure.add_trace(_equatorial_circle())

    figure.add_trace(_segment(foot, x, _XI_COLOR, width=4, dash="dash"))
    figure.add_trace(_segment(np.zeros(3), foot, _T_COLOR, width=10))
    figure.add_trace(_label(foot / 2 + np.array([-0.16, -0.06, 0.0]), "t", _T_COLOR))

    for vector, color, name in (
        (mu, _MU_COLOR, "μ"),
        (xi, _XI_COLOR, "ξ"),
        (x, _X_COLOR, "x"),
    ):
        figure.add_trace(_segment(np.zeros(3), vector, color, width=6))
        figure.add_trace(_arrowhead(vector, color))
        figure.add_trace(_label(vector * 1.18, name, color))

    figure.update_layout(
        template=TEMPLATE_NAME,
        title={
            "text": "Tangent-normal Decomposition Illustration",
            "y": 0.98,
            "yanchor": "top",
        },
        showlegend=False,
        scene={
            "aspectmode": "cube",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 0.95, "y": -1.05, "z": 0.55}},
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        },
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
    )
    return figure


def _sphere_mesh(resolution: int = 40) -> go.Surface:
    """A faint unit sphere, matching ``notebooks/plot_sphere_3d.py``'s style."""
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    return go.Surface(
        x=np.outer(np.sin(theta), np.cos(phi)),
        y=np.outer(np.sin(theta), np.sin(phi)),
        z=np.outer(np.cos(theta), np.ones_like(phi)),
        opacity=0.08,
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
        colorscale=[[0, "#888888"], [1, "#888888"]],
    )


def _equatorial_disk(resolution: int = 60) -> go.Mesh3d:
    """Shaded unit disk in the plane through the origin orthogonal to mu (z=0).

    This is the ambient 2-D subspace ``xi`` lives in; its boundary
    (:func:`_equatorial_circle`) is the actual ``S^{d-2}`` support.
    """
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    x = np.concatenate([[0.0], np.cos(angles)])
    y = np.concatenate([[0.0], np.sin(angles)])
    z = np.zeros_like(x)
    center = 0
    rim = np.arange(1, resolution + 1)
    next_rim = np.roll(rim, -1)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=np.full(resolution, center),
        j=rim,
        k=next_rim,
        color=_DISK_COLOR,
        opacity=0.15,
        hoverinfo="skip",
        showlegend=False,
    )


def _equatorial_circle(resolution: int = 200) -> go.Scatter3d:
    """The boundary circle of :func:`_equatorial_disk`: the true ``S^{d-2}``."""
    angles = np.linspace(0, 2 * np.pi, resolution)
    return go.Scatter3d(
        x=np.cos(angles),
        y=np.sin(angles),
        z=np.zeros_like(angles),
        mode="lines",
        line={"color": _DISK_COLOR, "width": 4},
        hoverinfo="skip",
        showlegend=False,
    )


def _segment(
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    color: str,
    width: float,
    dash: str = "solid",
) -> go.Scatter3d:
    """A straight 3-D line segment from ``start`` to ``end``."""
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines",
        line={"color": color, "width": width, "dash": dash},
        hoverinfo="skip",
        showlegend=False,
    )


def _arrowhead(tip: npt.NDArray[np.float64], color: str) -> go.Cone:
    """A small cone pointing along ``tip``'s direction, anchored at ``tip``."""
    direction = tip / np.linalg.norm(tip)
    return go.Cone(
        x=[tip[0]],
        y=[tip[1]],
        z=[tip[2]],
        u=[direction[0]],
        v=[direction[1]],
        w=[direction[2]],
        sizemode="absolute",
        sizeref=0.2,
        anchor="tip",
        showscale=False,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
    )


def _label(position: npt.NDArray[np.float64], text: str, color: str) -> go.Scatter3d:
    """A text label floating at ``position``."""
    return go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode="text",
        text=[text],
        textfont={"color": color, "size": 22},
        hoverinfo="skip",
        showlegend=False,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
