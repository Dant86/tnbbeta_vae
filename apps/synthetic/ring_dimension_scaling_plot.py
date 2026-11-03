"""Figures for the ring dimension-scaling study.

Two figures: the test ELBO gap versus ambient dimension per family (per-seed points and
the seed mean) for conditions A (true frozen decoder) and B (learned MLP decoder), and
the paired difference ``gap(comparator) - gap(tnbbeta)`` versus dimension (positive =
TNBBeta better). Rendered with plotly and the project's ``alptex`` template.

Usage:
    uv run python -m apps.synthetic.ring_dimension_scaling_plot \
        --out-dir ring_recovery/dimension_scaling --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.synthetic.ring_dimension_scaling import (
    COMPARATORS,
    DIMENSIONS,
    MARGIN_NATS,
    Runs,
    load_runs,
    valid,
)
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

STAMP = "2026-11-02"
CONDITIONS = (("A", "True frozen decoder"), ("B", "Learned MLP decoder"))
ARMS = (
    ("vmf", "vMF", "#2a78d6", "circle"),
    ("power_spherical", "PowerSpherical", "#1baf7a", "square"),
    ("tnbbeta", "TNBBeta", "#eb6834", "diamond"),
    ("tnbbeta_init0", "TNBBeta (p init 0)", "#b8861b", "triangle-up"),
    ("tnbbeta_epsinit", "TNBBeta (epsilon init (d-1)/2)", "#6250d6", "cross"),
)
INK = "#1a1a1a"
GRID = "#e4e4e4"
JITTER = 0.07


def main(argv: list[str] | None = None) -> None:
    """Writes the two figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.out_dir)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    gap_figure(runs).write_image(
        args.fig_dir / f"ring_dimension_gap_vs_d_{STAMP}.png", scale=3
    )
    difference_figure(runs).write_image(
        args.fig_dir / f"ring_dimension_paired_difference_vs_d_{STAMP}.png", scale=3
    )


def gap_figure(runs: Runs, key: str = "gap_elbo") -> go.Figure:
    """Test gap versus ``d`` per family, conditions side by side."""
    fig = make_subplots(
        rows=1,
        cols=len(CONDITIONS),
        subplot_titles=[title for _, title in CONDITIONS],
        horizontal_spacing=0.09,
    )
    rng = np.random.default_rng(0)
    for column, (condition, _) in enumerate(CONDITIONS, start=1):
        for index, (arm, label, colour, symbol) in enumerate(ARMS):
            offset = (index - (len(ARMS) - 1) / 2) * 0.13
            positions, means, points_x, points_y = [], [], [], []
            for position, dim in enumerate(DIMENSIONS):
                values = np.array(
                    list(valid(runs.get((condition, dim, arm), {}), key).values())
                )
                if len(values) == 0:
                    continue
                positions.append(position + offset)
                means.append(values.mean())
                points_x += list(
                    position + offset + rng.uniform(-JITTER, JITTER, len(values))
                )
                points_y += list(values)
            fig.add_trace(
                go.Scatter(
                    x=points_x,
                    y=points_y,
                    mode="markers",
                    marker={
                        "color": colour,
                        "symbol": symbol,
                        "size": 5,
                        "opacity": 0.45,
                    },
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=means,
                    mode="lines+markers",
                    line={"color": colour, "width": 2},
                    marker={"color": colour, "symbol": symbol, "size": 9},
                    name=label,
                    legendgroup=arm,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(DIMENSIONS))),
            ticktext=[str(d) for d in DIMENSIONS],
            title_text="Ambient dimension d",
            row=1,
            col=column,
        )
        fig.update_yaxes(
            title_text="Test ELBO gap (nats)" if column == 1 else None,
            row=1,
            col=column,
        )
    return _finish(fig, 1500, 600)


def difference_figure(runs: Runs, key: str = "gap_elbo") -> go.Figure:
    """Paired difference (comparator minus TNBBeta) versus ``d``, per seed and mean."""
    fig = make_subplots(
        rows=1,
        cols=len(CONDITIONS),
        subplot_titles=[title for _, title in CONDITIONS],
        horizontal_spacing=0.09,
    )
    rng = np.random.default_rng(1)
    names = {
        "vmf": ("vMF minus TNBBeta", "#2a78d6", "circle"),
        "power_spherical": ("PowerSpherical minus TNBBeta", "#1baf7a", "square"),
    }
    for column, (condition, _) in enumerate(CONDITIONS, start=1):
        for index, other in enumerate(COMPARATORS):
            label, colour, symbol = names[other]
            offset = (index - 0.5) * 0.18
            positions, means, points_x, points_y = [], [], [], []
            for position, dim in enumerate(DIMENSIONS):
                tnb = valid(runs.get((condition, dim, "tnbbeta"), {}), key)
                cap = valid(runs.get((condition, dim, other), {}), key)
                seeds = sorted(set(tnb) & set(cap))
                if not seeds:
                    continue
                diff = np.array([cap[s] - tnb[s] for s in seeds])
                positions.append(position + offset)
                means.append(diff.mean())
                points_x += list(
                    position + offset + rng.uniform(-JITTER, JITTER, len(diff))
                )
                points_y += list(diff)
            fig.add_trace(
                go.Scatter(
                    x=points_x,
                    y=points_y,
                    mode="markers",
                    marker={
                        "color": colour,
                        "symbol": symbol,
                        "size": 5,
                        "opacity": 0.45,
                    },
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=means,
                    mode="lines+markers",
                    line={"color": colour, "width": 2},
                    marker={"color": colour, "symbol": symbol, "size": 9},
                    name=label,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        for level, colour, dash in (
            (0.0, INK, "dot"),
            (MARGIN_NATS, "#999999", "dash"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[-0.4, len(DIMENSIONS) - 0.6],
                    y=[level, level],
                    mode="lines",
                    line={"color": colour, "width": 1, "dash": dash},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(DIMENSIONS))),
            ticktext=[str(d) for d in DIMENSIONS],
            title_text="Ambient dimension d",
            row=1,
            col=column,
        )
        fig.update_yaxes(
            title_text="Gap difference (nats; positive = TNBBeta better)"
            if column == 1
            else None,
            row=1,
            col=column,
        )
    return _finish(fig, 1500, 600)


def _finish(fig: go.Figure, width: int, height: int) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 30, "t": 60, "b": 130},
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": -0.2,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 15},
        },
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=INK, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, linecolor=INK, ticks="outside", zeroline=False)
    return fig


if __name__ == "__main__":
    main()
