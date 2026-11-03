"""Figures for the linear-decoder ring study (decoder ``L``).

Three figures, one panel per noise scale ``sigma``: the test ELBO gap versus ambient
dimension per family (per-seed points and the seed mean), the learned axis's
``|cos(u, a)|`` with the true axis, and the median ring statistic ``R`` (about 0 = ring,
about 1 = cap). Rendered with plotly and the project's ``alptex`` template.

Usage:
    uv run python -m apps.synthetic.linear_decoder_ring_plot \
        --out-dir ring_recovery/linear_decoder --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.synthetic.linear_decoder_ring import DIMS, SIGMAS, Runs, load_runs, valid
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

STAMP = "2026-11-02"
ARMS = (
    ("vmf", "vMF", "#2a78d6", "circle"),
    ("power_spherical", "PowerSpherical", "#1baf7a", "square"),
    ("tnbbeta", "TNBBeta", "#eb6834", "diamond"),
    ("tnbbeta_init0", "TNBBeta (p init 0)", "#b8861b", "triangle-up"),
)
INK = "#1a1a1a"
GRID = "#e4e4e4"
JITTER = 0.07
FIGURES = (
    ("gap_elbo", "Test ELBO gap (nats)", "linear_ring_gap_vs_d", "log"),
    (
        "abs_cos_axis",
        "|cos(learned axis, true axis)|",
        "linear_ring_alignment_vs_d",
        None,
    ),
    (
        "ring_r_median",
        "Ring statistic R (0 = ring, 1 = cap)",
        "linear_ring_ringstat_vs_d",
        None,
    ),
)


def main(argv: list[str] | None = None) -> None:
    """Writes the three figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.out_dir)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for key, label, stem, axis_type in FIGURES:
        figure = metric_figure(runs, key, label, axis_type)
        figure.write_image(args.fig_dir / f"{stem}_{STAMP}.png", scale=3)
        print(f"wrote {args.fig_dir / stem}_{STAMP}.png")


def metric_figure(
    runs: Runs, key: str, label: str, axis_type: str | None = None
) -> go.Figure:
    """One metric versus ``d`` for decoder ``L`` (a panel per ``sigma``, trace per arm).

    Args:
        runs: Loaded runs.
        key: Per-run summary key to plot.
        label: Y-axis title.
        axis_type: Plotly y-axis type (``"log"`` for the gap), or ``None`` for linear.

    Returns:
        The figure (legend at the bottom).
    """
    fig = make_subplots(
        rows=1,
        cols=len(SIGMAS),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[f"sigma = {s:g}" for s in SIGMAS],
    )
    rng = np.random.default_rng(0)
    for col, sigma in enumerate(SIGMAS, start=1):
        for index, (arm, name, color, symbol) in enumerate(ARMS):
            xs, ys, means = [], [], []
            for position, dim in enumerate(DIMS):
                values = np.array(
                    list(valid(runs.get(("L", dim, sigma, arm), {}), key).values())
                )
                if axis_type == "log":
                    values = values[values > 0]
                offset = (index - (len(ARMS) - 1) / 2) * 0.16
                xs += list(
                    position + offset + rng.uniform(-JITTER, JITTER, len(values))
                )
                ys += list(values)
                means.append(values.mean() if len(values) else np.nan)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker={
                        "color": color,
                        "symbol": symbol,
                        "size": 6,
                        "opacity": 0.45,
                    },
                    showlegend=False,
                    legendgroup=arm,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        p + (index - (len(ARMS) - 1) / 2) * 0.16
                        for p in range(len(DIMS))
                    ],
                    y=means,
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"color": color, "symbol": symbol, "size": 11},
                    name=name,
                    legendgroup=arm,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(DIMS))),
            ticktext=[str(d) for d in DIMS],
            title_text="Ambient dimension d" if col == 2 else None,
            gridcolor=GRID,
            linecolor=INK,
            row=1,
            col=col,
        )
    fig.update_yaxes(gridcolor=GRID, linecolor=INK, title_text=label, col=1)
    if axis_type:
        fig.update_yaxes(type=axis_type)
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=1500,
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 20, "t": 50, "b": 130},
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": -0.2,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 15},
        },
    )
    return fig


if __name__ == "__main__":
    main()
