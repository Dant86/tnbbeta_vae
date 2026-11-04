# pyright: reportArgumentType=false
"""Figures for the tuned-kappa control (decoder ``L``).

Three figures (plotly, ``alptex`` template, 3x, legend at the bottom): (1) test ELBO
gap against pinned kappa per ``(d, sigma)`` with the TNBBeta mean as a horizontal line
and the trainable arms at their fitted kappa; (2) kappa trajectories of the trainable
cap arms; (3) per-seed paired difference ``best-tuned cap - TNBBeta`` per cell.

Usage:
    uv run python -m apps.synthetic.linear_decoder_kappa_control_plot \
        --out-dir ring_recovery/kappa_control \
        --parent-dir ring_recovery/linear_decoder \
        --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.synthetic.linear_decoder_kappa_control import (
    DIMS,
    FAMILIES,
    PIN_GRID,
    SIGMAS,
    Runs,
    best_arm,
    cap_pool,
    load_parent,
    load_runs,
    valid_gaps,
)
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

STAMP = "2026-11-03"
COLORS = {"vmf": "#2a78d6", "power_spherical": "#1baf7a", "tnbbeta": "#eb6834"}
NAMES = {"vmf": "vMF", "power_spherical": "PowerSpherical", "tnbbeta": "TNBBeta"}
INK = "#1a1a1a"
GRID = "#e4e4e4"
TRAJECTORY_ARMS = (
    ("{f}", "init d", "solid"),
    ("{f}_k5d", "init 5d", "dash"),
    ("{f}_k20d", "init 20d", "dot"),
    ("{f}_H", "fast head, 600 ep", "dashdot"),
)


def main(argv: list[str] | None = None) -> None:
    """Writes the three figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.out_dir)
    for key, value in load_parent(args.parent_dir).items():
        runs.setdefault(key, value)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for stem, figure in (
        ("kappa_control_gap_vs_kappa", gap_vs_kappa_figure(runs)),
        ("kappa_control_trajectories", trajectory_figure(runs)),
        ("kappa_control_paired_difference", paired_difference_figure(runs)),
    ):
        path = args.fig_dir / f"{stem}_{STAMP}.png"
        figure.write_image(path, scale=3)
        print(f"wrote {path}")


def gap_vs_kappa_figure(runs: Runs) -> go.Figure:
    """Gap against pinned kappa: rows ``d``, columns ``sigma``."""
    fig = make_subplots(
        rows=len(DIMS),
        cols=len(SIGMAS),
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.06,
        column_titles=[f"sigma = {s:g}" for s in SIGMAS],
        row_titles=[f"d = {d}" for d in DIMS],
    )
    for row, dim in enumerate(DIMS, start=1):
        for col, sigma in enumerate(SIGMAS, start=1):
            show = row == 1 and col == 1
            for family in FAMILIES:
                xs, ys = [], []
                for kappa in PIN_GRID:
                    gaps = valid_gaps(
                        runs.get((dim, sigma, f"{family}_pin{kappa}"), {})
                    )
                    if gaps:
                        xs.append(kappa)
                        ys.append(float(np.mean(list(gaps.values()))))
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        line={"color": COLORS[family], "width": 2},
                        marker={"size": 8},
                        name=f"{NAMES[family]} pinned kappa",
                        legendgroup=family,
                        showlegend=show,
                    ),
                    row=row,
                    col=col,
                )
                free = [
                    a
                    for a in (family, f"{family}_k5d", f"{family}_k20d", f"{family}_H")
                ]
                fx, fy = [], []
                for arm in free:
                    group = runs.get((dim, sigma, arm), {})
                    gaps = valid_gaps(group)
                    kappas = valid_gaps(group, "kappa_median")
                    if gaps and kappas:
                        fx.append(float(np.median(list(kappas.values()))))
                        fy.append(float(np.mean(list(gaps.values()))))
                fig.add_trace(
                    go.Scatter(
                        x=fx,
                        y=fy,
                        mode="markers",
                        marker={
                            "size": 11,
                            "symbol": "x",
                            "color": COLORS[family],
                            "line": {"width": 2},
                        },
                        name=f"{NAMES[family]} trainable (fitted kappa)",
                        legendgroup=family,
                        showlegend=show,
                    ),
                    row=row,
                    col=col,
                )
            tnb = valid_gaps(runs.get((dim, sigma, "tnbbeta"), {}))
            if tnb:
                fig.add_hline(
                    y=float(np.mean(list(tnb.values()))),
                    line={"color": COLORS["tnbbeta"], "width": 2, "dash": "dash"},
                    row=row,
                    col=col,
                )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": COLORS["tnbbeta"], "dash": "dash", "width": 2},
            name="TNBBeta mean gap",
        )
    )
    fig.update_xaxes(type="log", gridcolor=GRID, linecolor=INK)
    fig.update_xaxes(title_text="Kappa", row=len(DIMS))
    fig.update_yaxes(type="log", gridcolor=GRID, linecolor=INK)
    fig.update_yaxes(title_text="Test ELBO gap (nats)", col=1)
    _style(fig, 1500, 1300)
    return fig


def trajectory_figure(runs: Runs) -> go.Figure:
    """Median kappa against epoch for trainable cap arms (rows sigma, columns d)."""
    fig = make_subplots(
        rows=len(SIGMAS),
        cols=len(DIMS),
        vertical_spacing=0.07,
        horizontal_spacing=0.05,
        column_titles=[f"d = {d}" for d in DIMS],
        row_titles=[f"sigma = {s:g}" for s in SIGMAS],
    )
    for row, sigma in enumerate(SIGMAS, start=1):
        for col, dim in enumerate(DIMS, start=1):
            for family in FAMILIES:
                for template, label, dash in TRAJECTORY_ARMS:
                    arm = template.format(f=family)
                    curves = [
                        r["trajectory"]["kappa"]
                        for r in runs.get((dim, sigma, arm), {}).values()
                        if r.get("trajectory", {}).get("kappa")
                    ]
                    if not curves:
                        continue
                    length = min(len(c) for c in curves)
                    median = np.median([c[:length] for c in curves], axis=0)
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(1, length + 1),
                            y=median,
                            mode="lines",
                            line={"color": COLORS[family], "dash": dash, "width": 2},
                            name=f"{NAMES[family]}, {label}",
                            legendgroup=f"{family}{label}",
                            showlegend=row == 1 and col == 1,
                        ),
                        row=row,
                        col=col,
                    )
    fig.update_xaxes(gridcolor=GRID, linecolor=INK)
    fig.update_xaxes(title_text="Epoch", row=len(SIGMAS))
    fig.update_yaxes(type="log", gridcolor=GRID, linecolor=INK)
    fig.update_yaxes(title_text="Median kappa", col=1)
    _style(fig, 1500, 1100)
    return fig


def paired_difference_figure(runs: Runs) -> go.Figure:
    """Per-seed ``best-tuned cap - TNBBeta`` per cell: one panel per ``sigma``."""
    fig = make_subplots(
        rows=1,
        cols=len(SIGMAS),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[f"sigma = {s:g}" for s in SIGMAS],
    )
    rng = np.random.default_rng(0)
    for col, sigma in enumerate(SIGMAS, start=1):
        for index, family in enumerate(FAMILIES):
            xs, ys, means = [], [], []
            for position, dim in enumerate(DIMS):
                tnb = valid_gaps(runs.get((dim, sigma, "tnbbeta"), {}))
                pool = cap_pool(runs, dim, sigma, family)
                arm = best_arm(runs, dim, sigma, pool) if pool else None
                cap = valid_gaps(runs[(dim, sigma, arm)]) if arm else {}
                diff = np.array([cap[s] - tnb[s] for s in sorted(set(cap) & set(tnb))])
                offset = (index - 0.5) * 0.2
                xs += list(position + offset + rng.uniform(-0.05, 0.05, len(diff)))
                ys += list(diff)
                means.append(diff.mean() if len(diff) else np.nan)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker={"color": COLORS[family], "size": 6, "opacity": 0.5},
                    showlegend=False,
                    legendgroup=family,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[p + (index - 0.5) * 0.2 for p in range(len(DIMS))],
                    y=means,
                    mode="lines+markers",
                    line={"color": COLORS[family], "width": 2},
                    marker={"color": COLORS[family], "size": 11},
                    name=f"{NAMES[family]} (best-tuned cap) - TNBBeta",
                    legendgroup=family,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0.05, line={"color": INK, "dash": "dot"}, row=1, col=col)
        fig.add_hline(y=0.0, line={"color": INK, "width": 1}, row=1, col=col)
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
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=INK,
        title_text="Gap difference (nats), dotted = 0.05",
        col=1,
    )
    _style(fig, 1500, 600)
    return fig


def _style(fig: go.Figure, width: int, height: int) -> None:
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 40, "t": 50, "b": 150},
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": -0.08 if height > 800 else -0.2,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 15},
        },
    )


if __name__ == "__main__":
    main()
