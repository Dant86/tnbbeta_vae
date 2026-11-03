"""Figures for the metric-first graph-search study (plotly, ``alptex`` template).

Usage:
    uv run python -m apps.eval.graph_search_metric_first_plot \
        --fig-dir writeup/results/figures

Reads ``results_exploratory.json``, ``results_fresh.json`` and the covariate tables
under ``$TNBBETA_CHECKPOINT_DIR/graph_search/metric_first`` and writes four PNGs at 3x
scale with the legend at the bottom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.eval.graph_search_covariates import covariate_path
from apps.eval.graph_search_metric_first import (
    METRICS,
    design_matrix,
    family_of,
    output_dir,
)
from tnbbeta_vae.data.graph_space import FAMILIES
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

STAMP = "2026-11-02"
LABEL_COLOURS = {"better": "#1baf7a", "neither": "#9a9a9a", "worse": "#eb6834"}
STAGE_COLOURS = {"discovery": "#2a78d6", "fresh": "#eb6834"}
INK = "#1a1a1a"
GRID = "#e4e4e4"


def main(argv: list[str] | None = None) -> None:
    """Writes the four figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    exploratory = json.loads((output_dir() / "results_exploratory.json").read_text())
    fresh = json.loads((output_dir() / "results_fresh.json").read_text())
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "metric_first_family_by_metric": matrix_figure(fresh["records"]),
        "metric_first_truth_kl_by_config": effect_figure(exploratory, fresh),
        "metric_first_covariates": covariate_figure(exploratory, fresh),
        "metric_first_fresh_prediction": prediction_figure(fresh),
    }
    for name, figure in figures.items():
        figure.write_image(args.fig_dir / f"{name}_{STAMP}.png", scale=3)


def _layout(figure: go.Figure, height: int, width: int = 1100) -> go.Figure:
    figure.update_layout(
        template=TEMPLATE_NAME,
        font={"family": FONT_FAMILY, "color": INK},
        height=height,
        width=width,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.32,
            "x": 0.5,
            "xanchor": "center",
        },
        plot_bgcolor="white",
        margin={"l": 140, "r": 20, "t": 50, "b": 200},
    )
    figure.update_xaxes(gridcolor=GRID, linecolor=INK)
    figure.update_yaxes(gridcolor=GRID, linecolor=INK)
    return figure


def matrix_figure(records: list[dict[str, Any]]) -> go.Figure:
    """Heatmap of the median standardised effect per family x metric (fresh stage)."""
    metrics = [m for m in METRICS if any(r["metric"] == m for r in records)]
    rows = [*FAMILIES, "all"]
    z = np.full((len(rows), len(metrics)), np.nan)
    text = [["" for _ in metrics] for _ in rows]
    for i, family in enumerate(rows):
        for j, metric in enumerate(metrics):
            chosen = [
                r
                for r in records
                if r["metric"] == metric and (family == "all" or r["family"] == family)
            ]
            if not chosen:
                continue
            z[i, j] = np.median([r["dz_min"] for r in chosen])
            better = sum(r["label"] == "better" for r in chosen)
            worse = sum(r["label"] == "worse" for r in chosen)
            text[i][j] = f"{better}/{worse}"
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=metrics,
            y=rows,
            text=text,
            texttemplate="%{text}",
            zmid=0,
            zmin=-1.5,
            zmax=1.5,
            colorscale="RdBu",
            colorbar={
                "title": "median dz (positive: TNBBeta better)",
                "orientation": "h",
                "y": -0.6,
                "len": 0.6,
                "thickness": 14,
            },
        )
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(tickangle=-45)
    figure.update_layout(
        title="Fresh stage: median standardised effect and better/worse config counts"
    )
    return _layout(figure, 620, 1250)


def effect_figure(exploratory: dict[str, Any], fresh: dict[str, Any]) -> go.Figure:
    """Per-config truth-KL effect with intervals, by family, discovery and fresh."""
    figure = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=("Discovery", "Fresh")
    )
    for col, (_stage, data) in enumerate(
        (("discovery", exploratory), ("fresh", fresh)), 1
    ):
        rows = [r for r in data["records"] if r["metric"] == "truth_kl"]
        rows.sort(key=lambda r: (FAMILIES.index(r["family"]), r["E"]))
        for label, colour in LABEL_COLOURS.items():
            chosen = [(i, r) for i, r in enumerate(rows) if r["label"] == label]
            if not chosen:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[i for i, _ in chosen],
                    y=[r["E"] for _, r in chosen],
                    mode="markers",
                    marker={"color": colour, "size": 6},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [r["E_hi"] - r["E"] for _, r in chosen],
                        "arrayminus": [r["E"] - r["E_lo"] for _, r in chosen],
                        "thickness": 1,
                        "color": colour,
                    },
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        ticks = [
            (float(np.mean([i for i, r in enumerate(rows) if r["family"] == f])), f)
            for f in FAMILIES
            if any(r["family"] == f for r in rows)
        ]
        figure.update_xaxes(
            tickmode="array",
            tickvals=[t for t, _ in ticks],
            ticktext=[f for _, f in ticks],
            tickangle=-60,
            row=1,
            col=col,
        )
    figure.add_hline(y=0, line_color=INK, line_width=1)
    figure.update_yaxes(title="TNBBeta advantage in truth KL (nats)", col=1)
    return _layout(figure, 760, 1300)


def covariate_figure(exploratory: dict[str, Any], fresh: dict[str, Any]) -> go.Figure:
    """Config effect S_c against the three covariates with the largest rule weights."""
    rule = json.loads((output_dir() / "prediction_rule.json").read_text())
    top = np.argsort(-np.abs(rule["coefficients"]))[:3]
    names = [rule["names"][i] for i in top]
    figure = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=names)
    for stage, data in (("discovery", exploratory), ("fresh", fresh)):
        covariates = json.loads(covariate_path(stage).read_text())
        mean = {
            c: {
                k: float(np.mean([d[k] for d in v.values()]))
                for k in next(iter(v.values()))
            }
            for c, v in covariates.items()
        }
        x, columns = design_matrix(mean, data["configs"])
        for col, name in enumerate(names, 1):
            figure.add_trace(
                go.Scatter(
                    x=x[:, columns.index(name)],
                    y=data["S"],
                    mode="markers",
                    marker={
                        "color": STAGE_COLOURS[stage],
                        "size": 5,
                        "opacity": 0.65,
                        "symbol": "circle" if stage == "discovery" else "diamond",
                    },
                    name=stage,
                    legendgroup=stage,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
    figure.add_hline(y=0, line_color=INK, line_width=1)
    figure.update_yaxes(title="S_c = E_c / KL0 (truth KL, headroom share)", col=1)
    return _layout(figure, 520, 1300)


def prediction_figure(fresh: dict[str, Any]) -> go.Figure:
    """Fresh-stage predicted against observed S_c, by family."""
    figure = go.Figure()
    families = np.array([family_of(c) for c in fresh["configs"]])
    palette = [
        "#2a78d6", "#1baf7a", "#eb6834", "#b8861b", "#6250d6",
        "#c43e8a", "#4a4a4a", "#17a2b8", "#8c564b", "#7f9f1c",
    ]  # fmt: skip
    for family, colour in zip(FAMILIES, palette, strict=True):
        pick = families == family
        figure.add_trace(
            go.Scatter(
                x=np.asarray(fresh["prediction"])[pick],
                y=np.asarray(fresh["S"])[pick],
                mode="markers",
                marker={"color": colour, "size": 7},
                name=family,
            )
        )
    low = float(min(min(fresh["prediction"]), min(fresh["S"])))
    high = float(max(max(fresh["prediction"]), max(fresh["S"])))
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": INK, "dash": "dash", "width": 1},
            name="y = x",
        )
    )
    figure.update_xaxes(title="Rule prediction of S_c (graph covariates only)")
    figure.update_yaxes(title="Observed S_c on fresh configs")
    figure.update_layout(
        title=(
            f"Fresh stage: rho = {fresh['rho']:+.2f} "
            f"(lower bound {fresh['lower']:+.2f}), R2 = {fresh['r2']:+.2f}"
        )
    )
    return _layout(figure, 620, 900)


if __name__ == "__main__":
    main()
