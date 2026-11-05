"""Figures of the real-graph revision (layout fixed in the pre-registration).

Reads ``analysis.json``.
Usage (login node: kaleido fails on compute nodes):
    uv run python -m apps.real_graph.figures --fig-dir writeup/results/figures

Five figures, each written at scale 3 with the project ``alptex`` template: (1) the main
bars (AUROC and log loss, features on and off), (2) degree buckets, (3) edge thinning,
(4) the m-pinning dose-response and shape statistics (descriptive), (5) the Louvain
community subgraphs (descriptive). No captions are drawn inside the figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.real_graph.run import DATASETS, FAMILIES, FEATURES, PIN_FRACTIONS, root_dir
from tnbbeta_vae.models.real_graph_eval import BUCKET_LABELS
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

__all__ = ["main", "render_all"]

FAMILY_STYLE = {
    "gaussian": ("Gaussian", "#7f7f7f"),
    "vmf": ("vMF", "#1f77b4"),
    "tnbbeta": ("TNBBeta", "#d62728"),
    "power_spherical": ("Power Spherical", "#2ca02c"),
}
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "mag_cs": "MAG-CS",
    "amazon_computers": "Amazon Computers",
    "amazon_photo": "Amazon Photo",
    "squirrel": "Squirrel",
}
DATASET_COLORS = {
    "cora": "#1f77b4",
    "citeseer": "#ff7f0e",
    "pubmed": "#2ca02c",
    "mag_cs": "#9467bd",
    "amazon_computers": "#1f77b4",
    "amazon_photo": "#ff7f0e",
    "squirrel": "#2ca02c",
}
FEATURE_TITLES = {"on": "With Node Features", "off": "Without Node Features"}
INK = "#1a1a1a"
GRID = "#e6e6e6"


def main(argv: list[str] | None = None) -> None:
    """Renders every figure from ``analysis.json``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=Path("writeup/results/figures"))
    parser.add_argument("--analysis", type=Path, default=None)
    args = parser.parse_args(argv)
    path = args.analysis or root_dir() / "analysis.json"
    results = json.loads(path.read_text())
    for written in render_all(results, args.fig_dir):
        print(f"Wrote {written}")


def render_all(
    results: dict[str, Any], fig_dir: Path, prefix: str = "real_graph_revision"
) -> list[Path]:
    """Builds and writes every figure the analysis results allow.

    Args:
        results: The parsed ``analysis.json``.
        fig_dir: Output directory (created).
        prefix: File-name prefix.

    Returns:
        The written paths.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "main_bars": lambda: main_bars(results["main"]),
        "advantage": lambda: advantage_bars(results["main"]),
        "degree_buckets": lambda: degree_buckets(results["buckets"]),
        "thinning": lambda: thinning(results["thin"]),
        "m_pinning": lambda: m_pinning(results["shape"]),
        "community": lambda: community(results["community"]),
    }
    written = []
    for name, build in builders.items():
        key = {
            "main_bars": "main",
            "advantage": "main",
            "degree_buckets": "buckets",
            "thinning": "thin",
            "m_pinning": "shape",
            "community": "community",
        }[name]
        if key not in results:
            continue
        if name == "m_pinning" and not results["shape"]["pin"]:
            continue
        path = fig_dir / f"{prefix}_{name}.png"
        build().write_image(path, scale=3)
        written.append(path)
    return written


def main_bars(main: dict[str, Any]) -> go.Figure:
    """Figure 1: AUROC and log loss, features on (left) and off (right), 4 families."""
    metrics = [("sampled_auc", "Sampled AUROC"), ("log_loss", "Test Log Loss (Nats)")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{label.split(' (')[0]}, {FEATURE_TITLES[f]}"
            for _, label in metrics
            for f in FEATURES
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.17,
    )
    for row, (metric, label) in enumerate(metrics, start=1):
        for col, features in enumerate(FEATURES, start=1):
            lows, highs = [], []
            for family in FAMILIES:
                name, color = FAMILY_STYLE[family]
                means, plus, minus = [], [], []
                for dataset in DATASETS:
                    cell = main["cells"].get(f"{dataset}|{features}")
                    if cell is None:
                        continue
                    m = cell["families"][family][metric]
                    up, down = _asymmetric(m["mean"], m["ci"])
                    means.append(m["mean"])
                    plus.append(up)
                    minus.append(down)
                    lows.append(m["ci"][0])
                    highs.append(m["ci"][1])
                figure.add_trace(
                    go.Bar(
                        x=[DATASET_LABELS[d] for d in DATASETS],
                        y=means,
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": plus,
                            "arrayminus": minus,
                            "color": INK,
                            "thickness": 1,
                            "width": 3,
                        },
                        marker={"color": color},
                        name=name,
                        legendgroup=name,
                        showlegend=row == 1 and col == 1,
                    ),
                    row=row,
                    col=col,
                )
            pad = 0.1 * (max(highs) - min(lows))
            figure.update_yaxes(
                range=[min(lows) - pad, max(highs) + pad],
                title_text=f"$\\text{{{label}}}$" if col == 1 else None,
                row=row,
                col=col,
            )
    _base_layout(figure, width=1300, height=850)
    figure.update_layout(barmode="group", bargap=0.2)
    return figure


def advantage_bars(main: dict[str, Any]) -> go.Figure:
    """Paired advantage of TNBBeta over the primary comparator, per cell.

    Positive means TNBBeta is better. Three panels (log loss, ECE, sampled AUROC) with
    features on and off as two series and the 95% bootstrap interval over seeds; dashed
    lines mark the margin in the log-loss panel.
    """
    panels = [
        ("log_loss", "Log Loss Advantage (Nats)"),
        ("ece", "ECE Advantage"),
        ("sampled_auc", "Sampled AUROC Advantage"),
    ]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[label.rsplit(" Advantage", 1)[0] for _, label in panels],
        horizontal_spacing=0.09,
    )
    colors = {"on": "#d62728", "off": "#7f7f7f"}
    for col, (metric, label) in enumerate(panels, start=1):
        for features in FEATURES:
            xs, means, plus, minus = [], [], [], []
            for dataset in DATASETS:
                cell = main["cells"].get(f"{dataset}|{features}")
                if cell is None:
                    continue
                summary = cell["comparisons"][metric]["primary"]
                if "ci" not in summary:
                    continue
                up, down = _asymmetric(summary["mean"], summary["ci"])
                xs.append(DATASET_LABELS[dataset])
                means.append(summary["mean"])
                plus.append(up)
                minus.append(down)
            figure.add_trace(
                go.Bar(
                    x=xs,
                    y=means,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": plus,
                        "arrayminus": minus,
                        "color": INK,
                        "thickness": 1,
                        "width": 3,
                    },
                    marker={"color": colors[features]},
                    name=f"Features {features.capitalize()}",
                    legendgroup=features,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        _zero_line(figure, 1, col)
        figure.update_yaxes(
            title_text=f"$\\text{{{label}}}$", row=1, col=col, zeroline=False
        )
    margin = main.get("margin")
    if margin:
        for sign in (1, -1):
            cast("Any", figure).add_hline(
                y=sign * margin,
                line={"color": INK, "width": 1, "dash": "dash"},
                row=1,
                col=1,
            )
    _base_layout(figure, width=1400, height=500)
    figure.update_layout(barmode="group", bargap=0.25, margin={"b": 110})
    return figure


def degree_buckets(buckets: dict[str, Any]) -> go.Figure:
    """Figure 2: per-bucket advantage of TNBBeta (log loss, AUROC), features on/off."""
    metrics = [("log_loss", "Log-Loss Advantage (Nats)"), ("auc", "AUROC Advantage")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{label.split(' (')[0]}, {FEATURE_TITLES[f]}"
            for _, label in metrics
            for f in FEATURES
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.17,
    )
    for row, (metric, label) in enumerate(metrics, start=1):
        for col, features in enumerate(FEATURES, start=1):
            for index, dataset in enumerate(DATASETS):
                cell = buckets["cells"].get(f"{dataset}|{features}")
                if cell is None:
                    continue
                xs, ys, plus, minus = [], [], [], []
                for bucket_index, bucket in enumerate(cell["buckets"]):
                    summary = bucket[metric]
                    if "ci" not in summary:
                        continue
                    xs.append(bucket_index + (index - 1.5) * 0.12)
                    ys.append(summary["mean"])
                    up, down = _asymmetric(summary["mean"], summary["ci"])
                    plus.append(up)
                    minus.append(down)
                figure.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        line={"width": 1, "color": DATASET_COLORS[dataset]},
                        marker={"color": DATASET_COLORS[dataset], "size": 7},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": plus,
                            "arrayminus": minus,
                            "thickness": 1,
                            "width": 3,
                        },
                        name=DATASET_LABELS[dataset],
                        legendgroup=dataset,
                        showlegend=row == 1 and col == 1,
                    ),
                    row=row,
                    col=col,
                )
            _zero_line(figure, row, col)
            figure.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(BUCKET_LABELS))),
                ticktext=list(BUCKET_LABELS),
                title_text="Training Degree of the Lower-Degree Endpoint"
                if row == 2
                else None,
                row=row,
                col=col,
            )
            figure.update_yaxes(
                title_text=label if col == 1 else None, row=row, col=col
            )
    _base_layout(figure, width=1300, height=850)
    return figure


def thinning(thin: dict[str, Any]) -> go.Figure:
    """Figure 3: advantage versus realised mean degree under edge thinning."""
    metrics = [
        ("log_loss", "Log-Loss Advantage (Nats)"),
        ("sampled_auc", "AUROC Advantage"),
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{label.split(' (')[0]}, {FEATURE_TITLES[f]}"
            for _, label in metrics
            for f in FEATURES
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.17,
    )
    for row, (metric, label) in enumerate(metrics, start=1):
        for col, features in enumerate(FEATURES, start=1):
            for dataset in DATASETS:
                cell = thin["cells"].get(f"{dataset}|{features}")
                if cell is None:
                    continue
                levels = sorted(cell["levels"], key=float)
                xs = [cell["levels"][lv]["graph"]["mean_degree"] for lv in levels]
                summaries = [
                    cell["levels"][lv]["metrics"][metric]["advantage"] for lv in levels
                ]
                ys = [s["mean"] for s in summaries]
                errors = [_asymmetric(s["mean"], s["ci"]) for s in summaries]
                figure.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        line={"width": 1, "color": DATASET_COLORS[dataset]},
                        marker={"color": DATASET_COLORS[dataset], "size": 7},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": [e[0] for e in errors],
                            "arrayminus": [e[1] for e in errors],
                            "thickness": 1,
                            "width": 3,
                        },
                        name=DATASET_LABELS[dataset],
                        legendgroup=dataset,
                        showlegend=row == 1 and col == 1,
                    ),
                    row=row,
                    col=col,
                )
            _zero_line(figure, row, col)
            figure.update_xaxes(
                type="log",
                title_text="Realised Mean Degree of the Thinned Graph"
                if row == 2
                else None,
                row=row,
                col=col,
            )
            figure.update_yaxes(
                title_text=label if col == 1 else None, row=row, col=col
            )
    _base_layout(figure, width=1300, height=850)
    return figure


def m_pinning(shape: dict[str, Any]) -> go.Figure:
    """Figure 4 (descriptive): metrics versus the pinned modality fraction f."""
    panels = [
        ("log_loss", "Test Log Loss (Nats)"),
        ("sampled_auc", "Sampled AUROC"),
        ("ece", "Expected Calibration Error"),
        ("second_mode_mass", "Second-Mode Mass"),
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[label for _, label in panels],
        horizontal_spacing=0.1,
        vertical_spacing=0.18,
    )
    for index, (metric, label) in enumerate(panels):
        row, col = index // 2 + 1, index % 2 + 1
        for dataset in DATASETS:
            for features in FEATURES:
                cell = f"{dataset}|{features}"
                if cell not in shape["pin"]:
                    continue
                points = shape["pin"][cell]
                xs = [f for f in PIN_FRACTIONS if f"{f:g}" in points]
                ys = [points[f"{f:g}"].get(metric) for f in xs]
                figure.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+lines",
                        line={
                            "width": 1,
                            "color": DATASET_COLORS[dataset],
                            "dash": "solid" if features == "on" else "dash",
                        },
                        marker={"color": DATASET_COLORS[dataset], "size": 6},
                        name=_series_name(dataset, features),
                        legendgroup=cell,
                        showlegend=index == 0,
                    ),
                    row=row,
                    col=col,
                )
        figure.update_xaxes(
            title_text=r"$\text{Pinned Modality Fraction } f = m / ((d - 1) / 2)$"
            if row == 2
            else None,
            row=row,
            col=col,
        )
        figure.update_yaxes(title_text=label, row=row, col=col)
    _base_layout(figure, width=1300, height=900)
    figure.update_layout(
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.16}
    )
    return figure


def community(communities: dict[str, Any]) -> go.Figure:
    """Figure 5 (descriptive): advantage versus induced mean degree of subgraphs."""
    metrics = [
        ("log_loss_advantage", "Log-Loss Advantage (Nats)"),
        ("sampled_auc_advantage", "AUROC Advantage"),
    ]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[label.split(" (")[0] for _, label in metrics],
        horizontal_spacing=0.1,
    )
    for col, (metric, label) in enumerate(metrics, start=1):
        for cell, content in communities.items():
            dataset, features = cell.split("|")
            rows = content["rows"]
            figure.add_trace(
                go.Scatter(
                    x=[r["mean_degree"] for r in rows],
                    y=[r[metric] for r in rows],
                    mode="markers",
                    marker={
                        "color": DATASET_COLORS[dataset],
                        "symbol": "circle" if features == "on" else "diamond-open",
                        "size": 9,
                    },
                    name=_series_name(dataset, features),
                    legendgroup=cell,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        _zero_line(figure, 1, col)
        figure.update_xaxes(
            title_text="Induced Mean Degree of the Community", row=1, col=col
        )
        figure.update_yaxes(title_text=label, row=1, col=col)
    _base_layout(figure, width=1300, height=520)
    figure.update_layout(
        margin={"l": 80, "r": 30, "t": 60, "b": 150},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.28},
    )
    return figure


def _base_layout(figure: go.Figure, *, width: int, height: int) -> None:
    figure.update_layout(
        template=TEMPLATE_NAME,
        font={"family": FONT_FAMILY, "color": INK, "size": 13},
        width=width,
        height=height,
        plot_bgcolor="white",
        margin={"l": 80, "r": 30, "t": 60, "b": 130},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.16},
    )
    figure.update_xaxes(showline=True, linecolor=INK, ticks="outside")
    figure.update_yaxes(showline=True, linecolor=INK, gridcolor=GRID, ticks="outside")
    for annotation in figure.layout.annotations:  # type: ignore[union-attr]
        annotation.font = {"family": FONT_FAMILY, "size": 14, "color": INK}


def _series_name(dataset: str, features: str) -> str:
    return f"{DATASET_LABELS[dataset]}, Features {features.capitalize()}"


def _zero_line(figure: go.Figure, row: int, col: int) -> None:
    cast("Any", figure).add_hline(
        y=0, line={"color": INK, "width": 1, "dash": "dot"}, row=row, col=col
    )


def _asymmetric(mean: float, ci: list[float]) -> tuple[float, float]:
    return ci[1] - mean, mean - ci[0]


if __name__ == "__main__":
    main()
