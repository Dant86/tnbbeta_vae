"""Combined main-stage bars: the revision's four graphs plus the three dense graphs.

Stitches ``main["cells"]`` of the two ``analysis.json`` files (``real_graph`` and
``real_graph_dense`` under ``$TNBBETA_CHECKPOINT_DIR``) into one pair of panels per
metric (features on, features off) with the graphs ordered by mean degree, in the style
of the paper's ``features_onoff_bars`` figure. Bars are the seed means with the 95%
seed-bootstrap intervals of the analysis (not the paper's +-1 SD).

Usage (login node: kaleido fails on compute nodes):
    uv run python -m apps.real_graph.combined_bars --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.real_graph.figures import (
    DATASET_LABELS,
    FAMILY_STYLE,
    FEATURE_TITLES,
    GRID,
    INK,
)
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

__all__ = ["build_figure", "main"]

# Ordered by mean degree (realised, after the studies' preprocessing).
MEAN_DEGREE = {
    "citeseer": 2.74,
    "cora": 3.90,
    "pubmed": 4.50,
    "mag_cs": 8.8,
    "amazon_photo": 31.13,
    "amazon_computers": 35.76,
    "squirrel": 76.27,
}
FAMILIES = ("gaussian", "vmf", "tnbbeta", "power_spherical")
FEATURES = ("on", "off")
METRICS = (("sampled_auc", "Sampled AUROC"), ("log_loss", "Test Log Loss"))
DENSE_FROM = 24.0  # the synthetic study's degree threshold for the dense region
OUTPUT_NAME = "combined_real_graph_bars_2026-11-04.png"


def main(argv: list[str] | None = None) -> None:
    """Renders the combined figure.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=Path("writeup/results/figures"))
    args = parser.parse_args(argv)
    cells: dict[str, Any] = {}
    for study in ("real_graph", "real_graph_dense"):
        path = checkpoint_dir() / study / "analysis.json"
        cells.update(json.loads(path.read_text())["main"]["cells"])
    figure = build_figure(cells)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / OUTPUT_NAME
    figure.write_image(str(out), scale=3)
    print(f"Wrote {out}")


def build_figure(cells: dict[str, Any]) -> go.Figure:
    """Grouped bars: rows are metrics, columns features on/off, groups are graphs.

    Args:
        cells: ``"<dataset>|<features>"`` to the analysis cell
            (``cell["families"][family][metric]`` with ``mean`` and ``ci``).

    Returns:
        The figure (not yet written).
    """
    datasets = [d for d in MEAN_DEGREE if f"{d}|on" in cells and f"{d}|off" in cells]
    ticks = [
        f"{DATASET_LABELS[d].replace(' ', '<br>')}<br>{MEAN_DEGREE[d]:.1f}"
        for d in datasets
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        column_titles=[FEATURE_TITLES[f] for f in FEATURES],
        horizontal_spacing=0.08,
        vertical_spacing=0.07,
    )
    for row, (metric, label) in enumerate(METRICS, start=1):
        row_lows, row_highs = [], []
        for col, features in enumerate(FEATURES, start=1):
            for family in FAMILIES:
                name, color = FAMILY_STYLE[family]
                stats = [
                    cells[f"{d}|{features}"]["families"][family][metric]
                    for d in datasets
                ]
                row_lows += [s["ci"][0] for s in stats]
                row_highs += [s["ci"][1] for s in stats]
                figure.add_trace(
                    go.Bar(
                        x=ticks,
                        y=[s["mean"] for s in stats],
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": [s["ci"][1] - s["mean"] for s in stats],
                            "arrayminus": [s["mean"] - s["ci"][0] for s in stats],
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
        pad = 0.08 * (max(row_highs) - min(row_lows))
        for col in (1, 2):
            figure.update_yaxes(
                range=[min(row_lows) - pad, max(row_highs) + pad],
                title_text=f"$\\text{{{label}}}$" if col == 1 else None,
                row=row,
                col=col,
            )
    figure.update_layout(
        template=TEMPLATE_NAME,
        title={
            "text": "Link Prediction Across Seven Real Graphs, Ordered by Mean Degree",
            "x": 0.5,
        },
        font={"family": FONT_FAMILY, "color": INK, "size": 13},
        width=1500,
        height=900,
        plot_bgcolor="white",
        margin={"l": 80, "r": 30, "t": 90, "b": 120},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.14},
        barmode="group",
        bargap=0.2,
    )
    figure.update_xaxes(
        showline=True, linecolor=INK, ticks="outside", tickangle=0, tickfont_size=12
    )
    figure.update_xaxes(showticklabels=False, row=1)
    figure.update_yaxes(showline=True, linecolor=INK, gridcolor=GRID, ticks="outside")
    split = sum(MEAN_DEGREE[d] < DENSE_FROM for d in datasets) - 0.5
    for row in (1, 2):
        for col in (1, 2):
            cast("Any", figure).add_vline(
                x=split,
                line={"color": INK, "width": 1, "dash": "dot"},
                row=row,
                col=col,
            )
    for annotation in figure.layout.annotations:  # type: ignore[union-attr]
        annotation.font = {"family": FONT_FAMILY, "size": 14, "color": INK}
    return figure


if __name__ == "__main__":
    main()
