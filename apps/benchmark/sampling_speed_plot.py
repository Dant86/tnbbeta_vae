"""Heatmap of the vMF/TNBBeta sampling speedup, from sampling_speed.py's JSON.

Usage:
    uv run python -m apps.benchmark.sampling_speed_plot [--input PATH] [--output PATH]

Reads the JSON ``apps.benchmark.sampling_speed`` writes and plots the
``vmf_over_tnbbeta`` speedup ratio as a heatmap grid, one panel per batch size,
ambient dimension on the y-axis and concentration level on the x-axis -- the
fastest way to see all three swept variables' effect on "how many times slower
is vMF" at once. Every cell is annotated with its own ratio in addition to its
color, and all panels share one colorbar so they're directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tnbbeta_vae.paths import runs_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME

_CONCENTRATION_ORDER = ["weak", "moderate", "strong", "extreme"]


def main(argv: list[str] | None = None) -> None:
    """Builds and writes the heatmap grid.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    input_path = (
        Path(args.input) if args.input else runs_dir() / "sampling_speed_benchmark.json"
    )
    records = json.loads(input_path.read_text())

    output = args.output or str(runs_dir() / "sampling_speed_heatmap.png")
    _figure(records).write_image(output, width=1650, height=560, scale=2)
    print(f"Wrote {output}")


def _figure(records: list[dict]) -> go.Figure:
    """Builds the multi-panel heatmap figure."""
    dims = sorted({r["dim"] for r in records})
    batch_sizes = sorted({r["batch_size"] for r in records})
    concentrations = [
        c for c in _CONCENTRATION_ORDER if any(r["concentration"] == c for r in records)
    ]
    by_cell = {(r["dim"], r["batch_size"], r["concentration"]): r for r in records}

    ratios = [r["vmf_over_tnbbeta"] for r in records]
    zmin, zmax = min(ratios), max(ratios)

    figure = make_subplots(
        rows=1,
        cols=len(batch_sizes),
        subplot_titles=[f"batch size {b}" for b in batch_sizes],
        horizontal_spacing=0.04,
    )
    kappa_by_concentration = {r["concentration"]: r["vmf_kappa"] for r in records}
    concentration_labels = [
        f"{c}<br>({kappa_by_concentration[c]:g})" for c in concentrations
    ]
    for column, batch_size in enumerate(batch_sizes, start=1):
        z = [
            [by_cell[dim, batch_size, c]["vmf_over_tnbbeta"] for c in concentrations]
            for dim in dims
        ]
        text = [[f"{value:.1f}×" for value in row] for row in z]
        figure.add_trace(
            go.Heatmap(
                z=z,
                x=concentration_labels,
                y=[str(dim) for dim in dims],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 13},
                coloraxis="coloraxis",
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(
        title_text="concentration level (kappa / epsilon)", type="category"
    )
    # Force category type: dimension labels look numeric ("2", "5", ..., "100"), so
    # without this plotly spaces rows by numeric value instead of giving each one an
    # equal-height row, squeezing the low-dimension (and most extreme-ratio) rows
    # into an illegible sliver.
    figure.update_yaxes(type="category")
    figure.update_yaxes(title_text="ambient dimension", col=1)
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=(
            "vMF sampling time, as a multiple of TNBBeta's (GPU, 30 timed calls/cell)"
        ),
        coloraxis={
            "colorscale": "Plasma",
            "cmin": zmin,
            "cmax": zmax,
            "colorbar": {"title": "speedup", "ticksuffix": "×"},
        },
        margin={"l": 60, "r": 20, "t": 70, "b": 60},
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
