"""Scatter of per-node degree vs. TNBBeta's m, one color per SBM p_out level, with a
per-level line of best fit -- visualizes the degree-vs-m correlation reported in
``writeup/results/sbm_degree_correlation_and_weight_transfer_2026-10-10.md`` (and
its 2026-10-15 reproduction) as a figure instead of a bare correlation coefficient.

Usage:
    uv run python -m apps.eval.sbm_degree_vs_m_plot --input PATH [--output PATH]

``--input`` is the raw per-node dump written by the (uncommitted, ad hoc)
degree-correlation script's ``degree_vs_m_raw.json`` output: ``p_out -> {"degree":
[...], "m": [...]}``, one entry per node with both a valid degree and a valid m.
This script is not given the raw data hardcoded, unlike
``apps.eval.mnist_table1_lines``/``apps.eval.features_onoff_bars``: there are ~500
points per p_out (2500 total) here, too much to transcribe by hand, and the whole
point of keeping the degree-correlation script itself ad hoc/uncommitted is that its
raw output isn't meant to be committed either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr

from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = ["build_figure", "load_raw", "main"]

# Ordered low -> high p_out: a "heat" progression (cooler = cleaner communities,
# hotter = more cross-community mixing), not an arbitrary categorical palette,
# since p_out is itself an ordered quantity.
_P_OUT_ORDER = ["0.005", "0.01", "0.02", "0.05", "0.1"]
_P_OUT_COLORS = {
    "0.005": "#0d0887",
    "0.01": "#7e03a8",
    "0.02": "#cc4778",
    "0.05": "#f89441",
    "0.1": "#f0f921",
}


def main(argv: list[str] | None = None) -> None:
    """Builds the degree-vs-m scatter and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, default="sbm_degree_vs_m.png")
    args = parser.parse_args(argv)

    raw = load_raw(args.input)
    figure = build_figure(raw)
    figure.write_image(args.output, width=900, height=650, scale=2)
    print(f"Wrote {args.output}")


def load_raw(path: Path) -> dict[str, dict[str, list[float]]]:
    """Loads the ad hoc degree-correlation script's ``degree_vs_m_raw.json``.

    Args:
        path: Path to the JSON file (``p_out -> {"degree": [...], "m": [...]}``).

    Returns:
        The parsed mapping, unchanged.
    """
    return json.loads(path.read_text())


def build_figure(
    raw: dict[str, dict[str, list[float]]],
    *,
    p_out_order: list[str] = _P_OUT_ORDER,
    colors: dict[str, str] = _P_OUT_COLORS,
) -> go.Figure:
    """Builds the degree-vs-m scatter, one color per p_out, with a line of best fit.

    Args:
        raw: ``p_out -> {"degree": [...], "m": [...]}``, as returned by
            :func:`load_raw`. A ``p_out_order`` entry absent from ``raw`` is
            skipped, so callers don't need to pre-filter.
        p_out_order: Legend/plot order for the p_out levels to draw; defaults to
            module-level ``_P_OUT_ORDER``, overridable for tests.
        colors: ``p_out -> hex color``; defaults to module-level ``_P_OUT_COLORS``,
            overridable for tests.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    figure = go.Figure()
    for p_out in p_out_order:
        if p_out not in raw:
            continue
        degree = np.array(raw[p_out]["degree"])
        m = np.array(raw[p_out]["m"])
        color = colors[p_out]
        r, _ = pearsonr(degree, m)

        figure.add_trace(
            go.Scatter(
                x=degree,
                y=m,
                mode="markers",
                marker={"color": color, "size": 5, "opacity": 0.35},
                name=f"p_out={p_out} (r={r:.2f})",
                legendgroup=p_out,
                showlegend=True,
            )
        )

        slope, intercept = np.polyfit(degree, m, 1)
        x_fit = np.array([degree.min(), degree.max()])
        figure.add_trace(
            go.Scatter(
                x=x_fit,
                y=slope * x_fit + intercept,
                mode="lines",
                line={"color": color, "width": 3},
                legendgroup=p_out,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text="SBM dose-response: per-node degree vs. TNBBeta's m",
        xaxis_title="node degree",
        yaxis_title="m = epsilon - (d-1)/2",
        legend_title_text="p_out",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
