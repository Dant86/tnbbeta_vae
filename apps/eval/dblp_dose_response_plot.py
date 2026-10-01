"""Plots the com-DBLP dose-response result: link-prediction AUC/AP vs. community count.

Usage:
    uv run python -m apps.eval.dblp_dose_response_plot [--results-dir dblp_results] \
        [--output dblp_dose_response.png]

Reads the 15 ``bridge_diagnostic_final.json`` files (3 families x 5 seeds) produced by
``apps.eval.dblp_bridge_diagnostic``'s ``link_prediction_by_community_count`` field (see
``writeup/results/dblp_bridge_diagnostic_2026-10-01.md``'s "dose-response" update) and
plots, per metric (AUC, AP), each family's mean with a shaded +/- 1 std band across
seeds against community count. A third panel shows each bucket's edge count (identical
across families and seeds, since the diagnostic's test split uses a fixed seed
regardless of which checkpoint's embeddings are being scored) -- shown once, as its own
panel, rather than duplicated beneath both metrics (which it would be identical under
anyway, since it doesn't depend on the metric) -- so a thin, low-power bucket at the
tail is visible directly on the chart, not just in the underlying JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import glob
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = ["build_figure", "load_results"]

_BUCKETS = ["0", "1", "2", "3", "4", "5+"]
_METRICS = [("auc", "AUC"), ("ap", "AP")]
_FAMILIES = [
    ("tnbbeta", "TNBBeta", "#1b9e77"),
    ("power_spherical", "Power Spherical", "#d95f02"),
    ("vmf", "vMF", "#7570b3"),
]

Results = dict[str, dict[str, dict[str, list[float]]]]


def main(argv: list[str] | None = None) -> None:
    """Builds and writes the dose-response figure.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("dblp_results"))
    parser.add_argument("--output", type=str, default="dblp_dose_response.png")
    args = parser.parse_args(argv)

    per_family, counts = load_results(args.results_dir)
    figure = build_figure(per_family, counts)
    figure.write_image(args.output, width=1400, height=750, scale=2)
    print(f"Wrote {args.output}")


def load_results(results_dir: Path) -> tuple[Results, dict[str, int]]:
    """Loads per-(family, metric, bucket) seed lists and each bucket's edge count.

    Args:
        results_dir: Directory holding ``dblp_<family>_d16_seed<seed>/
            bridge_diagnostic_final.json`` for each family/seed.

    Returns:
        ``(per_family, counts)``: ``per_family[family][metric][bucket]`` is the list of
        per-seed values found; ``counts[bucket]`` is that bucket's edge count (the same
        across every family/seed, since the diagnostic's test split doesn't depend on
        which checkpoint is being scored -- reported once per bucket, not per family).
    """
    per_family: Results = {
        family: {metric: defaultdict(list) for metric, _ in _METRICS}
        for family, _, _ in _FAMILIES
    }
    counts: dict[str, int] = {}
    for family, _, _ in _FAMILIES:
        pattern = str(
            results_dir / f"dblp_{family}_d16_seed*" / "bridge_diagnostic_final.json"
        )
        for path in sorted(glob.glob(pattern)):
            with Path(path).open() as handle:
                record = json.load(handle)
            by_count = record["link_prediction_by_community_count"]
            for bucket in _BUCKETS:
                for metric, _ in _METRICS:
                    per_family[family][metric][bucket].append(by_count[bucket][metric])
                counts[bucket] = by_count[bucket]["count"]
    return per_family, counts


def build_figure(per_family: Results, counts: dict[str, int]) -> go.Figure:
    """Builds the 3-panel (AUC, AP, edge count) dose-response figure.

    AUC and AP each get a line-plus-shaded-std-band panel (one line per family, mean
    across seeds); edge count is a third, independent panel rather than a strip
    repeated under each metric -- it doesn't depend on the metric, so showing it twice
    would just be the same bars drawn under AUC and under AP for no reason. All three
    panels use the same category ordering on their own x-axis, so they read in
    alignment with each other despite being separate subplots rather than a single
    shared-width row.

    Args:
        per_family: As returned by :func:`load_results`.
        counts: As returned by :func:`load_results`.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    figure = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.4, 0.2],
        horizontal_spacing=0.06,
        subplot_titles=[*(label for _, label in _METRICS), "edges per bucket"],
    )

    for col_index, (metric, _) in enumerate(_METRICS, start=1):
        for family, display_name, color in _FAMILIES:
            series = per_family[family][metric]
            means = np.array([np.mean(series[bucket]) for bucket in _BUCKETS])
            stds = np.array([np.std(series[bucket]) for bucket in _BUCKETS])
            _add_band_and_line(
                figure,
                row=1,
                col=col_index,
                x=_BUCKETS,
                mean=means,
                std=stds,
                color=color,
                name=display_name,
                show_legend=col_index == 1,
            )

    figure.add_trace(
        go.Bar(
            x=_BUCKETS,
            y=[counts[bucket] for bucket in _BUCKETS],
            marker={"color": "#999999"},
            showlegend=False,
            hovertemplate="%{y} edges<extra></extra>",
        ),
        row=1,
        col=3,
    )

    figure.update_xaxes(type="category", categoryorder="array", categoryarray=_BUCKETS)
    for col_index in (1, 2, 3):
        figure.update_xaxes(title_text="communities", row=1, col=col_index)
    figure.update_yaxes(title_text="score", row=1, col=1)
    figure.update_yaxes(title_text="edges", row=1, col=3)
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=(
            "com-DBLP link prediction vs. node community count (mean ± std, 5 seeds)"
        ),
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
    )
    return figure


def _add_band_and_line(
    figure: go.Figure,
    *,
    row: int,
    col: int,
    x: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    name: str,
    show_legend: bool,
) -> None:
    """Adds a shaded +/- 1 std band (two invisible boundary traces, one filled) and
    the solid mean line on top of it, to one subplot.

    Args:
        figure: Figure to add the traces to.
        row: Subplot row.
        col: Subplot column.
        x: Category labels (community-count buckets).
        mean: Per-bucket mean, same length as ``x``.
        std: Per-bucket std, same length as ``x``.
        color: Line color (hex); the band uses the same hue at low alpha.
        name: Legend label.
        show_legend: Whether this family's entry should appear in the legend (only the
            first subplot needs to show it, since the legend is shared).
    """
    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean - std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean + std,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_to_rgba(color, 0.18),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            line={"color": color, "width": 2.5},
            marker={"size": 6},
            name=name,
            legendgroup=name,
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )


def _to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a ``#rrggbb`` hex color to an ``rgba(...)`` string at ``alpha``."""
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


if __name__ == "__main__":
    main(sys.argv[1:])
