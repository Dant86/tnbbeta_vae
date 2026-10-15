"""Grouped bar-chart summary of the features-on/off x dimension sweep's headline
results (AUC and AP, with vs. without node features), replacing a "massive series
of tables" with one 4-panel figure a reader can take in at a glance.

Usage:
    uv run python -m apps.eval.features_onoff_bars [--output PATH]

Data is a hardcoded transcription of the "Results" table in
``writeup/results/features_onoff_dimension_sweep_2026-10-09.md`` -- the
validation-selected configuration per (dataset variant, family), mean +- std over
seeds. This is what that write-up's top-level table already is: one number per
cell, aggregated over the dimension sweep. The much bigger by-dimension breakdown
tables underneath it are not reproduced here; collapsing *those* into one glance-able
figure is a different, harder problem than this script solves.

Layout: 2 rows (AUC, AP) x 2 columns (with features, without features), each panel
a grouped (not stacked) bar chart -- one bar per posterior family, per dataset.
Grouped rather than stacked because AUC/AP aren't additive across families; summing
them into one bar per dataset would produce a meaningless total and make the
individual families harder to compare, not easier. The y-axis is cropped (not
started at 0): every value here sits in a tight, high range (70-97), so a
zero-based axis would compress nearly every real difference into the chart's top
30% -- deliberately trading the "bar area is proportional to value" convention for
actually being able to see the differences the figure exists to show.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = ["build_figure", "main"]

Metric = Literal["auc", "ap"]
Condition = Literal["on", "off"]

_DATASETS = ["cora", "citeseer", "pubmed", "mag_cs"]
_DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "mag_cs": "MAG CS",
}
_FAMILIES = [
    ("gaussian", "N-VGAE", "#7f7f7f"),
    ("vmf", "S-VGAE (vMF)", "#1f77b4"),
    ("tnbbeta", "TNBBeta-VGAE", "#d62728"),
    ("power_spherical", "Power Spherical-VGAE", "#2ca02c"),
]
# Every value in the table sits in [70, 96.7]; a little padding on each side.
_Y_RANGE = [65.0, 100.0]
_PANELS: list[tuple[Metric, Condition, str]] = [
    ("auc", "on", "AUC — with features"),
    ("auc", "off", "AUC — without features"),
    ("ap", "on", "AP — with features"),
    ("ap", "off", "AP — without features"),
]

Cell = tuple[float, float]  # (mean, std)
# metric -> condition -> dataset -> family -> (mean, std)
_DATA: dict[Metric, dict[Condition, dict[str, dict[str, Cell]]]] = {
    "auc": {
        "on": {
            "cora": {
                "gaussian": (93.2, 1.0),
                "vmf": (93.2, 0.5),
                "tnbbeta": (93.7, 0.8),
                "power_spherical": (93.7, 0.7),
            },
            "citeseer": {
                "gaussian": (91.1, 1.1),
                "vmf": (94.0, 0.8),
                "tnbbeta": (93.1, 0.4),
                "power_spherical": (93.5, 0.8),
            },
            "pubmed": {
                "gaussian": (95.9, 0.3),
                "vmf": (95.1, 0.3),
                "tnbbeta": (95.4, 0.3),
                "power_spherical": (95.2, 0.2),
            },
            "mag_cs": {
                "gaussian": (96.7, 0.1),
                "vmf": (96.6, 0.1),
                "tnbbeta": (95.3, 0.2),
                "power_spherical": (96.5, 0.2),
            },
        },
        "off": {
            "cora": {
                "gaussian": (84.8, 1.0),
                "vmf": (78.1, 1.1),
                "tnbbeta": (84.0, 1.6),
                "power_spherical": (74.7, 2.1),
            },
            "citeseer": {
                "gaussian": (78.6, 1.3),
                "vmf": (74.8, 2.0),
                "tnbbeta": (77.0, 2.1),
                "power_spherical": (70.0, 0.9),
            },
            "pubmed": {
                "gaussian": (85.3, 1.0),
                "vmf": (82.3, 0.4),
                "tnbbeta": (82.7, 0.6),
                "power_spherical": (81.3, 0.6),
            },
            "mag_cs": {
                "gaussian": (94.3, 0.3),
                "vmf": (89.4, 0.5),
                "tnbbeta": (94.3, 0.3),
                "power_spherical": (86.1, 0.3),
            },
        },
    },
    "ap": {
        "on": {
            "cora": {
                "gaussian": (93.9, 1.0),
                "vmf": (93.6, 0.7),
                "tnbbeta": (94.0, 0.9),
                "power_spherical": (93.8, 0.7),
            },
            "citeseer": {
                "gaussian": (92.1, 1.0),
                "vmf": (94.4, 0.8),
                "tnbbeta": (93.8, 0.5),
                "power_spherical": (94.0, 0.9),
            },
            "pubmed": {
                "gaussian": (96.0, 0.2),
                "vmf": (94.9, 0.4),
                "tnbbeta": (95.2, 0.3),
                "power_spherical": (94.9, 0.2),
            },
            "mag_cs": {
                "gaussian": (96.7, 0.2),
                "vmf": (96.5, 0.2),
                "tnbbeta": (95.0, 0.2),
                "power_spherical": (96.5, 0.2),
            },
        },
        "off": {
            "cora": {
                "gaussian": (88.4, 0.7),
                "vmf": (81.7, 0.7),
                "tnbbeta": (87.6, 1.2),
                "power_spherical": (78.2, 2.1),
            },
            "citeseer": {
                "gaussian": (83.8, 1.0),
                "vmf": (80.1, 1.5),
                "tnbbeta": (82.7, 1.8),
                "power_spherical": (75.6, 1.1),
            },
            "pubmed": {
                "gaussian": (85.5, 0.8),
                "vmf": (85.8, 0.4),
                "tnbbeta": (86.7, 0.5),
                "power_spherical": (84.3, 0.8),
            },
            "mag_cs": {
                "gaussian": (95.0, 0.2),
                "vmf": (90.2, 0.5),
                "tnbbeta": (95.5, 0.2),
                "power_spherical": (88.7, 0.2),
            },
        },
    },
}


def main(argv: list[str] | None = None) -> None:
    """Builds the 4-panel bar chart and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="features_onoff_bars.png")
    args = parser.parse_args(argv)

    figure = build_figure(_DATA)
    figure.write_image(args.output, width=1400, height=1000, scale=2)
    print(f"Wrote {args.output}")


def build_figure(
    data: dict[Metric, dict[Condition, dict[str, dict[str, Cell]]]],
    *,
    panels: list[tuple[Metric, Condition, str]] = _PANELS,
    y_range: list[float] = _Y_RANGE,
    families: list[tuple[str, str, str]] = _FAMILIES,
    datasets: list[str] = _DATASETS,
) -> go.Figure:
    """Builds the 2x2 (AUC/AP x with/without features) grouped bar chart.

    Args:
        data: ``metric -> condition -> dataset -> family -> (mean, std)``, as
            module-level ``_DATA``.
        panels: ``(metric, condition, subplot title)`` tuples, in row-major order
            over a 2x2 grid; defaults to module-level ``_PANELS``, overridable for
            tests.
        y_range: Shared, cropped y-axis range for every panel; defaults to
            module-level ``_Y_RANGE``, overridable for tests.
        families: ``(data key, legend label, color)`` tuples; defaults to
            module-level ``_FAMILIES``, overridable for tests.
        datasets: Dataset keys (must be present in ``_DATASET_LABELS``); defaults to
            module-level ``_DATASETS``, overridable for tests.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[title for *_, title in panels],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )

    for panel_index, (metric, condition, _) in enumerate(panels):
        row, col = panel_index // 2 + 1, panel_index % 2 + 1
        per_dataset = data[metric][condition]
        for family_key, display_name, color in families:
            means = [per_dataset[d][family_key][0] for d in datasets]
            stds = [per_dataset[d][family_key][1] for d in datasets]
            figure.add_trace(
                go.Bar(
                    x=[_DATASET_LABELS[d] for d in datasets],
                    y=means,
                    error_y={"type": "data", "array": stds, "visible": True},
                    marker={"color": color},
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=panel_index == 0,
                ),
                row=row,
                col=col,
            )

    figure.update_yaxes(range=y_range, title_text="%")
    figure.update_layout(
        template=TEMPLATE_NAME,
        barmode="group",
        title_text="Link prediction AUC/AP, with vs. without node features",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.1},
        margin={"l": 50, "r": 20, "t": 70, "b": 80},
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
