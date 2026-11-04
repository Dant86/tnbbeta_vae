"""Edge-thinning line charts: test log loss per family against edges kept.

For the dense study (Amazon Computers, Amazon Photo, Squirrel), averages each family's
test log loss over the three graphs within each seed, then plots the mean over the ten
seeds with a 95% seed-bootstrap interval. Panels are features on and off.

Usage (login node: kaleido fails on compute nodes):
    uv run python -m apps.real_graph.thinning_lines --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.real_graph.analyze import metric_arrays
from apps.real_graph.figures import FAMILY_STYLE, FEATURE_TITLES, GRID, INK
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

__all__ = ["build_figure", "main"]

Array = npt.NDArray[np.float64]

DATASETS = ("amazon_computers", "amazon_photo", "squirrel")
FAMILIES = ("gaussian", "vmf", "tnbbeta", "power_spherical")
FEATURES = ("on", "off")
KEEPS = (0.3, 0.5, 0.7, 1.0)
BOOTSTRAPS = 4000
OUTPUT_NAME = "dense_thinning_lines_2026-11-04.png"


def main(argv: list[str] | None = None) -> None:
    """Renders the thinning line charts.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=Path("writeup/results/figures"))
    args = parser.parse_args(argv)
    arrays = metric_arrays(_load("main") + _load("thin"))
    figure = build_figure(arrays)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / OUTPUT_NAME
    figure.write_image(str(out), scale=3)
    print(f"Wrote {out}")


def build_figure(arrays: dict[tuple[Any, ...], dict[str, Array]]) -> go.Figure:
    """One line per family, log loss against the fraction of edges kept.

    Args:
        arrays: ``metric_arrays`` output: ``(dataset, features, family, keep,
            community, pin)`` to per-seed test metrics.

    Returns:
        The figure (not yet written).
    """
    figure = make_subplots(
        rows=1,
        cols=2,
        column_titles=[FEATURE_TITLES[f] for f in FEATURES],
        horizontal_spacing=0.06,
        shared_yaxes=True,
    )
    rng = np.random.default_rng(0)
    for col, features in enumerate(FEATURES, start=1):
        for family in FAMILIES:
            name, color = FAMILY_STYLE[family]
            means, lows, highs = [], [], []
            for keep in KEEPS:
                per_seed = np.mean(
                    [
                        _group(arrays, dataset, features, family, keep)["log_loss"]
                        for dataset in DATASETS
                    ],
                    axis=0,
                )
                mean, low, high = _bootstrap(per_seed, rng)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            figure.add_trace(
                go.Scatter(
                    x=list(KEEPS),
                    y=means,
                    mode="lines+markers",
                    line={"width": 1.5, "color": color},
                    marker={"color": color, "size": 7},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [h - m for h, m in zip(highs, means, strict=True)],
                        "arrayminus": [
                            m - lo for m, lo in zip(means, lows, strict=True)
                        ],
                        "color": color,
                        "thickness": 1,
                        "width": 3,
                    },
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
    figure.update_layout(
        template=TEMPLATE_NAME,
        font={"family": FONT_FAMILY, "color": INK, "size": 13},
        width=1300,
        height=520,
        plot_bgcolor="white",
        margin={"l": 80, "r": 30, "t": 60, "b": 70},
        legend={
            "x": 0.99,
            "y": 0.97,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    figure.update_xaxes(
        title_text="$\\text{Fraction of Edges Kept}$",
        tickvals=list(KEEPS),
        showline=True,
        linecolor=INK,
        ticks="outside",
    )
    figure.update_yaxes(showline=True, linecolor=INK, gridcolor=GRID, ticks="outside")
    figure.update_yaxes(title_text="$\\text{Test Log Loss}$", col=1)
    for annotation in figure.layout.annotations:  # type: ignore[union-attr]
        annotation.font = {"family": FONT_FAMILY, "size": 14, "color": INK}
    return figure


def _load(stage: str) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor
    import json

    paths = sorted(
        (checkpoint_dir() / "real_graph_dense" / stage / "runs").glob("*.json")
    )
    with ThreadPoolExecutor(16) as pool:
        records = list(pool.map(lambda path: json.loads(path.read_text()), paths))
    return [record for record in records if "test" in record]


def _group(
    arrays: dict[tuple[Any, ...], dict[str, Array]],
    dataset: str,
    features: str,
    family: str,
    keep: float,
) -> dict[str, Array]:
    # Full-density runs come from the main stage, which records no ``keep``.
    key_keep = None if keep == 1.0 else keep
    return arrays[(dataset, features, family, key_keep, None, None)]


def _bootstrap(values: Array, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values))).mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


if __name__ == "__main__":
    main()
