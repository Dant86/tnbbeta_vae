"""Plots real per-step training trajectories of posterior parameters.

Usage:
    uv run python -m apps.eval.posterior_trajectories [--runs-dir cluster_runs] \
        [--output-dir writeup/results] [--max-points 500] [--dims 2 5 10 20 40]

Reads straight from ``runs/<run_id>/metrics.jsonl`` (``RunLogger``'s per-step log,
written by every training run, not just the final checkpoint's diagnostics) to see
what (p, q, epsilon) or kappa a real training run actually converges to, rather than
inferring it only from synthetic limits -- the empirical anchor
``writeup/weekly_markdown_summaries/week_2/week_2_plan.md`` calls for under "Also use
the real training data, not just synthetic limits."

Four figures, each written as a plainly-named PNG (the accompanying dated
``writeup/results/*.md`` carries the provenance -- which run names, which
``cluster_runs/`` snapshot, which command):

* ``tnbbeta_posterior_trajectories.png``: p, q, epsilon vs. training step, one column
  per dimension in ``mnist_tnbs_d<dim>_seed<seed>`` (paper-convention sweep).
* ``kappa_trajectories.png``: vMF / Power Spherical / vMF (kappa-init) kappa vs.
  training step, one column per dimension
  (``mnist_{vmfs,pss,vmfks}_d<dim>_seed<seed>``).
* ``fixed_epsilon_ablation_trajectories.png``: q vs. training step for the fixed-epsilon
  ablation (``mnistfix_eps{0p5,1p0,1p5}_d5_seed<seed>``), d=5 only.
* ``fixed_mu_ablation_trajectories.png``: p and q vs. training step for the fixed-mean-
  direction ablation (``mnistfixmu_tnb_d5_seed<seed>``), d=5 only.

Each run's metrics.jsonl logs every training step (up to ~400k lines) -- far more
than a line chart needs -- so each series is decimated to at most ``--max-points``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = [
    "build_ablation_figures",
    "build_kappa_figure",
    "build_tnbbeta_figure",
    "discover_seeds",
    "read_metric_series",
]

_DEFAULT_DIMS = [2, 5, 10, 20, 40]
_COLORS = {
    "p": "#1b9e77",
    "q": "#d95f02",
    "epsilon": "#7570b3",
    "vmfs": "#1b9e77",
    "pss": "#d95f02",
    "vmfks": "#7570b3",
    "eps0.5": "#e7298a",
    "eps1.0": "#1b9e77",
    "eps1.5": "#7570b3",
}


def main(argv: list[str] | None = None) -> None:
    """Builds and writes all four trajectory figures.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("cluster_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("writeup/results"))
    parser.add_argument("--max-points", type=int, default=500)
    parser.add_argument("--dims", type=int, nargs="+", default=_DEFAULT_DIMS)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tnbbeta_figure = build_tnbbeta_figure(args.runs_dir, args.max_points, args.dims)
    _write(
        tnbbeta_figure,
        args.output_dir / "tnbbeta_posterior_trajectories.png",
        1500,
        900,
    )

    kappa_figure = build_kappa_figure(args.runs_dir, args.max_points, args.dims)
    _write(kappa_figure, args.output_dir / "kappa_trajectories.png", 1500, 420)

    eps_figure, mu_figure = build_ablation_figures(args.runs_dir, args.max_points)
    _write(
        eps_figure,
        args.output_dir / "fixed_epsilon_ablation_trajectories.png",
        800,
        500,
    )
    _write(mu_figure, args.output_dir / "fixed_mu_ablation_trajectories.png", 1000, 500)


def discover_seeds(runs_dir: Path, prefix: str) -> list[int]:
    """Returns the sorted seed numbers available for ``<prefix>_seed<seed>`` runs.

    Args:
        runs_dir: Directory holding one subdirectory per run.
        prefix: Run name without its ``_seed<seed>`` suffix.

    Returns:
        Sorted seed numbers found; empty if none exist.
    """
    seeds = []
    for path in glob.glob(str(runs_dir / f"{prefix}_seed*")):
        suffix = Path(path).name.removeprefix(f"{prefix}_seed")
        if suffix.isdigit():
            seeds.append(int(suffix))
    return sorted(seeds)


def read_metric_series(
    path: Path, field: str, max_points: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Reads one field's trajectory from a run's ``metrics.jsonl``, decimated.

    Args:
        path: Path to ``metrics.jsonl``.
        field: The field to extract, e.g. ``"posterior_p_mean"``.
        max_points: Maximum points to return, evenly decimated across the run.

    Returns:
        ``(steps, values)``, both shape ``(n,)`` with ``n <= max_points``. Both
        empty if ``path`` doesn't exist or ``field`` never appears in it.
    """
    if not path.exists():
        return np.array([]), np.array([])
    steps: list[float] = []
    values: list[float] = []
    with path.open() as handle:
        for line in handle:
            if field not in line:
                continue
            record = json.loads(line)
            if field in record:
                steps.append(record["step"])
                values.append(record[field])
    if not steps:
        return np.array([]), np.array([])
    steps_array, values_array = np.array(steps), np.array(values)
    if len(steps_array) > max_points:
        indices = np.linspace(0, len(steps_array) - 1, max_points).astype(int)
        steps_array, values_array = steps_array[indices], values_array[indices]
    return steps_array, values_array


def build_tnbbeta_figure(
    runs_dir: Path, max_points: int, dims: Sequence[int] = _DEFAULT_DIMS
) -> go.Figure:
    """Builds the 3 (p, q, epsilon) x len(dims) TNBBeta trajectory grid.

    Args:
        runs_dir: Directory holding one subdirectory per run.
        max_points: Passed to :func:`read_metric_series`.
        dims: Dimensions to show as columns, reading ``mnist_tnbs_d<dim>_seed<seed>``.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    rows = [
        ("posterior_p_mean", "p"),
        ("posterior_q_mean", "q"),
        ("posterior_epsilon_mean", "epsilon"),
    ]
    figure = make_subplots(
        rows=len(rows),
        cols=len(dims),
        column_titles=[f"d={dim}" for dim in dims],
        row_titles=[label for _, label in rows],
        vertical_spacing=0.06,
        horizontal_spacing=0.03,
    )
    for row_index, (field, label) in enumerate(rows, start=1):
        for col_index, dim in enumerate(dims, start=1):
            prefix = f"mnist_tnbs_d{dim}"
            for seed in discover_seeds(runs_dir, prefix):
                steps, series = read_metric_series(
                    runs_dir / f"{prefix}_seed{seed}" / "metrics.jsonl",
                    field,
                    max_points,
                )
                if len(steps) == 0:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=steps,
                        y=series,
                        mode="lines",
                        line={"color": _COLORS[label], "width": 1},
                        opacity=0.6,
                        showlegend=False,
                    ),
                    row=row_index,
                    col=col_index,
                )
    for col_index in range(1, len(dims) + 1):
        figure.update_xaxes(title_text="step", row=len(rows), col=col_index)
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text=(
            "TNBBeta posterior parameters over real training (mnist_tnbs, 5 seeds/dim)"
        ),
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
    )
    return figure


def build_kappa_figure(
    runs_dir: Path, max_points: int, dims: Sequence[int] = _DEFAULT_DIMS
) -> go.Figure:
    """Builds the 1 x len(dims) vMF / Power Spherical / vMF(kappa-init) kappa grid.

    Args:
        runs_dir: Directory holding one subdirectory per run.
        max_points: Passed to :func:`read_metric_series`.
        dims: Dimensions to show as columns.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    models = ["vmfs", "pss", "vmfks"]
    figure = make_subplots(
        rows=1,
        cols=len(dims),
        column_titles=[f"d={dim}" for dim in dims],
        horizontal_spacing=0.03,
    )
    for col_index, dim in enumerate(dims, start=1):
        for model in models:
            prefix = f"mnist_{model}_d{dim}"
            for seed_index, seed in enumerate(discover_seeds(runs_dir, prefix)):
                steps, series = read_metric_series(
                    runs_dir / f"{prefix}_seed{seed}" / "metrics.jsonl",
                    "posterior_kappa_mean",
                    max_points,
                )
                if len(steps) == 0:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=steps,
                        y=series,
                        mode="lines",
                        line={"color": _COLORS[model], "width": 1},
                        opacity=0.6,
                        name=model,
                        legendgroup=model,
                        showlegend=(col_index == 1 and seed_index == 0),
                    ),
                    row=1,
                    col=col_index,
                )
    figure.update_yaxes(title_text="kappa", row=1, col=1)
    for col_index in range(1, len(dims) + 1):
        figure.update_xaxes(title_text="step", row=1, col=col_index)
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text="vMF / Power Spherical / vMF (kappa-init) kappa over real training",
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
    )
    return figure


def build_ablation_figures(
    runs_dir: Path, max_points: int
) -> tuple[go.Figure, go.Figure]:
    """Builds the fixed-epsilon and fixed-mean-direction ablation trajectory figures.

    Both ablations were only run at d=5 (3 seeds each), so neither takes a ``dims``
    argument.

    Args:
        runs_dir: Directory holding one subdirectory per run.
        max_points: Passed to :func:`read_metric_series`.

    Returns:
        ``(fixed_epsilon_figure, fixed_mu_figure)``.
    """
    eps_figure = go.Figure()
    eps_groups = [
        ("mnistfix_eps0p5_d5", "eps0.5", "epsilon=0.5 (crashes)"),
        ("mnistfix_eps1p0_d5", "eps1.0", "epsilon=1.0"),
        ("mnistfix_eps1p5_d5", "eps1.5", "epsilon=1.5"),
    ]
    for prefix, color_key, label in eps_groups:
        for seed_index, seed in enumerate(discover_seeds(runs_dir, prefix)):
            steps, series = read_metric_series(
                runs_dir / f"{prefix}_seed{seed}" / "metrics.jsonl",
                "posterior_q_mean",
                max_points,
            )
            if len(steps) == 0:
                continue
            eps_figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=series,
                    mode="lines",
                    line={"color": _COLORS[color_key], "width": 1.5},
                    opacity=0.8,
                    name=label,
                    legendgroup=label,
                    showlegend=(seed_index == 0),
                )
            )
            _add_endpoint_marker(eps_figure, steps, series, _COLORS[color_key])
    eps_figure.update_layout(
        template=TEMPLATE_NAME,
        title_text="Fixed-epsilon ablation: q substitutes for the lost epsilon (d=5)",
        xaxis_title="training step",
        yaxis_title="q",
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )

    mu_figure = make_subplots(rows=1, cols=2, subplot_titles=["p", "q"])
    mu_prefix = "mnistfixmu_tnb_d5"
    mu_seeds = discover_seeds(runs_dir, mu_prefix)
    for field, color_key, col in [
        ("posterior_p_mean", "p", 1),
        ("posterior_q_mean", "q", 2),
    ]:
        for seed in mu_seeds:
            steps, series = read_metric_series(
                runs_dir / f"{mu_prefix}_seed{seed}" / "metrics.jsonl",
                field,
                max_points,
            )
            if len(steps) == 0:
                continue
            mu_figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=series,
                    mode="lines",
                    line={"color": _COLORS[color_key], "width": 1.5},
                    opacity=0.8,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            _add_endpoint_marker(
                mu_figure, steps, series, _COLORS[color_key], row=1, col=col
            )
    mu_figure.update_xaxes(title_text="training step")
    mu_figure.update_layout(
        template=TEMPLATE_NAME,
        title_text="Fixed-mean-direction ablation: p and q trajectories (d=5)",
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )
    return eps_figure, mu_figure


def _add_endpoint_marker(
    figure: go.Figure,
    steps: np.ndarray,
    series: np.ndarray,
    color: str,
    **kwargs: Any,
) -> None:
    """Marks a line's actual final point.

    A run that ends early (crashes, or is otherwise shorter than its siblings) would
    otherwise be visually invisible, compressed into a sliver of the x-axis against
    runs that trained far longer -- this flags exactly where its data actually stops.

    Args:
        figure: Figure to add the marker to.
        steps: The line's x-values, as returned by :func:`read_metric_series`.
        series: The line's y-values.
        color: Marker color, matching the line it belongs to.
        **kwargs: Forwarded to ``add_trace`` (e.g. ``row``/``col`` for a subplot).
    """
    figure.add_trace(
        go.Scatter(
            x=[steps[-1]],
            y=[series[-1]],
            mode="markers",
            marker={"color": color, "symbol": "x", "size": 7},
            showlegend=False,
        ),
        **kwargs,
    )


def _write(figure: go.Figure, path: Path, width: int, height: int) -> None:
    """Writes ``figure`` to ``path`` as a PNG, at 2x scale for print sharpness."""
    figure.write_image(path, width=width, height=height, scale=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
