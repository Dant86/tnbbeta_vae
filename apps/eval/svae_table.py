"""Aggregates per-run S-VAE metrics into a Table-1-style markdown table.

Usage:
    uv run python -m apps.eval.svae_table [--kind table1|knn|confidence] \
        [--prefix mnist] [--models gauss vmf tnb] [--dims 2 5 10 20 40] \
        [--seeds 0 1 2 3 4]

Reads, from ``$TNBBETA_CHECKPOINT_DIR/<prefix>_<model>_d<dim>_seed<seed>/`` (the
names used by ``scripts/slurm/mnist_sweep.sbatch``), one of:
``svae_metrics_final_<split>.json`` (``--kind table1``, ``apps.eval.svae_metrics``:
LL, L[q] = the ELBO, RE, KL), ``svae_knn_final.json`` (``--kind knn``,
``apps.eval.svae_knn``: k-NN accuracy for 100/600/1000 labels, direction only), or
``confidence_probe_final.json`` (``--kind confidence``, ``apps.eval.confidence_probe``:
the same k-NN accuracy using only the posterior's confidence scalar -- p, kappa or mean
std -- with no directional information). Prints mean +- standard deviation over seeds.
A mean is bold if it is the best for that metric (higher is better; KL is never bolded)
and beats every other model with a Welch t-test at p < ``--alpha``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
from scipy import stats

from tnbbeta_vae.paths import checkpoint_dir

MODEL_TITLES = {
    "gauss": "N-VAE",
    "vmf": "S-VAE (vMF)",
    "tnb": "TNBBeta",
    "vmfk": "S-VAE (vMF, kappa init)",
    "vmfs": "S-VAE (vMF, S^d)",
    "tnbs": "TNBBeta (S^d)",
    "vmfks": "S-VAE (vMF, kappa init, S^d)",
}
_TABLE1_METRICS = [("ll", "LL"), ("elbo", "L[q]"), ("re", "RE"), ("kl", "KL")]
_KNN_METRICS = [("acc_100", "N=100"), ("acc_600", "N=600"), ("acc_1000", "N=1000")]
_BOLDABLE = {"ll", "elbo", "re", "acc_100", "acc_600", "acc_1000"}

Results = dict[tuple[str, int], dict[str, list[float]]]


def main(argv: list[str] | None = None) -> None:
    """Prints the table for the runs found in the checkpoint directory.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["table1", "knn", "confidence"], default="table1"
    )
    parser.add_argument("--prefix", default="mnist")
    parser.add_argument("--models", nargs="+", default=["gauss", "vmf", "tnb"])
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10, 20, 40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split", default="test")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args(argv)

    metrics = _KNN_METRICS if args.kind in ("knn", "confidence") else _TABLE1_METRICS
    filenames = {
        "knn": "svae_knn_final.json",
        "confidence": "confidence_probe_final.json",
        "table1": f"svae_metrics_final_{args.split}.json",
    }
    filename = filenames[args.kind]
    results, missing = collect(
        args.prefix, args.models, args.dims, args.seeds, filename, metrics
    )
    print(render(results, args.models, args.dims, metrics, args.alpha))
    if missing:
        print(
            f"\n{missing} run(s) had no metrics file and were skipped.", file=sys.stderr
        )


def collect(
    prefix: str,
    models: list[str],
    dims: list[int],
    seeds: list[int],
    filename: str,
    metrics: list[tuple[str, str]],
) -> tuple[Results, int]:
    """Loads every run's metrics.

    Returns:
        A mapping ``(model, dim) -> metric -> values over seeds``, and the number
        of runs whose metrics file was missing.
    """
    results: Results = {}
    missing = 0
    for model in models:
        for dim in dims:
            values: dict[str, list[float]] = {name: [] for name, _ in metrics}
            for seed in seeds:
                path = (
                    checkpoint_dir() / f"{prefix}_{model}_d{dim}_seed{seed}" / filename
                )
                if not path.exists():
                    missing += 1
                    continue
                record = json.loads(path.read_text())
                for name, _ in metrics:
                    values[name].append(record[name])
            results[(model, dim)] = values
    return results, missing


def render(
    results: Results,
    models: list[str],
    dims: list[int],
    metrics: list[tuple[str, str]],
    alpha: float = 0.01,
) -> str:
    """Formats ``results`` as a markdown table (rows: dimensions)."""
    header = ["d"]
    for model in models:
        title = MODEL_TITLES.get(model, model)
        header += [f"{title} {label}" for _, label in metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dim in dims:
        cells = [str(dim)]
        winners = {
            name: _significant_winner(results, models, dim, name, alpha)
            for name, _ in metrics
            if name in _BOLDABLE
        }
        for model in models:
            for name, _ in metrics:
                values = results.get((model, dim), {}).get(name, [])
                cells.append(format_mean_std(values, winners.get(name) == model))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def significant_winner(series: dict[str, list[float]], alpha: float) -> str | None:
    """Returns the key with the highest mean if a Welch t-test beats every other one.

    Args:
        series: Values (one per seed) for each competitor; higher is better.
        alpha: Significance level.

    Returns:
        The winning key, or ``None`` if a competitor has fewer than two values or the
        best one does not beat every other significantly.
    """
    usable = {name: values for name, values in series.items() if len(values) >= 2}
    if len(usable) < 2 or len(usable) < len(series):
        return None
    best = max(usable, key=lambda name: float(np.mean(usable[name])))
    for name, values in usable.items():
        if name == best:
            continue
        test: Any = stats.ttest_ind(usable[best], values, equal_var=False)
        if not (float(test.pvalue) < alpha and np.mean(usable[best]) > np.mean(values)):
            return None
    return best


def format_mean_std(values: list[float], bold: bool, scale: float = 1.0) -> str:
    """Formats ``mean ± std`` (bold if ``bold``), or ``-`` if there are no values."""
    if not values:
        return "-"
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    text = f"{scale * np.mean(values):.2f} ± {scale * spread:.2f}"
    return f"**{text}**" if bold else text


def _significant_winner(
    results: Results, models: list[str], dim: int, metric: str, alpha: float
) -> str | None:
    """Returns the model with the best mean if it beats all others at ``alpha``."""
    series = {model: results.get((model, dim), {}).get(metric, []) for model in models}
    return significant_winner(series, alpha)


if __name__ == "__main__":
    main(sys.argv[1:])
