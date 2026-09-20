"""Aggregates per-run S-VAE metrics into a Table-1-style markdown table.

Usage:
    uv run python -m apps.eval.svae_table [--kind table1|knn] [--prefix mnist] \
        [--models gauss vmf tnb] [--dims 2 5 10 20 40] [--seeds 0 1 2 3 4]

Reads, from ``$TNBBETA_CHECKPOINT_DIR/<prefix>_<model>_d<dim>_seed<seed>/`` (the
names used by ``scripts/slurm/mnist_sweep.sbatch``), either
``svae_metrics_final_<split>.json`` (``--kind table1``, ``apps.eval.svae_metrics``:
LL, L[q] = the ELBO, RE, KL) or ``svae_knn_final.json`` (``--kind knn``,
``apps.eval.svae_knn``: k-NN accuracy for 100/600/1000 labels), and prints mean +-
standard deviation over seeds. A mean is bold if it is the best for that metric
(higher is better; KL is never bolded) and beats every other model with a Welch
t-test at p < ``--alpha``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
from scipy import stats

from tnbbeta_vae.paths import checkpoint_dir

MODEL_TITLES = {"gauss": "N-VAE", "vmf": "S-VAE (vMF)", "tnb": "TNBBeta"}
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
    parser.add_argument("--kind", choices=["table1", "knn"], default="table1")
    parser.add_argument("--prefix", default="mnist")
    parser.add_argument("--models", nargs="+", default=["gauss", "vmf", "tnb"])
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10, 20, 40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split", default="test")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args(argv)

    metrics = _KNN_METRICS if args.kind == "knn" else _TABLE1_METRICS
    filename = (
        "svae_knn_final.json"
        if args.kind == "knn"
        else f"svae_metrics_final_{args.split}.json"
    )
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
                cells.append(_format(values, winners.get(name) == model))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _significant_winner(
    results: Results, models: list[str], dim: int, metric: str, alpha: float
) -> str | None:
    """Returns the model with the best mean if it beats all others at ``alpha``."""
    series = {model: results.get((model, dim), {}).get(metric, []) for model in models}
    usable = {model: values for model, values in series.items() if len(values) >= 2}
    if len(usable) < 2 or len(usable) < len(models):
        return None
    best = max(usable, key=lambda model: float(np.mean(usable[model])))
    for model, values in usable.items():
        if model == best:
            continue
        test: Any = stats.ttest_ind(usable[best], values, equal_var=False)
        pvalue = float(test.pvalue)
        if not (pvalue < alpha and np.mean(usable[best]) > np.mean(values)):
            return None
    return best


def _format(values: list[float], bold: bool) -> str:
    if not values:
        return "-"
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    text = f"{np.mean(values):.2f} ± {spread:.2f}"
    return f"**{text}**" if bold else text


if __name__ == "__main__":
    main(sys.argv[1:])
