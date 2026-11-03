"""Aggregates ``fixed_m_sweep_graphs.sbatch``'s bucketed outputs into markdown tables.

Usage:
    uv run python -m apps.eval.fixed_m_graphs_table [--dataset cora ...] \
        [--breakdown degree|communities|none]

Reads ``$TNBBETA_CHECKPOINT_DIR/fixedmg_<dataset>_<cond>_seed<seed>/
bucketed_link_prediction.json`` (``cond`` is ``free`` or the fraction ``f`` of the
modality threshold, ``m = f * (latent_dim - 1) / 2``). For each dataset prints the
seed-pooled overall test AUC per condition and, with ``--breakdown``, the AUC within
each node bucket (training-degree quantile, or number of neighbor communities).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tnbbeta_vae.paths import checkpoint_dir

_DATASETS = ["cora", "citeseer", "pubmed", "mag_cs"]


def main(argv: list[str] | None = None) -> None:
    """Prints the tables for each requested dataset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=_DATASETS)
    parser.add_argument(
        "--breakdown", choices=["degree", "communities", "none"], default="none"
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or checkpoint_dir()
    for dataset in args.dataset:
        runs = collect(root, dataset)
        if not runs:
            continue
        print(f"\n### {dataset}\n")
        print(overall_table(runs))
        if args.breakdown != "none":
            key = (
                "by_degree"
                if args.breakdown == "degree"
                else ("by_neighbor_communities")
            )
            print(f"\n{dataset}: test AUC by {args.breakdown} bucket\n")
            print(bucket_table(runs, key))


def collect(root: Path, dataset: str) -> dict[str, list[dict]]:
    """Returns each condition's per-seed decomposition dicts for ``dataset``."""
    conditions: dict[str, list[dict]] = {}
    for path in sorted(
        root.glob(f"fixedmg_{dataset}_*_seed*/bucketed_link_prediction.json")
    ):
        condition = path.parent.name.removeprefix(f"fixedmg_{dataset}_")
        condition = condition.rsplit("_seed", 1)[0]
        conditions.setdefault(condition, []).append(json.loads(path.read_text()))
    return conditions


def overall_table(runs: dict[str, list[dict]]) -> str:
    """Overall AUC/AP per condition, free-epsilon first, then by increasing ``f``."""
    lines = ["| cond (f) | seeds | AUC | AP |", "|---|---|---|---|"]
    for condition in sorted(runs, key=_sort_key):
        auc = np.array([run["overall"]["auc"] for run in runs[condition]])
        ap = np.array([run["overall"]["ap"] for run in runs[condition]])
        lines.append(
            f"| {condition} | {len(auc)} | {auc.mean():.4f} ± {auc.std():.4f} "
            f"| {ap.mean():.4f} ± {ap.std():.4f} |"
        )
    return "\n".join(lines)


def bucket_table(runs: dict[str, list[dict]], key: str) -> str:
    """AUC (mean over seeds) per condition and bucket."""
    buckets = sorted(
        {
            int(b)
            for seed_runs in runs.values()
            for run in seed_runs
            for b in run.get(key, {})
        }
    )
    if not buckets:
        return "(no buckets)"
    lines = [
        "| cond (f) | " + " | ".join(f"b{b}" for b in buckets) + " |",
        "|---|" + "---|" * len(buckets),
    ]
    for condition in sorted(runs, key=_sort_key):
        cells = []
        for bucket in buckets:
            values = [
                run[key][str(bucket)]["auc"]
                for run in runs[condition]
                if str(bucket) in run.get(key, {})
            ]
            cells.append(f"{np.mean(values):.3f}" if values else "n/a")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sort_key(condition: str) -> float:
    return float("-inf") if condition == "free" else float(condition)


if __name__ == "__main__":
    main()
