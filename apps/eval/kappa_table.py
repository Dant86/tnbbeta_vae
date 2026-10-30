"""Aggregates the fixed-kappa sweep and the kappa-dynamics runs into markdown tables.

Usage:
    uv run python -m apps.eval.kappa_table [--dataset cora ...]

Reads ``$TNBBETA_CHECKPOINT_DIR/link_prediction/`` files written by
``fixed_kappa_sweep_graphs.sbatch`` (``fixedk_<dataset>_<family>_x<multiple>_seed<s>``)
and ``kappa_dynamics_graphs.sbatch`` (``kdyn_<condition>_<dataset>_seed<s>``), each a
summary JSON plus a ``_trajectory.json``. Prints, per dataset, (1) test AUC (seed mean
and standard deviation) per condition, free and fixed kappa side by side, and (2) the
training dynamics: untrained-initialization validation AUC, kappa and ``r_bar`` over
training, and the gradient reaching the kappa head row relative to the direction rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tnbbeta_vae.paths import checkpoint_dir

_DATASETS = ["cora", "citeseer", "pubmed", "mag_cs"]
_CHECKPOINTS = (0, 10, 50)


def main(argv: list[str] | None = None) -> None:
    """Prints the tables for each requested dataset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=_DATASETS)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    directory = (args.root or checkpoint_dir()) / "link_prediction"
    for dataset in args.dataset:
        conditions = collect(directory, dataset)
        if not conditions:
            continue
        print(f"\n### {dataset}\n")
        print(auc_table(conditions))
        print("\nTraining dynamics (seed means)\n")
        print(dynamics_table(conditions))


def collect(directory: Path, dataset: str) -> dict[str, list[dict]]:
    """Returns each condition's per-seed ``{"test_auc", "trajectory"}`` dicts."""
    patterns = {
        f"kdyn_*_{dataset}_seed*.json": lambda name: (
            name.split("_seed")[0].removeprefix("kdyn_").removesuffix(f"_{dataset}")
        ),
        f"fixedk_{dataset}_*_seed*.json": lambda name: (
            name.split("_seed")[0]
            .removeprefix(f"fixedk_{dataset}_")
            .replace("_x", " x")
        ),
    }
    conditions: dict[str, list[dict]] = {}
    for pattern, label in patterns.items():
        for path in sorted(directory.glob(pattern)):
            if path.stem.endswith("_trajectory"):
                continue
            summary = json.loads(path.read_text())["selected"]
            trajectory_path = path.with_name(f"{path.stem}_trajectory.json")
            trajectory = (
                next(iter(json.loads(trajectory_path.read_text()).values()))
                if trajectory_path.exists()
                else []
            )
            conditions.setdefault(label(path.stem), []).append(
                {"test_auc": summary["test_auc"], "trajectory": trajectory}
            )
    return conditions


def auc_table(conditions: dict[str, list[dict]]) -> str:
    """Markdown table of test AUC (mean +- std over seeds) per condition."""
    lines = ["| condition | seeds | test AUC |", "|---|---|---|"]
    for name, runs in sorted(conditions.items()):
        aucs = [run["test_auc"] for run in runs]
        lines.append(
            f"| {name} | {len(aucs)} | {np.mean(aucs):.3f} ± {np.std(aucs):.3f} |"
        )
    return "\n".join(lines)


def dynamics_table(conditions: dict[str, list[dict]]) -> str:
    """Markdown table of per-epoch statistics (seed means) per condition.

    Columns: validation AUC of the untrained initialization (epoch -1); kappa mean at
    init, epoch 10 and the last logged epoch; ``r_bar`` at epochs 0, 10 and 50; and the
    kappa-row / direction-rows gradient-norm ratio at epochs 0 and 50 (free kappa only).
    """
    header = (
        "| condition | init val AUC | kappa init | kappa e10 | kappa last | "
        "r_bar e0 | r_bar e10 | r_bar e50 | grad ratio e0 | grad ratio e50 |"
    )
    lines = [header, "|" + "---|" * 10]
    for name, runs in sorted(conditions.items()):
        runs = [run for run in runs if run["trajectory"]]
        if not runs:
            continue
        cells = [
            name,
            _mean(runs, -1, "val_auc"),
            _mean(runs, -1, "kappa_mean"),
            _mean(runs, 10, "kappa_mean"),
            _mean(runs, None, "kappa_mean"),
            *(_mean(runs, epoch, "r_bar") for epoch in _CHECKPOINTS),
            _ratio(runs, 0),
            _ratio(runs, 50),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _row(run: dict, epoch: int | None) -> dict[str, float] | None:
    trajectory = run["trajectory"]
    if epoch is None:
        return trajectory[-1]
    return next((row for row in trajectory if row["epoch"] == epoch), None)


def _mean(runs: list[dict], epoch: int | None, key: str) -> str:
    values = [
        row[key] for run in runs if (row := _row(run, epoch)) is not None and key in row
    ]
    return f"{np.mean(values):.3g}" if values else "-"


def _ratio(runs: list[dict], epoch: int) -> str:
    ratios = [
        row["grad_norm_kappa_row"] / row["grad_norm_direction_rows"]
        for run in runs
        if (row := _row(run, epoch)) is not None
        and "grad_norm_kappa_row" in row
        and row["grad_norm_direction_rows"] > 0
    ]
    return f"{np.mean(ratios):.3g}" if ratios else "-"


if __name__ == "__main__":
    main()
