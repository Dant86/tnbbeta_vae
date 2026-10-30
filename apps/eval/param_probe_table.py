"""Aggregates ``apps.eval.param_probe`` outputs over seeds into markdown tables.

Usage:
    uv run python -m apps.eval.param_probe_table [--dataset cora ...]

Prints, per dataset and condition (mean over the seeds found): the decoding AUCs from
centres vs. samples, the parameters' effective rank, and each parameter's Spearman
correlation with log degree. With ``--partial``, the partial correlations (given log
degree) with the other covariates and the ``R^2`` gain over degree alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tnbbeta_vae.paths import checkpoint_dir

_DATASETS = ["cora", "citeseer", "pubmed", "mag_cs"]
_CONDITIONS = [
    ("tnbbeta free", "fixedmg_{ds}_free"),
    ("tnbbeta m>=0", "fixedmg_{ds}_0.5"),
    ("vmf", "fixedmgbase_{ds}_vmf"),
    ("power_spherical", "fixedmgbase_{ds}_power_spherical"),
    ("gaussian", "fixedmgbase_{ds}_gaussian"),
]


def main(argv: list[str] | None = None) -> None:
    """Prints the tables.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=_DATASETS)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or checkpoint_dir()
    for dataset in args.dataset:
        runs = {
            label: _load(root, pattern.format(ds=dataset))
            for label, pattern in _CONDITIONS
        }
        print(f"\n### {dataset}\n")
        print(decoding_table(runs))
        print()
        print(degree_table(runs))
        if args.partial:
            print()
            print(partial_table(runs))


def decoding_table(runs: dict[str, list[dict]]) -> str:
    """Centre vs. sampled AUC, effective rank and r_bar per condition."""
    lines = [
        "| condition | seeds | centre AUC | sampled AUC (S=1) | sampled AUC (S=16) "
        "| r_bar | eff. rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, seed_runs in runs.items():
        if not seed_runs:
            continue
        lines.append(
            f"| {label} | {len(seed_runs)} "
            f"| {_mean(seed_runs, lambda r: r['decoding']['centre_auc'])} "
            f"| {_mean(seed_runs, lambda r: r['decoding']['sampled_auc_1'])} "
            f"| {_mean(seed_runs, lambda r: r['decoding']['sampled_auc_16'])} "
            f"| {_mean(seed_runs, _r_bar)} "
            f"| {_mean(seed_runs, _effective_rank)} |"
        )
    return "\n".join(lines)


def degree_table(runs: dict[str, list[dict]]) -> str:
    """Each parameter's Spearman with log degree, and the R^2 for degree."""
    lines = [
        "| condition | parameter | Spearman vs log degree | R^2 (degree) |",
        "|---|---|---|---|",
    ]
    for label, seed_runs in runs.items():
        if not seed_runs:
            continue
        names = seed_runs[0]["parameter_map"]["parameters"]
        r2 = _mean(seed_runs, lambda r: r["parameter_map"]["r2"]["log_degree"])
        for name in names:
            spearman = _mean(
                seed_runs,
                lambda r, n=name: r["parameter_map"]["spearman"]["log_degree"][n],
            )
            lines.append(f"| {label} | {name} | {spearman} | {r2} |")
    return "\n".join(lines)


def partial_table(runs: dict[str, list[dict]]) -> str:
    """Information beyond degree: R^2 gain and partial correlations per covariate."""
    lines = [
        "| condition | covariate | R^2 degree only | R^2 degree + params | gain |",
        "|---|---|---|---|---|",
    ]
    for label, seed_runs in runs.items():
        if not seed_runs:
            continue
        for covariate in seed_runs[0]["parameter_map"].get("r2_over_degree", {}):
            base = _values(
                seed_runs,
                lambda r, c=covariate: r["parameter_map"]["r2_over_degree"][c][
                    "degree_only"
                ],
            )
            full = _values(
                seed_runs,
                lambda r, c=covariate: r["parameter_map"]["r2_over_degree"][c]["plus"],
            )
            lines.append(
                f"| {label} | {covariate} | {np.mean(base):.3f} | {np.mean(full):.3f} "
                f"| {np.mean(full) - np.mean(base):+.3f} |"
            )
    return "\n".join(lines)


def _load(root: Path, run: str) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob(f"{run}_seed*/param_probe.json"))
    ]


def _effective_rank(run: dict) -> float | None:
    return run["parameter_map"].get("effective_rank")


def _r_bar(run: dict) -> float | None:
    return run.get("r_bar")


def _values(seed_runs: list[dict], getter) -> list[float]:
    values = [getter(run) for run in seed_runs]
    return [v for v in values if v is not None]


def _mean(seed_runs: list[dict], getter) -> str:
    values = _values(seed_runs, getter)
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values else "n/a"


if __name__ == "__main__":
    main()
