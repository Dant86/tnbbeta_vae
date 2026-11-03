"""Aggregates ``fixed_m_sweep_sbm.sbatch``'s outputs into a mean +/- std table.

Usage:
    uv run python -m apps.eval.fixed_m_sweep_table --root runs/fixed_m_sweep_sbm

Reads ``<root>/<condition>_seed<seed>/sbm_recovery.json`` (``condition`` is ``free`` or
the fixed ``m``), pools the seeds of each condition, and prints a markdown table of
test AUC/AP, the learned ``p``/``q``/temperature and the posterior's ``m`` and bimodal
fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_COLUMNS = [
    ("test_auc", "AUC"),
    ("test_ap", "AP"),
    ("m_mean", "m"),
    ("frac_bimodal", "frac_bimodal"),
    ("p_mean", "p"),
    ("q_mean", "q"),
    ("temperature", "temperature"),
]


def main(argv: list[str] | None = None) -> None:
    """Prints the table for every condition found under ``--root``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(render_table(collect(args.root)))


def collect(root: Path) -> dict[str, list[dict[str, float]]]:
    """Returns each condition's per-seed TNBBeta result dicts."""
    conditions: dict[str, list[dict[str, float]]] = {}
    for path in sorted(root.glob("*_seed*/sbm_recovery.json")):
        condition = path.parent.name.rsplit("_seed", 1)[0]
        result = json.loads(path.read_text())["tnbbeta"]
        if "no_checkpoint" in result:
            print(f"skipping {path}: no checkpoint (diverged)", file=sys.stderr)
            continue
        conditions.setdefault(condition, []).append(result)
    return conditions


def render_table(conditions: dict[str, list[dict[str, float]]]) -> str:
    """Formats ``collect``'s output as a markdown table, free-epsilon row first."""
    header = ["condition (m)", "seeds", *(label for _, label in _COLUMNS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for condition in sorted(conditions, key=_sort_key):
        runs = conditions[condition]
        cells = [condition, str(len(runs))]
        for key, _ in _COLUMNS:
            values = np.array([run[key] for run in runs if key in run], dtype=float)
            cells.append(
                f"{values.mean():.4f} ± {values.std():.4f}" if len(values) else "n/a"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sort_key(condition: str) -> float:
    return float("-inf") if condition == "free" else float(condition)


if __name__ == "__main__":
    main()
