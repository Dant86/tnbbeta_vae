"""Summarises ``apps.synthetic.head_bias_sweep`` output as markdown tables.

Usage: python -m apps.eval.head_bias_report DIR_WITH_sweep_shard*.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_ORDER = [
    "tnb",
    "tnb_hb",
    "tnb_hb_p3",
    "vmf_init",
    "vmf_def",
    "vmf_def_hb",
    "ps_def",
    "ps_def_hb",
]
_METRICS = [
    ("escape_rate", "escaped"),
    ("escape_epoch", "median esc ep"),
    ("r_bar_final", "r_bar final"),
    ("r_bar_best", "r_bar best"),
    ("kappa_final", "kappa final"),
    ("sampled_auc", "AUC"),
    ("ll", "log loss"),
    ("recal_ll", "recal ll"),
    ("best_epoch", "best ep"),
]
# (treatment, control, metric, higher_is_better)
_PAIRS = [
    ("tnb_hb", "tnb", "sampled_auc", True),
    ("tnb_hb", "tnb", "r_bar_best", True),
    ("tnb_hb", "tnb", "recal_ll", False),
    ("tnb_hb_p3", "tnb_hb", "sampled_auc", True),
    ("tnb_hb_p3", "tnb_hb", "r_bar_best", True),
    ("tnb_hb_p3", "tnb_hb", "recal_ll", False),
    ("tnb_hb_p3", "tnb", "sampled_auc", True),
    ("vmf_def_hb", "vmf_def", "sampled_auc", True),
    ("vmf_def_hb", "vmf_def", "kappa_final", True),
    ("vmf_def_hb", "vmf_def", "recal_ll", False),
    ("ps_def_hb", "ps_def", "sampled_auc", True),
    ("ps_def_hb", "ps_def", "kappa_final", True),
    ("ps_def_hb", "ps_def", "recal_ll", False),
    ("vmf_def", "vmf_init", "kappa_final", True),
]


def main(argv: list[str] | None = None) -> None:
    """Prints the tables for the directory in ``argv[0]``."""
    args = sys.argv[1:] if argv is None else argv
    rows = [
        json.loads(line)
        for path in sorted(Path(args[0]).glob("sweep_shard*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    print(f"{len(rows)} runs\n")
    for lr in sorted({r["lr"] for r in rows}):
        for scope in [None, *sorted({r["family"] for r in rows})]:
            subset = [r for r in rows if r["lr"] == lr and scope in (None, r["family"])]
            print(f"### lr {lr}, {'all families pooled' if scope is None else scope}\n")
            print(_table(subset))
    print("### Paired comparisons (same family, draw, lr); win = treatment better\n")
    print("| lr | treatment vs control | metric | n | mean diff (sd) | wins | ties |")
    print("|---|---|---|---|---|---|---|")
    for lr in sorted({r["lr"] for r in rows}):
        for treatment, control, metric, higher in _PAIRS:
            print(_pair(rows, lr, treatment, control, metric, higher))
    print("\n### Learning-rate gap (mean over all runs of lr 0.01 minus lr 0.003)\n")
    print("| cond | dAUC | d r_bar best | d recal ll |")
    print("|---|---|---|---|")
    for cond in _ORDER:
        cells = {
            lr: [r for r in rows if r["cond"] == cond and r["lr"] == lr]
            for lr in (0.003, 0.01)
        }
        if not all(cells.values()):
            continue
        diffs = [
            f"{_mean(cells[0.01], k) - _mean(cells[0.003], k):+.3f}"
            for k in ("sampled_auc", "r_bar_best", "recal_ll")
        ]
        print(f"| {cond} | " + " | ".join(diffs) + " |")


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def _table(rows: list[dict]) -> str:
    header = "| cond | n | " + " | ".join(label for _, label in _METRICS) + " |"
    lines = [header, "|" + "---|" * (len(_METRICS) + 2)]
    for cond in _ORDER:
        group = [r for r in rows if r["cond"] == cond]
        if not group:
            continue
        cells = []
        for key, _ in _METRICS:
            if key == "escape_rate":
                done = sum(r["escape_epoch"] is not None for r in group)
                cells.append(f"{done}/{len(group)}")
            elif key == "escape_epoch":
                values = [r[key] for r in group if r[key] is not None]
                cells.append(f"{np.median(values):.0f}" if values else "-")
            else:
                values = [r[key] for r in group if r.get(key) is not None]
                if not values:
                    cells.append("-")
                elif len(values) == 1:
                    cells.append(f"{values[0]:.3f}")
                else:
                    cells.append(
                        f"{np.mean(values):.3f} +- {np.std(values, ddof=1):.3f}"
                    )
        lines.append(f"| {cond} | {len(group)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _pair(
    rows: list[dict],
    lr: float,
    treatment: str,
    control: str,
    metric: str,
    higher: bool,
) -> str:
    def index(cond: str) -> dict[tuple[str, int], dict]:
        return {
            (r["family"], r["draw"]): r
            for r in rows
            if r["cond"] == cond and r["lr"] == lr
        }

    a, b = index(treatment), index(control)
    diffs = []
    for key in a.keys() & b.keys():
        x, y = a[key].get(metric), b[key].get(metric)
        if x is not None and y is not None:
            diffs.append(x - y)
    if not diffs:
        return f"| {lr} | {treatment} vs {control} | {metric} | 0 | - | - | - |"
    arr = np.asarray(diffs)
    wins = int((arr > 1e-4).sum() if higher else (arr < -1e-4).sum())
    ties = int((np.abs(arr) <= 1e-4).sum())
    return (
        f"| {lr} | {treatment} vs {control} | {metric} | {len(arr)} | "
        f"{arr.mean():+.3f} ({arr.std(ddof=1):.3f}) | {wins} | {ties} |"
    )


if __name__ == "__main__":
    main()
