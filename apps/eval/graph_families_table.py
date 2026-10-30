"""Aggregates ``graph_families_screen.sbatch`` results into markdown tables.

Usage:
    uv run python -m apps.eval.graph_families_table [--graph assortative ...]

Reads ``$TNBBETA_CHECKPOINT_DIR/synthetic_graphs/gf_<preset>_<condition>.json``. Per
preset prints seed means (and standard deviations of the AUC) per condition, then the
sampled-prediction divergence from the true pair probabilities broken down by pair kind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apps.synthetic.graph_families import PRESETS
from tnbbeta_vae.paths import checkpoint_dir

_METRICS = [
    ("centre_test_auc", "centre AUC"),
    ("sampled_test_auc", "sampled AUC"),
    ("centre_test_log_loss", "centre logloss"),
    ("sampled_test_log_loss", "sampled logloss"),
    ("sampled_test_ece", "sampled ECE"),
    ("centre_truth_kl", "centre KL"),
    ("sampled_truth_kl", "sampled KL"),
]


def main(argv: list[str] | None = None) -> None:
    """Prints the tables for each requested preset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", nargs="+", default=sorted(PRESETS))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    directory = (args.root or checkpoint_dir()) / "synthetic_graphs"
    for graph in args.graph:
        conditions = collect(directory, graph)
        if not conditions:
            continue
        print(f"\n### {graph}\n")
        print(summary_table(conditions))
        print("\nSampled KL to the true pair probabilities, by pair kind\n")
        print(kind_table(conditions))


def collect(directory: Path, graph: str) -> dict[str, list[dict]]:
    """Returns each condition's per-seed result dicts for ``graph``."""
    prefix = f"gf_{graph}_"
    return {
        path.stem.removeprefix(prefix): json.loads(path.read_text())["runs"]
        for path in sorted(directory.glob(f"{prefix}*.json"))
    }


def summary_table(conditions: dict[str, list[dict]]) -> str:
    """Markdown table of seed-mean metrics per condition (AUC also shows its std)."""
    oracle = np.mean(
        [run["oracle_test_auc"] for runs in conditions.values() for run in runs]
    )
    header = ["condition", "seeds", *(label for _, label in _METRICS)]
    lines = [
        f"Oracle (true-probability) test AUC: {oracle:.3f}\n",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, runs in conditions.items():
        cells = [name, str(len(runs))]
        for key, _ in _METRICS:
            values = [run[key] for run in runs]
            spread = f" ± {np.std(values):.3f}" if key.endswith("auc") else ""
            cells.append(f"{np.mean(values):.3f}{spread}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kind_table(conditions: dict[str, list[dict]]) -> str:
    """Markdown table of the sampled truth KL per pair kind (seed means)."""
    kinds = sorted(
        {
            kind
            for runs in conditions.values()
            for run in runs
            for kind in run["sampled_truth_kl_by_kind"]
        }
    )
    lines = [
        "| condition | " + " | ".join(kinds) + " |",
        "|" + "---|" * (len(kinds) + 1),
    ]
    for name, runs in conditions.items():
        cells = []
        for kind in kinds:
            values = [
                run["sampled_truth_kl_by_kind"][kind]
                for run in runs
                if kind in run["sampled_truth_kl_by_kind"]
            ]
            cells.append(f"{np.mean(values):.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
