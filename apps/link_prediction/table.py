"""Prints the link-prediction results table (S-VAE paper, Table 4).

Usage:
    uv run python -m apps.link_prediction.table [--datasets cora citeseer pubmed] \
        [--by-dimension]

Reads ``$TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_<family>.json`` (from
``apps.link_prediction.main``) and prints the test AUC and AP, mean +- standard
deviation over seeds, at each family's configuration selected on validation AUC.

``--by-dimension`` additionally prints, per dataset, a breakdown across every
latent dimension in the sweep (``apps.link_prediction.main``'s grid search already
keeps every ``(lr, dropout, latent_dim)`` cell in its output JSON's
``"configurations"`` list, not just the validation-selected one -- this reads that,
it doesn't need a re-run). Rows are by *position* in each family's own sorted
dimension list, not by raw ``latent_dim`` value, since the Gaussian family and the
sphere families use different raw dimensions at the same manifold dimension ``d``
(``scripts/slurm/link_prediction.sbatch``'s ``d``/``d+1`` convention, per
``CLAUDE.md``) -- each cell is labeled with its own family's actual ``latent_dim``
so the breakdown is unambiguous even if a sweep's dimension lists aren't the same
length across families.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from tnbbeta_vae.paths import checkpoint_dir

_FAMILIES = {
    "gaussian": "N-VGAE",
    "vmf": "S-VGAE (vMF)",
    "tnbbeta": "TNBBeta-VGAE",
    "power_spherical": "Power Spherical-VGAE",
}
_METRICS = (("test_auc", "AUC"), ("test_ap", "AP"))


def main(argv: list[str] | None = None) -> None:
    """Prints the markdown table(s).

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument(
        "--by-dimension",
        action="store_true",
        help="Also print a per-latent_dim breakdown for each dataset.",
    )
    args = parser.parse_args(argv)

    header = ["dataset", "metric", *_FAMILIES.values()]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for dataset in args.datasets:
        for metric, label in _METRICS:
            cells = [dataset, label]
            for family in _FAMILIES:
                path = checkpoint_dir() / "link_prediction" / f"{dataset}_{family}.json"
                if not path.exists():
                    cells.append("-")
                    continue
                selected = json.loads(path.read_text())["selected"]
                mean, std = selected[metric], selected[f"{metric}_std"]
                cells.append(f"{100 * mean:.1f} ± {100 * std:.1f}")
            print("| " + " | ".join(cells) + " |")

    if args.by_dimension:
        for dataset in args.datasets:
            print(f"\n### {dataset}, by latent dimension\n")
            _print_by_dimension(dataset)


def _print_by_dimension(dataset: str) -> None:
    """Prints one AUC/AP table (dimension index x family) for ``dataset``."""
    per_family_rows = {
        family: _best_per_dimension(path)
        for family in _FAMILIES
        if (
            path := checkpoint_dir() / "link_prediction" / f"{dataset}_{family}.json"
        ).exists()
    }
    if not per_family_rows:
        print("(no runs found)\n")
        return

    num_rows = max(len(rows) for rows in per_family_rows.values())
    present = [family for family in _FAMILIES if family in per_family_rows]
    for metric, label in _METRICS:
        print(f"**{label}**\n")
        print(f"| dim index | {' | '.join(_FAMILIES[f] for f in present)} |")
        print("|" + "---|" * (len(present) + 1))
        for i in range(num_rows):
            cells = [str(i)]
            for family in present:
                rows = per_family_rows[family]
                if i >= len(rows):
                    cells.append("-")
                    continue
                row = rows[i]
                mean, std = row[metric], row[f"{metric}_std"]
                cells.append(
                    f"dim={row['latent_dim']}: {100 * mean:.1f} ± {100 * std:.1f}"
                )
            print("| " + " | ".join(cells) + " |")
        print()


def _best_per_dimension(path: Any) -> list[dict[str, Any]]:
    """The best (lr, dropout) configuration at each distinct latent_dim in ``path``'s
    ``"configurations"`` list, sorted by latent_dim, picked by validation AUC among
    that dimension's own configurations.
    """
    configurations = json.loads(path.read_text())["configurations"]
    by_dim: dict[int, list[dict[str, Any]]] = {}
    for config in configurations:
        by_dim.setdefault(config["latent_dim"], []).append(config)
    return [
        max(configs, key=lambda c: c["val_auc"])
        for _, configs in sorted(by_dim.items())
    ]


if __name__ == "__main__":
    main(sys.argv[1:])
