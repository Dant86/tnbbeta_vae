"""Prints the link-prediction results table (S-VAE paper, Table 4).

Usage:
    uv run python -m apps.link_prediction.table [--datasets cora citeseer pubmed]

Reads ``$TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_<family>.json`` (from
``apps.link_prediction.main``) and prints the test AUC and AP, mean +- standard
deviation over seeds, at each family's configuration selected on validation AUC.
"""

from __future__ import annotations

import argparse
import json
import sys

from tnbbeta_vae.paths import checkpoint_dir

_FAMILIES = {"gaussian": "N-VGAE", "vmf": "S-VGAE (vMF)", "tnbbeta": "TNBBeta-VGAE"}


def main(argv: list[str] | None = None) -> None:
    """Prints the markdown table.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    args = parser.parse_args(argv)

    header = ["dataset", "metric", *_FAMILIES.values()]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for dataset in args.datasets:
        for metric, label in (("test_auc", "AUC"), ("test_ap", "AP")):
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


if __name__ == "__main__":
    main(sys.argv[1:])
