"""Downloads the raw Planetoid files for Cora, Citeseer and Pubmed.

Run this once before link prediction (it needs network access, so use the login node
if compute nodes have none); jobs read the files from disk. Files are stored under
``$TNBBETA_DATA_DIR/planetoid`` and fetched from the Planetoid repository
(``PLANETOID_URL`` in ``tnbbeta_vae.data.planetoid``).

Usage:
    uv run python -m apps.data.download_planetoid [--datasets cora citeseer pubmed]
"""

from __future__ import annotations

import argparse
import sys
from urllib.request import urlretrieve

from tnbbeta_vae.data.planetoid import (
    PLANETOID_DATASETS,
    PLANETOID_FILE_SUFFIXES,
    PLANETOID_URL,
    load_planetoid,
)
from tnbbeta_vae.paths import data_dir


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies the requested datasets.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PLANETOID_DATASETS),
        choices=PLANETOID_DATASETS,
    )
    args = parser.parse_args(argv)

    root = data_dir() / "planetoid"
    root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        for suffix in PLANETOID_FILE_SUFFIXES:
            target = root / f"ind.{name}.{suffix}"
            if not target.exists():
                urlretrieve(f"{PLANETOID_URL}/ind.{name}.{suffix}", target)  # noqa: S310
        graph = load_planetoid(root, name)
        edges = int(graph.adjacency.sum() // 2)
        print(
            f"{name}: {graph.adjacency.shape[0]} nodes, {edges} edges, "
            f"{graph.features.shape[1]} features OK in {root}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
