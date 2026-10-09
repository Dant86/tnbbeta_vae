"""Downloads the NOCD paper's MAG co-authorship networks.

Run this once before link prediction on MAG co-authorship datasets (it needs network
access, so use the login node if compute nodes have none); jobs read the files from
disk. Files are stored under ``$TNBBETA_DATA_DIR/mag_coauthor`` and fetched from the
NOCD paper's own repository (archived, but its ``data/`` files are still served from
GitHub's raw-content host).

Usage:
    uv run python -m apps.data.download_mag_coauthor [--datasets cs eng chem med]
"""

from __future__ import annotations

import argparse
import sys
from urllib.request import urlretrieve

from tnbbeta_vae.data.mag_coauthor import (
    MAG_COAUTHOR_DATASETS,
    load_mag_coauthor,
    load_mag_communities,
)
from tnbbeta_vae.paths import data_dir

# Shchur & Guennemann's "Overlapping Community Detection with Graph Neural Networks"
# (NOCD) repository is archived but its data/ files are still served directly from
# GitHub's raw-content host (verified against the live URL, not assumed).
_BASE_URL = (
    "https://raw.githubusercontent.com/shchur/overlapping-community-detection"
    "/master/data"
)


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies the requested MAG co-authorship networks.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(MAG_COAUTHOR_DATASETS),
        choices=MAG_COAUTHOR_DATASETS,
    )
    args = parser.parse_args(argv)

    root = data_dir() / "mag_coauthor"
    root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        target = root / f"mag_{name}.npz"
        if not target.exists():
            urlretrieve(f"{_BASE_URL}/mag_{name}.npz", target)  # noqa: S310

        # Verify by loading.
        graph = load_mag_coauthor(root, name)
        memberships = load_mag_communities(root, name)
        edges = int(graph.adjacency.sum() // 2)
        num_communities = len(
            {
                community
                for communities in memberships.values()
                for community in communities
            }
        )
        print(
            f"{name}: {graph.adjacency.shape[0]} nodes, {edges} edges, "
            f"{graph.features.shape[1]} features, {num_communities} communities "
            f"OK in {root}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
