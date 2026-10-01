"""Downloads the SNAP community-structured graphs (e.g., com-DBLP).

Run this once before link prediction on SNAP datasets (it needs network access, so use
the login node if compute nodes have none); jobs read the files from disk. Files are
stored under ``$TNBBETA_DATA_DIR/snap_community`` and fetched from the SNAP repository.

Usage:
    uv run python -m apps.data.download_snap_community [--datasets dblp amazon]
"""

from __future__ import annotations

import argparse
import sys
from urllib.request import urlretrieve

from tnbbeta_vae.data.snap_community import (
    SNAP_COMMUNITY_DATASETS,
    load_snap_community,
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
        default=list(SNAP_COMMUNITY_DATASETS),
        choices=SNAP_COMMUNITY_DATASETS,
    )
    args = parser.parse_args(argv)

    root = data_dir() / "snap_community"
    root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        base_url = "https://snap.stanford.edu/data"
        # Download edge list (ungraph = undirected graph).
        edge_file = root / f"com-{name}.ungraph.txt.gz"
        if not edge_file.exists():
            urlretrieve(
                f"{base_url}/com-{name}.ungraph.txt.gz",
                edge_file,  # noqa: S310
            )

        # Download community membership file.
        community_file = root / f"com-{name}.all.cmty.txt.gz"
        if not community_file.exists():
            urlretrieve(
                f"{base_url}/com-{name}.all.cmty.txt.gz",
                community_file,  # noqa: S310
            )

        # Verify by loading.
        graph = load_snap_community(root, name)
        edges = int(graph.adjacency.sum() // 2)
        print(
            f"{name}: {graph.adjacency.shape[0]} nodes, {edges} edges, "
            f"{graph.features.shape[1]} features (sparse identity) OK in {root}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
