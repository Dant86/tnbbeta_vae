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

# SNAP's "ground-truth communities" datasets are hosted under this path, not under
# /data/ directly (verified against the live site, not the HTML page's prose, which
# describes the same files with URLs that 404).
_BASE_URL = "https://snap.stanford.edu/data/bigdata/communities"

# Most datasets' community file is "com-<name>.all.cmty.txt.gz", but Amazon's is named
# "com-amazon.all.dedup.cmty.txt.gz" on SNAP's server -- saved locally under the common
# "com-<name>.all.cmty.txt.gz" name regardless, so the loader doesn't need to know this.
_COMMUNITY_FILE_SOURCE_NAME = {
    "dblp": "com-dblp.all.cmty.txt.gz",
    "amazon": "com-amazon.all.dedup.cmty.txt.gz",
}


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
        # Download edge list (ungraph = undirected graph).
        edge_file = root / f"com-{name}.ungraph.txt.gz"
        if not edge_file.exists():
            urlretrieve(
                f"{_BASE_URL}/com-{name}.ungraph.txt.gz",
                edge_file,  # noqa: S310
            )

        # Download community membership file (source filename varies by dataset; see
        # _COMMUNITY_FILE_SOURCE_NAME).
        community_file = root / f"com-{name}.all.cmty.txt.gz"
        if not community_file.exists():
            urlretrieve(
                f"{_BASE_URL}/{_COMMUNITY_FILE_SOURCE_NAME[name]}",
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
