"""Downloads MNIST (train and test splits) into the configured data directory.

Run this once before training (it needs network access, so use the login node if
compute nodes have none); training and evaluation jobs read the data from disk.

Usage:
    uv run python -m apps.data.download_mnist [--data-dir PATH]

The default directory is ``TNBBETA_DATA_DIR`` (see ``.env.sample``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Literal

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.paths import data_dir

_EXPECTED_SIZES: dict[Literal["train", "val", "test"], int] = {
    "train": 50_000,
    "val": 10_000,
    "test": 10_000,
}


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies MNIST.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        RuntimeError: If a split doesn't have the expected number of images.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Destination directory (default: $TNBBETA_DATA_DIR).",
    )
    args = parser.parse_args(argv)

    root = args.data_dir or data_dir()
    root.mkdir(parents=True, exist_ok=True)
    for split, expected in _EXPECTED_SIZES.items():
        dataset = load_mnist(root, split=split, download=True)
        if len(dataset) != expected:
            raise RuntimeError(
                f"MNIST {split} split has {len(dataset)} images, expected {expected}."
            )
        print(f"MNIST {split}: {expected} images OK in {root}")


if __name__ == "__main__":
    main(sys.argv[1:])
