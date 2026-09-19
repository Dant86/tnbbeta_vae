"""Downloads CIFAR-10 (train and test splits) into the configured data directory.

Run this once before training (it needs network access); training and
evaluation jobs read the data from disk and never download.

Usage:
    uv run python -m apps.data.download_cifar10 [--data-dir PATH]

The default directory is ``TNBBETA_DATA_DIR`` (see ``.env.sample``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tnbbeta_vae.data.cifar10 import load_cifar10
from tnbbeta_vae.paths import data_dir

_EXPECTED_SIZES = {True: 50_000, False: 10_000}


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies CIFAR-10.

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
    for train, expected in _EXPECTED_SIZES.items():
        dataset = load_cifar10(root, train=train, download=True)
        split = "train" if train else "test"
        if len(dataset) != expected:
            raise RuntimeError(
                f"CIFAR-10 {split} split has {len(dataset)} images, expected "
                f"{expected}."
            )
        print(f"CIFAR-10 {split}: {expected} images OK in {root}")


if __name__ == "__main__":
    main(sys.argv[1:])
