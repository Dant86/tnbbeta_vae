"""Downloads DTD (all three splits, all 47 categories) into the data directory.

Run this once before training (it needs network access); training, eval, and
category curation read the data from disk and never download.

Usage:
    uv run python -m apps.data.download_dtd [--data-dir PATH]

The default directory is ``TNBBETA_DATA_DIR`` (see ``.env.sample``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tnbbeta_vae.paths import data_dir

# DTD's official partition 1 splits its 5,640 images (47 categories x 120
# images/category) into three equal thirds: 1,880 images (40/category) each
# for train/val/test.
_EXPECTED_SIZE = 1_880
_EXPECTED_CATEGORIES = 47
_SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies DTD's three splits.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        RuntimeError: If a split doesn't have the expected number of images
            or categories.
    """
    from torchvision import datasets

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
    for split in _SPLITS:
        dataset = datasets.DTD(root=str(root), split=split, download=True)
        if len(dataset) != _EXPECTED_SIZE:
            raise RuntimeError(
                f"DTD {split} split has {len(dataset)} images, expected "
                f"{_EXPECTED_SIZE}."
            )
        if len(dataset.classes) != _EXPECTED_CATEGORIES:
            raise RuntimeError(
                f"DTD {split} split has {len(dataset.classes)} categories, "
                f"expected {_EXPECTED_CATEGORIES}."
            )
        print(f"DTD {split}: {_EXPECTED_SIZE} images OK in {root}")


if __name__ == "__main__":
    main(sys.argv[1:])
