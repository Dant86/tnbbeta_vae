"""Measures DTD's structure-tensor coherence per candidate oriented-texture category.

Run this yourself, once DTD is downloaded (e.g. via ``apps/data/download_dtd.py``
on the cluster, where the data needs to live for training anyway) -- it is not
run as part of building this tool, and ``tnbbeta_vae.data.dtd.ORIENTED_CATEGORIES``
ships as the full, uncurated 10-candidate list until you do.

Usage:
    uv run python -m apps.data.curate_dtd_categories [--data-dir PATH] \
        [--categories banded braided ...] [--sample-size 60] [--image-size 64] \
        [--seed 0] [--out dtd_category_coherence.json]

For each candidate category, samples up to ``--sample-size`` real images and
computes each one's structure-tensor coherence (``tnbbeta_vae.data.orientation``
-- how strongly and consistently oriented the image actually is, not a label),
then prints the categories sorted by mean coherence. This is the filter the
design doc (``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``,
Part 2) describes for deciding the final oriented-texture subset: inspect the
printed table (and the JSON this writes) and update
``tnbbeta_vae.data.dtd.ORIENTED_CATEGORIES`` yourself with whichever categories
your numbers support -- this script only measures and reports, it does not
decide or edit anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from tnbbeta_vae.data.dtd import DtdImages
from tnbbeta_vae.data.orientation import structure_tensor_orientation
from tnbbeta_vae.paths import data_dir

# Every candidate from the design doc: a dominant line with no head/tail.
_CANDIDATES = (
    "banded", "braided", "cracked", "fibrous", "grooved",
    "lined", "striped", "veined", "wrinkled", "zigzagged",
)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    """Measures and prints per-category mean structure-tensor coherence.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=list(_CANDIDATES))
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("dtd_category_coherence.json"))
    args = parser.parse_args(argv)

    root = args.data_dir or data_dir()
    base = datasets.DTD(
        root=str(root),
        split=args.split,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]
        ),
    )

    results = _measure(base, args)
    ranked = sorted(
        results.items(), key=lambda item: item[1]["mean_coherence"], reverse=True
    )
    print(f"{'category':<12} {'n':>6} {'mean':>8} {'std':>8}")
    for name, stats in ranked:
        print(
            f"{name:<12} {stats['num_sampled']:>6} {stats['mean_coherence']:>8.4f} "
            f"{stats['std_coherence']:>8.4f}"
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


def _measure(base: Any, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Computes per-category coherence statistics from a shared base dataset."""
    generator = torch.Generator().manual_seed(args.seed)
    results: dict[str, dict[str, float]] = {}
    for name in args.categories:
        dataset = DtdImages(base, categories=(name,))
        sample_size = min(args.sample_size, len(dataset))
        indices = torch.randperm(len(dataset), generator=generator)[:sample_size]
        subset = Subset(dataset, indices.tolist())
        loader = DataLoader(subset, batch_size=args.batch_size)
        coherences = []
        for batch in loader:
            _, coherence = structure_tensor_orientation(batch)
            coherences.append(coherence)
        values = torch.cat(coherences)
        results[name] = {
            "num_images": len(dataset),
            "num_sampled": len(values),
            "mean_coherence": float(values.mean()),
            "std_coherence": float(values.std()),
        }
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
