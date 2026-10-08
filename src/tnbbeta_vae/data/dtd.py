"""DTD (Describable Textures Dataset) image loading via ``torchvision``.

Filtered, by default, to the oriented-texture category subset this project
trains on -- see
``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``
(Part 2) and ``apps/data/curate_dtd_categories.py``, which measured the
structure-tensor coherence that decided ``ORIENTED_CATEGORIES``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["DtdImages", "ORIENTED_CATEGORIES", "load_dtd"]

# Candidate axial (no head/tail, orientation defined mod pi) texture categories --
# docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md Part 2.
# Placeholder: apps/data/curate_dtd_categories.py measures real structure-tensor
# coherence per candidate and this gets narrowed to the decided subset afterward
# (see that script's commit message for the numbers behind the final cut).
ORIENTED_CATEGORIES: tuple[str, ...] = (
    "banded",
    "braided",
    "cracked",
    "fibrous",
    "grooved",
    "lined",
    "striped",
    "veined",
    "wrinkled",
    "zigzagged",
)


def load_dtd(
    root: str | Path,
    *,
    split: Literal["train", "val", "test"],
    image_size: int = 64,
    categories: tuple[str, ...] = ORIENTED_CATEGORIES,
    download: bool = False,
) -> DtdImages:
    """Loads DTD as a dataset of image tensors, filtered to ``categories``.

    Args:
        root: Directory DTD is stored in (or downloaded to).
        split: Which split to load (DTD defines train/val/test, each a
            fixed 1/3 partition of the data -- unlike CIFAR-10/MNIST's
            train/test, there is no separate validation split to carve out).
        image_size: Height/width images are resized to (square).
        categories: Texture category names to keep. DTD has 47 in total;
            this project trains only on the oriented-texture subset.
        download: Whether to download the data if it isn't already in
            ``root``. Off by default so training jobs never touch the
            network; use ``apps/data/download_dtd.py`` once instead.

    Returns:
        A dataset yielding float32 RGB images of shape
        ``(3, image_size, image_size)`` with values in ``[0, 1]``.
    """
    from torchvision import datasets, transforms

    base = datasets.DTD(
        root=str(root),
        split=split,
        download=download,
        transform=transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        ),
    )
    return DtdImages(base, categories)


class DtdImages(Dataset):
    """Wraps a torchvision ``DTD`` dataset, filtered to a set of categories."""

    def __init__(self, base: Any, categories: tuple[str, ...]) -> None:
        """Wraps ``base``, keeping only images in ``categories``.

        Args:
            base: A torchvision ``DTD`` dataset (or a fake with the same
                ``classes``/``class_to_idx``/``__getitem__`` shape), yielding
                ``(image, label)`` pairs.
            categories: Category names to keep.
        """
        self._base = base
        keep = {base.class_to_idx[name] for name in categories}
        labels = _labels_of(base)
        self._kept = [index for index, label in enumerate(labels) if label in keep]

    def __len__(self) -> int:
        """Returns the number of kept images."""
        return len(self._kept)

    def __getitem__(self, index: int) -> Tensor:
        """Returns kept image ``index`` as a ``(3, image_size, image_size)`` tensor."""
        return cast("Any", self._base)[self._kept[index]][0]

    def labels(self) -> Tensor:
        """Returns the DTD category index per kept image, in dataset order."""
        labels = _labels_of(self._base)
        return torch.as_tensor([labels[i] for i in self._kept], dtype=torch.long)


def _labels_of(base: Any) -> list[int]:
    """Returns ``base``'s per-item label index, without relying on iteration order."""
    labels = getattr(base, "_labels", None)
    if labels is None:
        labels = [base[i][1] for i in range(len(base))]
    return labels
