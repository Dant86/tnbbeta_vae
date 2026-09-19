"""CIFAR-10 image loading via ``torchvision``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sized
    from pathlib import Path

    from torch import Tensor

__all__ = ["Cifar10Images", "load_cifar10"]


def load_cifar10(
    root: str | Path, *, train: bool, download: bool = False
) -> Cifar10Images:
    """Loads CIFAR-10 as a dataset of image tensors.

    Args:
        root: Directory CIFAR-10 is stored in (or downloaded to).
        train: Whether to load the train split (``False`` for test).
        download: Whether to download the data if it isn't already in
            ``root``. Off by default so training jobs never touch the
            network; use ``apps/data/download_cifar10.py`` once instead.

    Returns:
        A dataset yielding float32 images of shape ``(3, 32, 32)`` with
        values in ``[0, 1]`` (labels are dropped).
    """
    from torchvision import datasets, transforms

    base = datasets.CIFAR10(
        root=str(root), train=train, download=download, transform=transforms.ToTensor()
    )
    return Cifar10Images(base)


class Cifar10Images(Dataset):
    """Wraps an ``(image, label)`` dataset to yield just the image."""

    def __init__(self, base: Sized) -> None:
        """Wraps ``base``, an indexable dataset of ``(image, label)`` pairs."""
        self._base = base

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self._base)

    def __getitem__(self, index: int) -> Tensor:
        """Returns image ``index`` as a ``(3, 32, 32)`` tensor in ``[0, 1]``."""
        return cast("Any", self._base)[index][0]
