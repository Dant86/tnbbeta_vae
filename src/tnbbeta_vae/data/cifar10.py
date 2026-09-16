"""CIFAR-10 data loading (not yet implemented).

Loading CIFAR-10 will most likely use ``torchvision.datasets.CIFAR10``,
which is not yet a project dependency -- add it (``uv add torchvision``)
when this is implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ["load_cifar10"]


def load_cifar10(root: str, *, train: bool) -> Dataset:
    """Loads the CIFAR-10 dataset.

    Args:
        root: Directory to download/read CIFAR-10 from.
        train: Whether to load the train split (``False`` for test).

    Raises:
        NotImplementedError: Always -- not yet implemented.
    """
    raise NotImplementedError("CIFAR-10 loading is not yet implemented.")
