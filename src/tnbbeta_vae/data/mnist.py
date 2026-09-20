"""MNIST loading with the binarization protocol of the S-VAE paper.

Davidson et al. (2018) train on dynamically binarized MNIST: every time a
training image is drawn, each pixel is resampled as Bernoulli(intensity). The
validation and test images are binarized once with a fixed seed, so the numbers
are reproducible and comparable across models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["MnistImages", "load_mnist"]

_TRAIN_SIZE = 50_000
_SPLIT_SEED = 0
_BINARIZATION_SEED = 1


class MnistImages(Dataset):
    """Binarized MNIST images of shape ``(1, 28, 28)`` with values in {0, 1}."""

    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        *,
        dynamic: bool,
        seed: int = _BINARIZATION_SEED,
    ) -> None:
        """Wraps grayscale images.

        Args:
            images: Intensities in [0, 1], shape ``(n, 1, 28, 28)``.
            labels: Integer class labels, shape ``(n,)``.
            dynamic: If True, resample the binarization on every access (using the
                global torch RNG, so DataLoader workers draw fresh samples each
                epoch). If False, binarize once with ``seed``.
            seed: Seed for the fixed binarization.
        """
        self._labels = labels
        self._dynamic = dynamic
        if dynamic:
            self._images = images
        else:
            generator = torch.Generator().manual_seed(seed)
            self._images = torch.bernoulli(images, generator=generator)

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self._images)

    def __getitem__(self, index: int) -> Tensor:
        """Returns image ``index``, freshly binarized if the dataset is dynamic."""
        image = self._images[index]
        return torch.bernoulli(image) if self._dynamic else image

    def labels(self) -> Tensor:
        """Returns all class labels (0-9), in dataset order, as an int64 tensor."""
        return self._labels


def load_mnist(
    root: str | Path,
    *,
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> MnistImages:
    """Loads one split of MNIST.

    The 60,000 training images are split into 50,000 train and 10,000
    validation images by a fixed permutation, so every run sees the same split.

    Args:
        root: Directory MNIST is stored in (or downloaded to).
        split: ``"train"`` (dynamically binarized), ``"val"`` or ``"test"``
            (both binarized once, with a fixed seed).
        download: Whether to download the data if it isn't already in ``root``.
            Off by default so training jobs never touch the network; use
            ``apps/data/download_mnist.py`` once instead.

    Returns:
        The requested split.
    """
    from torchvision import datasets

    base = datasets.MNIST(root=str(root), train=split != "test", download=download)
    images = (base.data.float() / 255.0).unsqueeze(1)
    labels = base.targets.long()
    if split != "test":
        order = torch.randperm(
            len(images), generator=torch.Generator().manual_seed(_SPLIT_SEED)
        )
        chosen = order[:_TRAIN_SIZE] if split == "train" else order[_TRAIN_SIZE:]
        images, labels = images[chosen], labels[chosen]
    return MnistImages(images, labels, dynamic=split == "train")
