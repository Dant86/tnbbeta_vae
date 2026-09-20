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
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

__all__ = ["DeviceBatches", "MnistImages", "load_mnist"]

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

    @property
    def images(self) -> Tensor:
        """Returns the stored images: intensities if dynamic, else the fixed bits."""
        return self._images

    @property
    def dynamic(self) -> bool:
        """Returns whether images are re-binarized on every access."""
        return self._dynamic


class DeviceBatches:
    """Re-iterable batches of an :class:`MnistImages`, kept on one device.

    MNIST is small, so holding it on the GPU and binarizing each batch there avoids
    the per-image Python overhead of a ``DataLoader`` (which dominated training time).
    Dynamic datasets are resampled as Bernoulli(intensity) on every batch, using the
    global torch RNG (so ``torch.manual_seed`` makes runs reproducible).
    """

    def __init__(
        self,
        dataset: MnistImages,
        batch_size: int,
        device: torch.device,
        *,
        shuffle: bool,
        drop_last: bool = False,
    ) -> None:
        """Copies ``dataset`` to ``device``.

        Args:
            dataset: The images to batch.
            batch_size: Images per batch.
            device: Where the images live and the batches are produced.
            shuffle: Whether to reshuffle at the start of every iteration.
            drop_last: Whether to drop a final incomplete batch.
        """
        self._images = dataset.images.to(device)
        self._dynamic = dataset.dynamic
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __len__(self) -> int:
        """Returns the number of batches per pass."""
        full, rest = divmod(len(self._images), self._batch_size)
        return full + (0 if self._drop_last or rest == 0 else 1)

    def __iter__(self) -> Iterator[Tensor]:
        """Yields one pass of batches."""
        count = len(self._images)
        device = self._images.device
        order = torch.randperm(count, device=device) if self._shuffle else None
        stop = count - count % self._batch_size if self._drop_last else count
        for start in range(0, stop, self._batch_size):
            end = min(start + self._batch_size, count)
            indices = order[start:end] if order is not None else slice(start, end)
            batch = self._images[indices]
            yield torch.bernoulli(batch) if self._dynamic else batch


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
