"""Labelled/unlabelled MNIST batches for semi-supervised training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tnbbeta_vae.models.semi_supervised import SemiBatch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tnbbeta_vae.data.mnist import MnistImages

__all__ = ["SemiSupervisedBatches", "UnlabeledBatches", "balanced_label_indices"]


def balanced_label_indices(
    labels: torch.Tensor, num_labels: int, seed: int
) -> torch.Tensor:
    """Picks ``num_labels`` examples, the same number from every class.

    Args:
        labels: Integer class of every training example.
        num_labels: How many labelled examples to keep (divisible by the class count).
        seed: Seed for the choice.

    Returns:
        Indices into ``labels``.

    Raises:
        ValueError: If ``num_labels`` is not divisible by the number of classes.
    """
    classes = int(labels.max().item()) + 1
    if num_labels % classes != 0:
        raise ValueError(
            f"num_labels must be divisible by {classes}; got {num_labels}."
        )
    generator = torch.Generator().manual_seed(seed)
    per_class = num_labels // classes
    chosen = []
    for cls in range(classes):
        members = torch.nonzero(labels == cls).squeeze(-1)
        order = torch.randperm(len(members), generator=generator)
        chosen.append(members[order[:per_class]])
    return torch.cat(chosen)


class SemiSupervisedBatches:
    """Each step: every labelled example plus a batch of unlabelled ones.

    The unlabelled pool is the whole training set with its labels hidden (as in Kingma
    et al.), so the labelled examples appear in it too. Images stay on ``device`` and
    are re-binarized (Bernoulli of the intensity) at every step, flattened to vectors.
    """

    def __init__(
        self,
        dataset: MnistImages,
        num_labels: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        """Chooses the labelled subset and copies the data to ``device``.

        Args:
            dataset: The dynamically binarized training set.
            num_labels: Number of labelled examples (class balanced).
            batch_size: Unlabelled examples per step.
            device: Where the data lives.
            seed: Seed for the labelled subset.
        """
        labels = dataset.labels()
        chosen = balanced_label_indices(labels, num_labels, seed)
        self._images = dataset.images.flatten(1).to(device)
        self._labeled = self._images[chosen.to(device)]
        self._labels = labels[chosen].to(device)
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Returns the number of steps per pass over the unlabelled pool."""
        return len(self._images) // self._batch_size

    def __iter__(self) -> Iterator[SemiBatch]:
        """Yields one pass of steps over a fresh shuffle of the unlabelled pool."""
        order = torch.randperm(len(self._images), device=self._images.device)
        for step in range(len(self)):
            start = step * self._batch_size
            unlabeled = self._images[order[start : start + self._batch_size]]
            yield SemiBatch(
                torch.bernoulli(self._labeled),
                self._labels,
                torch.bernoulli(unlabeled),
            )


class UnlabeledBatches:
    """Validation batches with no labelled examples (early stopping uses no labels)."""

    def __init__(
        self, dataset: MnistImages, batch_size: int, device: torch.device
    ) -> None:
        """Copies a fixed-binarized dataset to ``device`` as flat vectors."""
        self._images = dataset.images.flatten(1).to(device)
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[SemiBatch]:
        """Yields the dataset once, in order."""
        empty_x = self._images[:0]
        empty_y = torch.zeros(0, dtype=torch.long, device=self._images.device)
        for start in range(0, len(self._images), self._batch_size):
            yield SemiBatch(
                empty_x, empty_y, self._images[start : start + self._batch_size]
            )
