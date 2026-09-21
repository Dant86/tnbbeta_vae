"""Tests for tnbbeta_vae.data.semi_supervised."""

from __future__ import annotations

import pytest
import torch

from tnbbeta_vae.data.mnist import MnistImages
from tnbbeta_vae.data.semi_supervised import (
    SemiSupervisedBatches,
    UnlabeledBatches,
    balanced_label_indices,
)


def _dataset(count: int = 40, *, dynamic: bool = True) -> MnistImages:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(count, 1, 28, 28, generator=generator)
    return MnistImages(images, torch.arange(count) % 10, dynamic=dynamic)


def test_balanced_indices_take_equal_counts_per_class_and_are_reproducible() -> None:
    labels = torch.arange(100) % 10

    chosen = balanced_label_indices(labels, 20, seed=3)

    assert len(set(chosen.tolist())) == 20
    assert torch.bincount(labels[chosen], minlength=10).tolist() == [2] * 10
    assert torch.equal(chosen, balanced_label_indices(labels, 20, seed=3))
    assert not torch.equal(chosen, balanced_label_indices(labels, 20, seed=4))


def test_balanced_indices_reject_a_count_the_classes_do_not_divide() -> None:
    with pytest.raises(ValueError, match="divisible by 10"):
        balanced_label_indices(torch.arange(100) % 10, 15, seed=0)


def test_batches_pair_every_label_with_fresh_unlabelled_examples() -> None:
    torch.manual_seed(0)
    batches = SemiSupervisedBatches(_dataset(), 10, 8, torch.device("cpu"), seed=0)

    steps = list(batches)

    assert len(batches) == len(steps) == 5
    first = steps[0]
    assert first.labeled_x.shape == (10, 784) and first.unlabeled_x.shape == (8, 784)
    assert sorted(first.labels.tolist()) == list(range(10))
    for step in steps:
        assert set(step.unlabeled_x.unique().tolist()) <= {0.0, 1.0}
    assert not torch.equal(steps[0].labeled_x, list(batches)[0].labeled_x)


def test_unlabelled_validation_batches_have_no_labelled_part() -> None:
    fixed = _dataset(25, dynamic=False)

    batches = list(UnlabeledBatches(fixed, 10, torch.device("cpu")))

    assert [len(batch) for batch in batches] == [10, 10, 5]
    assert all(batch.labeled_x.shape == (0, 784) for batch in batches)
    assert torch.equal(
        torch.cat([b.unlabeled_x for b in batches]), fixed.images.flatten(1)
    )
