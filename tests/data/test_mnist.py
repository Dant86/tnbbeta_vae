"""Tests for tnbbeta_vae.data.mnist (with a tiny fake torchvision MNIST)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision

from tnbbeta_vae.data import mnist
from tnbbeta_vae.data.mnist import DeviceBatches, MnistImages, load_mnist


def test_dynamic_images_are_resampled_and_fixed_ones_are_not() -> None:
    images = torch.full((4, 1, 28, 28), 0.5)
    labels = torch.arange(4)
    dynamic = MnistImages(images, labels, dynamic=True)
    fixed = MnistImages(images, labels, dynamic=False)

    assert not torch.equal(dynamic[0], dynamic[0])
    assert torch.equal(fixed[0], fixed[0])
    for dataset in (dynamic, fixed):
        assert set(dataset[1].unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(fixed.labels(), labels)
    assert len(fixed) == 4


def test_fixed_binarization_is_reproducible() -> None:
    images = torch.rand(6, 1, 28, 28)
    first = MnistImages(images, torch.arange(6), dynamic=False, seed=3)
    second = MnistImages(images, torch.arange(6), dynamic=False, seed=3)

    assert torch.equal(first[2], second[2])


class _FakeMNIST:
    def __init__(self, root: str, train: bool, download: bool) -> None:
        count = 100 if train else 20
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randint(0, 256, (count, 28, 28), generator=generator).byte()
        self.targets = torch.arange(count) % 10
        self.download = download


def test_load_mnist_splits_train_val_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.datasets, "MNIST", _FakeMNIST)
    monkeypatch.setattr(mnist, "_TRAIN_SIZE", 80)

    train = load_mnist(Path("unused"), split="train")
    val = load_mnist(Path("unused"), split="val")
    test = load_mnist(Path("unused"), split="test")

    assert (len(train), len(val), len(test)) == (80, 20, 20)
    assert train[0].shape == (1, 28, 28)
    assert set(torch.cat([train.labels(), val.labels()]).tolist()) == set(range(10))
    assert any(not torch.equal(train[3], train[3]) for _ in range(5))
    assert torch.equal(val[0], val[0]) and torch.equal(test[0], test[0])
    again = load_mnist(Path("unused"), split="val")
    assert torch.equal(val.labels(), again.labels())


def _intensities(count: int) -> torch.Tensor:
    return torch.full((count, 1, 28, 28), 0.5)


def test_device_batches_cover_every_image_once_and_respect_drop_last() -> None:
    images = torch.arange(10).float().view(10, 1, 1, 1).expand(10, 1, 28, 28) / 10
    fixed = MnistImages(images, torch.arange(10), dynamic=False)
    device = torch.device("cpu")

    keep = DeviceBatches(fixed, 4, device, shuffle=False)
    drop = DeviceBatches(fixed, 4, device, shuffle=False, drop_last=True)

    assert [len(batch) for batch in keep] == [4, 4, 2] and len(keep) == 3
    assert [len(batch) for batch in drop] == [4, 4] and len(drop) == 2
    assert torch.equal(torch.cat(list(keep)), fixed.images)


def test_device_batches_shuffle_each_pass_and_resample_dynamic_pixels() -> None:
    torch.manual_seed(0)
    dynamic = MnistImages(_intensities(64), torch.arange(64), dynamic=True)
    batches = DeviceBatches(dynamic, 16, torch.device("cpu"), shuffle=True)

    first = torch.cat(list(batches))
    second = torch.cat(list(batches))

    assert set(first.unique().tolist()) <= {0.0, 1.0}
    assert not torch.equal(first, second)
    assert 0.4 < first.mean() < 0.6


def test_fixed_device_batches_are_identical_across_passes() -> None:
    fixed = MnistImages(torch.rand(20, 1, 28, 28), torch.arange(20), dynamic=False)
    batches = DeviceBatches(fixed, 8, torch.device("cpu"), shuffle=False)

    assert torch.equal(torch.cat(list(batches)), torch.cat(list(batches)))
