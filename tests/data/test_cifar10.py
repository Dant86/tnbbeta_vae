"""Tests for tnbbeta_vae.data.cifar10 (torchvision mocked; no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tnbbeta_vae.data.cifar10 import load_cifar10


class _FakeCifar10:
    calls: list[dict[str, Any]] = []

    def __init__(self, root: str, train: bool, download: bool, transform: Any) -> None:
        type(self).calls.append({"root": root, "train": train, "download": download})
        self._transform = transform

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.full((3, 32, 32), index / 10), index


def test_yields_images_without_labels_and_forwards_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.CIFAR10", _FakeCifar10)
    _FakeCifar10.calls.clear()

    dataset = load_cifar10(tmp_path, train=False)

    assert len(dataset) == 5
    assert dataset[3].shape == (3, 32, 32)
    assert torch.allclose(dataset[3], torch.full((3, 32, 32), 0.3))
    assert _FakeCifar10.calls == [
        {"root": str(tmp_path), "train": False, "download": False}
    ]


def test_download_flag_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.CIFAR10", _FakeCifar10)
    _FakeCifar10.calls.clear()

    load_cifar10(tmp_path, train=True, download=True)

    assert _FakeCifar10.calls[0]["download"] is True
