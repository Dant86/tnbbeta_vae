"""Tests for tnbbeta_vae.data.dtd (torchvision mocked; no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import pytest
import torch

from tnbbeta_vae.data.dtd import DtdImages, load_dtd


class _FakeDtd:
    """Mimics torchvision's ``DTD``: real PIL images + an applied transform."""

    calls: list[dict[str, Any]] = []
    classes = ["banded", "bumpy", "striped"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    # banded, bumpy, banded, striped, striped.
    _labels = [0, 1, 0, 2, 2]

    def __init__(
        self, root: str, split: str, download: bool, transform: Any = None
    ) -> None:
        type(self).calls.append({"root": root, "split": split, "download": download})
        self._transform = transform

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = Image.new("RGB", (300, 300), color=(index * 10, 0, 0))
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[index]


class _FakeDtdWithoutPrivateLabels:
    """Mimics a torchvision ``DTD`` whose ``_labels`` attribute is missing."""

    classes = ["banded", "bumpy"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    _items = [0, 1, 0]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        return torch.zeros(3, 4, 4), self._items[index]


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    _FakeDtd.calls.clear()


def test_filters_to_categories_resizes_and_forwards_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.DTD", _FakeDtd)

    dataset = load_dtd(
        tmp_path, split="val", categories=("banded", "striped"), image_size=32
    )

    assert len(dataset) == 4  # 2 banded + 2 striped; "bumpy" (index 1) dropped.
    for index in range(len(dataset)):
        image = dataset[index]
        assert image.shape == (3, 32, 32)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    assert dataset.labels().tolist() == [0, 0, 2, 2]
    assert _FakeDtd.calls == [
        {"root": str(tmp_path), "split": "val", "download": False}
    ]


def test_download_flag_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.DTD", _FakeDtd)

    load_dtd(tmp_path, split="train", categories=("banded",), download=True)

    assert _FakeDtd.calls[0]["download"] is True


def test_default_image_size_is_64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.DTD", _FakeDtd)

    dataset = load_dtd(tmp_path, split="test", categories=("striped",))

    assert dataset[0].shape == (3, 64, 64)


def test_falls_back_to_iterating_getitem_when_private_labels_is_missing() -> None:
    base = _FakeDtdWithoutPrivateLabels()

    dataset = DtdImages(base, categories=("banded",))

    assert len(dataset) == 2
    assert dataset.labels().tolist() == [0, 0]
