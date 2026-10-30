"""Tests for apps/data/curate_dtd_categories.py (torchvision mocked; no network)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from apps.data import curate_dtd_categories

_SIZE = 32


class _FakeDtd:
    """A striped "oriented" category and a noisy "unoriented" one."""

    classes = ["oriented", "unoriented"]
    class_to_idx = {"oriented": 0, "unoriented": 1}

    def __init__(
        self, root: str, split: str, download: bool, transform: object = None
    ) -> None:
        generator = torch.Generator().manual_seed(0)
        ys, xs = torch.meshgrid(
            torch.arange(_SIZE, dtype=torch.float32),
            torch.arange(_SIZE, dtype=torch.float32),
            indexing="ij",
        )
        stripes = (torch.cos(2 * math.pi * 6 / _SIZE * xs) + 1) / 2
        stripes = stripes.unsqueeze(0).expand(3, -1, -1)
        self._images = [stripes] * 10 + [
            torch.rand(3, _SIZE, _SIZE, generator=generator) for _ in range(10)
        ]
        self._labels = [0] * 10 + [1] * 10

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._images[index], self._labels[index]


def test_ranks_the_oriented_category_above_the_unoriented_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.DTD", _FakeDtd)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "coherence.json"

    curate_dtd_categories.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--categories",
            "oriented",
            "unoriented",
            "--sample-size",
            "10",
            "--image-size",
            str(_SIZE),
            "--out",
            str(out),
        ]  # fmt: skip
    )

    results = json.loads(out.read_text())
    assert set(results) == {"oriented", "unoriented"}
    assert results["oriented"]["num_sampled"] == 10
    assert (
        results["oriented"]["mean_coherence"] > results["unoriented"]["mean_coherence"]
    )
    assert results["unoriented"]["mean_coherence"] < 0.3


def test_sample_size_is_capped_by_the_category_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchvision.datasets.DTD", _FakeDtd)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "coherence.json"

    curate_dtd_categories.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--categories",
            "oriented",
            "--sample-size",
            "1000",
            "--image-size",
            str(_SIZE),
            "--out",
            str(out),
        ]  # fmt: skip
    )

    results = json.loads(out.read_text())
    assert results["oriented"]["num_images"] == 10
    assert results["oriented"]["num_sampled"] == 10
