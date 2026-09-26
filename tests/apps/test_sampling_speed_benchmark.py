"""Tests for apps.benchmark.sampling_speed (CPU, small scope for speed)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from apps.benchmark import sampling_speed


def test_cli_writes_one_record_per_cell(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"

    sampling_speed.main(
        [
            "--dims",
            "3",
            "5",
            "--batch-sizes",
            "8",
            "--concentrations",
            "weak",
            "strong",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]  # fmt: skip
    )

    records = json.loads(output.read_text())
    assert len(records) == 4  # 2 dims x 1 batch size x 2 concentrations
    cells = {(r["dim"], r["concentration"]) for r in records}
    assert cells == {(3, "weak"), (3, "strong"), (5, "weak"), (5, "strong")}


def test_each_record_has_both_families_stats_and_a_positive_ratio(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bench.json"

    sampling_speed.main(
        [
            "--dims",
            "5",
            "--batch-sizes",
            "16",
            "--concentrations",
            "moderate",
            "--repeats",
            "3",
            "--warmup",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]  # fmt: skip
    )

    (record,) = json.loads(output.read_text())
    for family in ("vmf_ms", "tnbbeta_ms"):
        stats = record[family]
        for key in ("mean", "std", "min", "max"):
            assert stats[key] >= 0.0
        assert stats["min"] <= stats["mean"] <= stats["max"]
    assert record["vmf_over_tnbbeta"] > 0.0


def test_time_rsample_returns_one_stat_per_repeat_regardless_of_family() -> None:
    device = torch.device("cpu")
    dist = sampling_speed._tnbbeta(dim=4, batch_size=8, epsilon=5.0, device=device)

    stats = sampling_speed._time_rsample(dist, repeats=10, warmup=2, device=device)

    assert set(stats) == {"mean", "std", "min", "max"}
    assert stats["mean"] >= 0.0
