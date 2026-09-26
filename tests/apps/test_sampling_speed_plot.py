"""Tests for apps.benchmark.sampling_speed_plot."""

from __future__ import annotations

import json
from pathlib import Path

from apps.benchmark import sampling_speed_plot


def _fake_records() -> list[dict]:
    records = []
    for dim in (2, 10, 40):
        for batch_size in (64, 1024):
            for concentration, kappa in (("weak", 1.0), ("extreme", 1000.0)):
                records.append(
                    {
                        "dim": dim,
                        "batch_size": batch_size,
                        "concentration": concentration,
                        "vmf_kappa": kappa,
                        "tnbbeta_epsilon": kappa,
                        "vmf_ms": {"mean": 5.0, "std": 0.1, "min": 4.8, "max": 5.2},
                        "tnbbeta_ms": {
                            "mean": 1.0,
                            "std": 0.05,
                            "min": 0.9,
                            "max": 1.1,
                        },
                        "vmf_over_tnbbeta": 5.0,
                    }
                )
    return records


def test_writes_a_png(tmp_path: Path) -> None:
    input_path = tmp_path / "bench.json"
    input_path.write_text(json.dumps(_fake_records()))
    output = tmp_path / "heatmap.png"

    sampling_speed_plot.main(["--input", str(input_path), "--output", str(output)])

    assert output.exists() and output.stat().st_size > 0


def test_figure_has_one_heatmap_trace_per_batch_size() -> None:
    figure = sampling_speed_plot._figure(_fake_records())

    assert len(figure.data) == 2  # pyright: ignore[reportArgumentType]  # batch sizes 64, 1024
    for trace in figure.data:
        assert trace.type == "heatmap"  # pyright: ignore[reportAttributeAccessIssue]
        assert len(trace.y) == 3  # pyright: ignore[reportAttributeAccessIssue]  # dims 2, 10, 40
        assert len(trace.x) == 2  # pyright: ignore[reportAttributeAccessIssue]  # weak, extreme
