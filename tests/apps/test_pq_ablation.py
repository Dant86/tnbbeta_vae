"""Tests for apps.distributions.pq_ablation's CLI (grid/layout logic is tested
in test_ablation_grid.py, which this reuses)."""

from __future__ import annotations

from pathlib import Path

from apps.distributions import pq_ablation


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "pq_ablation.png"

    pq_ablation.main(
        ["--output", str(output), "--resolution", "6", "--curve-points", "20"]
    )

    assert output.exists() and output.stat().st_size > 0


def test_p_values_and_q_values_override_the_default_grid(tmp_path: Path) -> None:
    output = tmp_path / "pq_ablation_2x2.png"

    pq_ablation.main(
        [
            "--output",
            str(output),
            "--resolution",
            "6",
            "--curve-points",
            "20",
            "--p-values",
            "0.5",
            "0.95",
            "--q-values",
            "0.0",
            "0.9",
        ]
    )

    assert output.exists() and output.stat().st_size > 0
