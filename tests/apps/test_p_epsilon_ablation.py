"""Tests for apps.distributions.p_epsilon_ablation's CLI (grid/layout logic is
tested in test_ablation_grid.py, which this reuses)."""

from __future__ import annotations

from pathlib import Path

from apps.distributions import p_epsilon_ablation


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "p_epsilon_ablation.png"

    p_epsilon_ablation.main(
        ["--output", str(output), "--resolution", "6", "--curve-points", "20"]
    )

    assert output.exists() and output.stat().st_size > 0


def test_p_values_and_epsilon_values_override_the_default_grid(
    tmp_path: Path,
) -> None:
    output = tmp_path / "p_epsilon_ablation_2x2.png"

    p_epsilon_ablation.main(
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
            "--epsilon-values",
            "0.5",
            "10.0",
        ]
    )

    assert output.exists() and output.stat().st_size > 0
