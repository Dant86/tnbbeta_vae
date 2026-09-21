"""Tests for apps/semi_supervised/table.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.semi_supervised import table


def _write(
    tmp_path: Path, variant: str, z1: int, z2: int, accuracies: list[float]
) -> None:
    for seed, accuracy in enumerate(accuracies):
        run_dir = tmp_path / "ckpt" / f"semi_{variant}_z{z1}_{z2}_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "semi_supervised_final.json").write_text(
            json.dumps({"test_accuracy": accuracy})
        )


def test_table_reports_percent_bolds_significant_winners_and_counts_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    _write(tmp_path, "nn", 5, 5, [0.70, 0.71, 0.69, 0.70])
    _write(tmp_path, "ss", 5, 5, [0.80, 0.81, 0.79, 0.80])
    _write(tmp_path, "nn", 5, 10, [0.75, 0.76])

    table.main(
        ["--variants", "nn", "ss", "--dims", "5", "10", "--seeds", "0", "1", "2", "3"]
    )

    captured = capsys.readouterr()
    rows = captured.out.strip().splitlines()
    assert rows[0] == "| z1 dim | z2 dim | N+N | S+S |"
    assert rows[2] == "| 5 | 5 | 70.00 ± 0.82 | **80.00 ± 0.82** |"
    assert "| 5 | 10 | 75.50 ± 0.71 | - |" in rows[3]
    assert "run(s) had no results file" in captured.err
