"""Tests for apps.eval.fixed_m_sweep_table."""

from __future__ import annotations

import json
from pathlib import Path

from apps.eval import fixed_m_sweep_table


def _write(root: Path, name: str, auc: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    result = {"test_auc": auc, "test_ap": auc, "m_mean": -3.0, "frac_bimodal": 1.0}
    (run_dir / "sbm_recovery.json").write_text(json.dumps({"tnbbeta": result}))


def test_pools_seeds_and_sorts_free_first(tmp_path: Path) -> None:
    _write(tmp_path, "-3_seed0", 0.8)
    _write(tmp_path, "-3_seed1", 0.6)
    _write(tmp_path, "free_seed0", 0.7)
    _write(tmp_path, "-7_seed0", 0.5)

    table = fixed_m_sweep_table.render_table(fixed_m_sweep_table.collect(tmp_path))

    rows = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
    assert rows == ["free", "-7", "-3"]
    assert "0.7000 ± 0.1000" in table
    assert "n/a" in table  # temperature absent from the fake results


def test_skips_runs_without_a_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "0_seed0"
    run_dir.mkdir()
    (run_dir / "sbm_recovery.json").write_text(
        json.dumps({"tnbbeta": {"no_checkpoint": 1.0}})
    )

    assert fixed_m_sweep_table.collect(tmp_path) == {}
