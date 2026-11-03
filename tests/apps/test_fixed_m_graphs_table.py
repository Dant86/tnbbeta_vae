"""Tests for apps.eval.fixed_m_graphs_table."""

from __future__ import annotations

import json
from pathlib import Path

from apps.eval import fixed_m_graphs_table


def _write(root: Path, name: str, auc: float, bucket_auc: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    result = {
        "overall": {"auc": auc, "ap": auc},
        "by_degree": {"0": {"auc": bucket_auc}, "1": {"auc": bucket_auc + 0.1}},
    }
    (run_dir / "bucketed_link_prediction.json").write_text(json.dumps(result))


def test_collect_groups_seeds_by_condition_and_orders_free_first(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "fixedmg_cora_0.5_seed0", 0.8, 0.7)
    _write(tmp_path, "fixedmg_cora_0.5_seed1", 0.6, 0.5)
    _write(tmp_path, "fixedmg_cora_free_seed0", 0.7, 0.6)
    _write(tmp_path, "fixedmg_cora_-0.9_seed0", 0.5, 0.4)
    _write(tmp_path, "fixedmg_citeseer_free_seed0", 0.9, 0.9)

    runs = fixed_m_graphs_table.collect(tmp_path, "cora")
    table = fixed_m_graphs_table.overall_table(runs)

    assert sorted(runs) == ["-0.9", "0.5", "free"]
    rows = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
    assert rows == ["free", "-0.9", "0.5"]
    assert "0.7000 ± 0.1000" in table


def test_bucket_table_reports_one_column_per_bucket(tmp_path: Path) -> None:
    _write(tmp_path, "fixedmg_cora_free_seed0", 0.7, 0.6)

    runs = fixed_m_graphs_table.collect(tmp_path, "cora")
    table = fixed_m_graphs_table.bucket_table(runs, "by_degree")

    assert "b0" in table and "b1" in table
    assert "0.600" in table and "0.700" in table
