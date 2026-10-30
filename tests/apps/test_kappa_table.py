"""Tests for apps.eval.kappa_table."""

from __future__ import annotations

import json
from pathlib import Path

from apps.eval.kappa_table import auc_table, collect, dynamics_table


def _write(directory: Path, stem: str, auc: float, trajectory: list[dict]) -> None:
    (directory / f"{stem}.json").write_text(json.dumps({"selected": {"test_auc": auc}}))
    (directory / f"{stem}_trajectory.json").write_text(json.dumps({"cfg": trajectory}))


def test_collect_labels_conditions_and_tables_summarize_them(tmp_path: Path) -> None:
    free = [
        {"epoch": -1.0, "val_auc": 0.7, "kappa_mean": 1.7, "r_bar": 0.2},
        {
            "epoch": 0.0,
            "val_auc": 0.7,
            "kappa_mean": 1.7,
            "r_bar": 0.2,
            "grad_norm_kappa_row": 0.1,
            "grad_norm_direction_rows": 0.4,
        },
    ]
    _write(tmp_path, "kdyn_matched_vmf_cora_seed0", 0.75, free)
    _write(tmp_path, "kdyn_matched_vmf_cora_seed1", 0.77, free)
    _write(tmp_path, "fixedk_cora_vmf_x2_seed0", 0.85, [])
    _write(tmp_path, "kdyn_matched_vmf_pubmed_seed0", 0.8, free)

    conditions = collect(tmp_path, "cora")

    assert set(conditions) == {"matched_vmf", "vmf x2"}
    assert len(conditions["matched_vmf"]) == 2
    assert "| matched_vmf | 2 | 0.760 ± 0.010 |" in auc_table(conditions)
    dynamics = dynamics_table(conditions)
    assert "matched_vmf" in dynamics
    assert "vmf x2" not in dynamics  # no trajectory rows to report
    assert "0.25" in dynamics  # 0.1 / 0.4 gradient ratio at epoch 0
