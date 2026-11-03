"""Tests for apps.eval.graph_families_table."""

from __future__ import annotations

import json
from pathlib import Path

from apps.eval.graph_families_table import collect, kind_table, summary_table


def _run(auc: float, kl: float) -> dict:
    run = {"oracle_test_auc": 0.9}
    for name in ("centre", "sampled"):
        run |= {
            f"{name}_test_auc": auc,
            f"{name}_test_log_loss": 0.6,
            f"{name}_test_ece": 0.05,
            f"{name}_truth_kl": kl,
            f"{name}_truth_kl_by_kind": {"a": kl, "b": kl * 2},
        }
    return run


def test_tables_summarize_conditions_for_one_graph(tmp_path: Path) -> None:
    for condition, auc in (("vmf_free", 0.7), ("vmf_k1", 0.8)):
        (tmp_path / f"gf_core_periphery_{condition}.json").write_text(
            json.dumps({"runs": [_run(auc, 0.1), _run(auc + 0.02, 0.3)]})
        )
    (tmp_path / "gf_bipartite_vmf_free.json").write_text(
        json.dumps({"runs": [_run(0.5, 0.1)]})
    )

    conditions = collect(tmp_path, "core_periphery")

    assert set(conditions) == {"vmf_free", "vmf_k1"}
    table = summary_table(conditions)
    assert "Oracle (true-probability) test AUC: 0.900" in table
    assert "| vmf_k1 | 2 | 0.810 ± 0.010 |" in table
    assert "| vmf_free | 0.200 | 0.400 |" in kind_table(conditions)
