"""Tests for apps.eval.graph_search_metric_first."""

from __future__ import annotations

import numpy as np

from apps.eval import graph_search_metric_first as mf


def _table(shift: float, draws: int = 20, seed: int = 0) -> mf.Table:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.3, 0.01, draws)
    table: mf.Table = {}
    for config, effect in (("ring-00", shift), ("ring-01", 0.0), ("ring-02", -shift)):
        tnb = base + rng.normal(0, 0.005, draws)
        comparators = {
            c: base + effect + rng.normal(0, 0.005, draws) for c in mf.PRIMARY
        }
        mrr = {
            "tnbbeta": 0.5 + rng.normal(0, 0.005, draws) - effect,
            **{c: 0.5 + rng.normal(0, 0.005, draws) for c in mf.PRIMARY},
        }
        table[config] = {
            "tnbbeta": {"truth_kl": tnb, "hard_mrr": mrr["tnbbeta"]},
            **{c: {"truth_kl": comparators[c], "hard_mrr": mrr[c]} for c in mf.PRIMARY},
        }
    return table


def test_orientation_and_the_less_favourable_comparator() -> None:
    table = _table(0.05)
    lower = mf.oriented_difference(table, "ring-00", "vmf_init", "truth_kl")
    higher = mf.oriented_difference(table, "ring-00", "vmf_init", "hard_mrr")
    rng = np.random.default_rng(0)
    record = mf.config_metric_record(table, "ring-00", "truth_kl", mf.PRIMARY, rng)

    assert lower.mean() > 0.04  # comparator has larger KL: TNBBeta better
    assert higher.mean() < 0  # TNBBeta's MRR was shifted down by the effect
    assert record is not None
    assert record["E"] == min(record["mean_d"].values())
    assert record["E_lo"] <= record["E"] <= record["E_hi"]
    assert record["dz_min"] == min(record["dz"].values())


def test_classification_labels_better_worse_and_neither() -> None:
    table = _table(0.05)
    records = mf.all_records(table, ("truth_kl", "hard_mrr"), mf.PRIMARY)
    mf.classify(records, fresh=True)
    labels = {(r["config_id"], r["metric"]): r["label"] for r in records}

    assert labels[("ring-00", "truth_kl")] == "better"
    assert labels[("ring-01", "truth_kl")] == "neither"
    assert labels[("ring-02", "truth_kl")] == "worse"
    assert labels[("ring-00", "hard_mrr")] == "worse"  # opposite effect on MRR
    summary = mf.family_metric_tables(records, ("truth_kl", "hard_mrr"))
    assert "| ring | 1/1/1 | 1/1/1 |" in summary


def test_confirmatory_metrics_use_holm_and_others_bh() -> None:
    table = _table(0.05)
    records = mf.all_records(table, ("truth_kl", "hard_mrr"), mf.PRIMARY)
    mf.classify(records, fresh=True)
    holm_rows = [r for r in records if r["metric"] == "truth_kl"]
    # Holm over 3 tests: the smallest p is multiplied by 3
    smallest = min(holm_rows, key=lambda r: r["p"])
    assert smallest["adj"] >= smallest["p"] * 3 - 1e-12 or smallest["adj"] == 1.0


def test_block_clusters_group_collinear_columns() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(60)
    x = np.stack([a, a + 0.01 * rng.standard_normal(60), rng.standard_normal(60)], 1)

    assert mf.clusters_of(x, ["a", "b", "c"]) == [[0, 1], [2]]
