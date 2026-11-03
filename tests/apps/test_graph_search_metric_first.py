"""Tests for the ``fresh`` (metric-first) stage of apps.synthetic.graph_search."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from apps.synthetic import graph_search
from apps.synthetic.graph_search import SEED_BASE, build_manifest, run_item

LEGACY_KEYS = {
    "centre_auc",
    "sampled_auc",
    "sampled_log_loss",
    "cal_slope",
    "cal_intercept",
    "cal_slope_dev",
    "uncertainty_auroc",
    "truth_kl",
    "recal_log_loss",
    "recal_truth_kl",
    "rank_corr",
}
EXTENDED_KEYS = {
    "ece",
    "hard_hits10",
    "hard_hits50",
    "hard_mrr",
    "hardc_hits10",
    "hardc_hits50",
    "hardc_mrr",
    "x_truth_kl",
    "truth_kl_eps1e4",
    "truth_kl_w",
    "gs_degree_w1",
    "gs_clustering_err",
    "gs_triangle_err",
    "latent_gram_rho",
    "latent_procrustes",
    "latent_nmi",
    "temperature",
    "logit_bias",
    "kappa_log_mean",
}


def _config() -> dict[str, Any]:
    return {
        "config_id": "ring-00",
        "family": "ring",
        "num_nodes": 80,
        "mean_degree": 8.0,
        "num_groups": 4,
        "heterogeneity": "none",
        "features": "none",
        "contrast": 20.0,
        "target_auc": 0.8,
        "achieved_auc": 0.8,
    }


def _tuned() -> dict[str, dict[str, float]]:
    names = ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
    return {"ring": {name: 0.01 for name in names}}


def test_fresh_stage_has_its_own_disjoint_seed_range() -> None:
    others = [v for k, v in SEED_BASE.items() if k != "fresh"]

    assert SEED_BASE["fresh"] == 50_000_000
    assert all(abs(SEED_BASE["fresh"] - v) >= 10_000_000 for v in others)


def test_only_fresh_manifest_items_are_extended_and_old_items_keep_their_keys() -> None:
    old = build_manifest("discovery", [_config()], draws=1, tuned_lr=_tuned())
    new = build_manifest(
        "fresh",
        [_config()],
        draws=2,
        conditions=["tnbbeta", "vmf_init", "power_spherical_pin"],
        tuned_lr=_tuned(),
    )

    base_keys = {"run_id", "config_id", "draw", "seed", "condition", "lr", "num_nodes"}
    assert all(set(item) == base_keys for item in old)
    assert all(set(item) == base_keys | {"extended", "checkpoint"} for item in new)
    assert len(new) == 2 * 3
    assert all(item["seed"] >= 50_000_000 for item in new)
    assert all(item["checkpoint"].startswith("graph_search/fresh/") for item in new)


@pytest.mark.parametrize("condition", ["tnbbeta", "vmf_init", "power_spherical_init"])
def test_extended_run_adds_metrics_and_leaves_legacy_ones_unchanged(
    condition: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(graph_search, "EPOCHS", 6)
    monkeypatch.setattr(graph_search, "SAMPLES", 4)
    monkeypatch.setattr(graph_search, "checkpoint_dir", lambda: tmp_path)
    item = {
        "run_id": "r",
        "config_id": "ring-00",
        "draw": 0,
        "seed": 5,
        "condition": condition,
        "lr": 0.01,
        "num_nodes": 80,
    }
    plain = run_item(item, _config(), torch.device("cpu"))
    extended = run_item(
        {**item, "extended": True, "checkpoint": "graph_search/fresh/checkpoints/r.pt"},
        _config(),
        torch.device("cpu"),
    )

    assert set(plain["metrics"]) >= LEGACY_KEYS
    assert not (set(plain["metrics"]) & EXTENDED_KEYS)
    assert set(extended["metrics"]) >= LEGACY_KEYS | EXTENDED_KEYS
    for key in LEGACY_KEYS:
        assert extended["metrics"][key] == pytest.approx(
            plain["metrics"][key], nan_ok=True
        ), key
    metrics = extended["metrics"]
    for key in ("ece", "hard_mrr", "x_truth_kl", "truth_kl_w", "gs_degree_w1"):
        assert math.isfinite(metrics[key]), key
    assert 0.0 < metrics["hard_mrr"] <= 1.0
    assert metrics["x_truth_kl"] == pytest.approx(metrics["truth_kl"], rel=0.5)
    assert math.isfinite(metrics["latent_gram_rho"])  # ring has a latent
    assert math.isnan(metrics["latent_nmi"])
    if condition == "tnbbeta":
        assert math.isnan(metrics["kappa_log_mean"])
    else:
        assert math.isfinite(metrics["kappa_log_mean"])
    assert (tmp_path / "graph_search/fresh/checkpoints/r.pt").exists()
