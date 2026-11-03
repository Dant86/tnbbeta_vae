"""Tests for apps.synthetic.graph_search."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from apps.synthetic import graph_search
from apps.synthetic.graph_search import (
    CONDITIONS,
    build_manifest,
    model_config_for,
    run_item,
    sample_configs,
)
from tnbbeta_vae.data.graph_space import FAMILIES
from tnbbeta_vae.models import GraphVAE


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


def test_sampled_configs_are_reachable_calibrated_and_cover_each_family() -> None:
    configs = sample_configs("unit", per_family=2)

    assert {c["family"] for c in configs} == set(FAMILIES)
    assert all(abs(c["achieved_auc"] - c["target_auc"]) <= 0.02 for c in configs)
    assert all(
        300 <= c["num_nodes"] <= 1200 and 4 <= c["mean_degree"] <= 40 for c in configs
    )
    assert len({c["config_id"] for c in configs}) == len(configs)
    assert configs == sample_configs("unit", per_family=2)  # reproducible


def test_pilot_manifest_crosses_conditions_with_learning_rates_and_pairs_seeds() -> (
    None
):
    items = build_manifest("pilot", [_config()], draws=2)

    assert len(items) == 1 * 2 * 4 * 3
    assert len({i["run_id"] for i in items}) == len(items)
    by_draw = {(i["draw"], i["seed"]) for i in items}
    assert len(by_draw) == 2  # every condition of a draw shares its seed


def test_discovery_manifest_uses_tuned_rates_and_secondaries_inherit_theirs() -> None:
    tuned = {
        "ring": {
            c: 0.01 for c in ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
        }
    }
    tuned["ring"]["vmf_init"] = 0.03
    items = build_manifest("discovery", [_config()], draws=1, tuned_lr=tuned)
    rate = {i["condition"]: i["lr"] for i in items}

    assert set(rate) == {c for c in CONDITIONS if "_k" not in c}
    assert rate["vmf_init"] == rate["vmf_pin"] == rate["vmf_default"] == 0.03
    assert rate["power_spherical_pin"] == 0.01


def test_conditions_build_the_intended_models() -> None:
    vmf_init = model_config_for("vmf_init", 10)
    assert vmf_init.initial_kappa == 16.0 and vmf_init.fixed_kappa is None
    assert model_config_for("vmf_pin", 10).fixed_kappa == 16.0
    assert model_config_for("gaussian", 10).latent_dim == 15
    assert all(model_config_for(c, 10).decoder_bias for c in CONDITIONS)
    for condition in CONDITIONS:
        GraphVAE(model_config_for(condition, 10))  # all construct


@pytest.mark.parametrize("condition", ["tnbbeta", "vmf_init", "gaussian"])
def test_run_item_reports_every_metric_finite(
    condition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_search, "EPOCHS", 8)
    monkeypatch.setattr(graph_search, "SAMPLES", 4)
    item = {
        "run_id": "x",
        "config_id": "ring-00",
        "draw": 0,
        "seed": 3,
        "condition": condition,
        "lr": 0.01,
        "num_nodes": 80,
    }

    result = run_item(item, _config(), torch.device("cpu"))
    metrics = result["metrics"]

    for key in (
        "centre_auc",
        "sampled_auc",
        "sampled_log_loss",
        "cal_slope_dev",
        "truth_kl",
        "rank_corr",
    ):
        assert math.isfinite(metrics[key]), key
    assert 0.0 <= metrics["centre_auc"] <= 1.0
    assert result["seconds"] > 0


def test_kappa_grid_stage_adds_fixed_kappa_conditions_at_the_init_rate() -> None:
    tuned = {"ring": {c: 0.01 for c in ("tnbbeta", "vmf_init", "gaussian")}}
    tuned["ring"]["vmf_init"] = 0.03
    tuned["ring"]["power_spherical_init"] = 0.003
    items = build_manifest("kappa_grid", [_config()], draws=1, tuned_lr=tuned)
    rate = {i["condition"]: i["lr"] for i in items}

    assert rate["vmf_k64"] == 0.03
    assert rate["power_spherical_k4"] == 0.003
    assert len({i["seed"] for i in items}) == 1


def test_axis_diagnostic_stage_reuses_the_init_rates_with_its_own_seeds() -> None:
    tuned = {"ring": {c: 0.01 for c in ("tnbbeta", "vmf_init", "gaussian")}}
    tuned["ring"]["power_spherical_init"] = 0.003
    items = build_manifest(
        "axis_diagnostic",
        [_config()],
        draws=1,
        conditions=["tnbbeta", "power_spherical_k4"],
        tuned_lr=tuned,
    )

    assert {i["condition"]: i["lr"] for i in items}["power_spherical_k4"] == 0.003
    assert all(i["seed"] >= 40_000_000 for i in items)
