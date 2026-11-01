"""Tests for apps.eval.graph_search_explore."""

from __future__ import annotations

from typing import Any

import numpy as np

from apps.eval.graph_search_explore import advantage, aggregate, disadvantage, report
from tnbbeta_vae.data.graph_space import FAMILIES

_CONDITIONS = (
    "tnbbeta",
    "vmf_init",
    "power_spherical_init",
    "gaussian",
    "vmf_pin",
    "power_spherical_pin",
)


def _metrics(loss: float, rng: np.random.Generator) -> dict[str, float]:
    return {
        "centre_auc": 0.8 + 0.01 * rng.standard_normal(),
        "sampled_auc": 0.8,
        "sampled_log_loss": loss + 0.01 * rng.standard_normal(),
        "cal_slope": 1.0,
        "cal_slope_dev": 0.1 + 0.01 * rng.standard_normal(),
        "uncertainty_auroc": 0.5,
        "truth_kl": 0.1 + 0.01 * rng.standard_normal(),
        "rank_corr": 0.5,
        "post_r_bar": 0.7,
        "second_mode_mass": 0.0,
    }


def _stage(tnb_gain: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(0)
    configs, runs = [], []
    for family in FAMILIES[:2]:
        spec = {
            "config_id": f"{family}-00",
            "family": family,
            "num_nodes": 400,
            "mean_degree": 10.0,
            "num_groups": 4,
            "heterogeneity": "mild",
            "features": "none",
            "achieved_auc": 0.8,
        }
        configs.append(spec)
        for draw in range(10):
            for condition in _CONDITIONS:
                loss = 0.6 - (tnb_gain if condition == "tnbbeta" else 0.0)
                runs.append(
                    {
                        "config_id": spec["config_id"],
                        "draw": draw,
                        "condition": condition,
                        "metrics": _metrics(loss, rng),
                        "diverged": 0.0,
                    }
                )
    return runs, configs


def test_aggregate_orients_effects_so_positive_means_tnbbeta_is_better() -> None:
    runs, configs = _stage(tnb_gain=0.05)
    records = aggregate(runs, configs)
    record = records["assortative-00"]

    assert record["effect"]["vmf_init"]["sampled_log_loss"]["mean"] > 0.03
    assert advantage(record, "sampled_log_loss") > 1.0
    assert disadvantage(record, "sampled_log_loss") > 1.0
    assert abs(advantage(record, "centre_auc")) < 1.5  # no real effect
    assert record["mean"]["tnbbeta"]["sampled_log_loss"] < 0.57


def test_a_worse_tnbbeta_gives_a_negative_advantage() -> None:
    runs, configs = _stage(tnb_gain=-0.05)
    record = aggregate(runs, configs)["assortative-00"]

    assert advantage(record, "sampled_log_loss") < -1.0


def test_report_renders_every_section() -> None:
    runs, configs = _stage(tnb_gain=0.05)
    text = report(aggregate(runs, configs))

    for heading in ("### 1.", "### 2.", "### 3.", "### 4.", "### 5.", "### 6."):
        assert heading in text
