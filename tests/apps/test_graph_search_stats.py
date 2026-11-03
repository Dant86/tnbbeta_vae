"""Tests for apps.eval.graph_search_stats."""

from __future__ import annotations

from typing import Any

import numpy as np

from apps.eval.graph_search_stats import (
    benjamini_hochberg,
    holm,
    one_sided_p,
    paired_differences,
    pooled_family_tests,
    rejected_pairs,
    required_draws,
    run_config_tests,
)


def _runs(
    effect: dict[str, float], draws: int = 16, seed: int = 0
) -> list[dict[str, Any]]:
    """TNBBeta, vmf_init and power_spherical_init runs; log loss is lower-is-better."""
    rng = np.random.default_rng(seed)
    runs = []
    for config_id, gain in effect.items():
        for draw in range(draws):
            base = 0.5 + 0.02 * rng.standard_normal()
            for condition, shift in (
                ("vmf_init", 0.0),
                ("power_spherical_init", 0.0),
                ("tnbbeta", -gain),
            ):
                noise = 0.002 * rng.standard_normal()
                metrics = {
                    "centre_auc": 0.8,
                    "sampled_log_loss": base + shift + noise,
                    "cal_slope_dev": 0.1,
                    "uncertainty_auroc": 0.5,
                    "truth_kl": 0.1,
                    "rank_corr": 0.5,
                }
                runs.append(
                    {
                        "config_id": config_id,
                        "draw": draw,
                        "condition": condition,
                        "lr": 0.01,
                        "family": config_id.rsplit("-", 1)[0],
                        "metrics": metrics,
                    }
                )
    return runs


def test_benjamini_hochberg_matches_a_hand_computed_example() -> None:
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])

    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_holm_matches_a_hand_computed_example_and_is_at_least_bh() -> None:
    p = [0.01, 0.04, 0.03, 0.005]

    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert all(
        h >= b - 1e-12 for h, b in zip(holm(p), benjamini_hochberg(p), strict=True)
    )


def test_one_sided_p_needs_the_margin_to_be_cleared() -> None:
    differences = [0.004] * 15 + [0.0041]

    assert one_sided_p(differences, margin=0.0) < 1e-3
    assert one_sided_p(differences, margin=0.005) > 0.9  # real but below the margin
    assert one_sided_p([-0.01] * 16, margin=0.0, better=False) < 1e-3


def test_paired_differences_orient_lower_better_metrics_and_drop_nonfinite() -> None:
    runs = _runs({"ring-00": 0.02}, draws=4)
    runs[0]["metrics"]["sampled_log_loss"] = float("nan")

    diffs = paired_differences(runs, "tnbbeta", "vmf_init", "sampled_log_loss")

    assert len(diffs["ring-00"]) <= 4
    assert all(d > 0.01 for d in diffs["ring-00"])  # tnbbeta lower loss -> positive


def test_intersection_union_needs_both_comparators_to_lose() -> None:
    runs = _runs({"ring-00": 0.03, "ring-01": 0.0})
    for run in runs:  # make power_spherical_init as good as tnbbeta on ring-00
        if run["config_id"] == "ring-00" and run["condition"] == "power_spherical_init":
            run["metrics"]["sampled_log_loss"] -= 0.03
    rows = run_config_tests(runs, ("vmf_init", "power_spherical_init"))
    loss = {r["config_id"]: r for r in rows if r["metric"] == "sampled_log_loss"}

    assert loss["ring-00"]["p"] > 0.05  # not better than power_spherical_init
    assert loss["ring-00"]["p_by_comparator"]["vmf_init"] < 1e-3
    assert loss["ring-01"]["p"] > 0.05


def test_a_real_effect_is_detected_and_a_null_is_not_across_many_tests() -> None:
    effect = {f"ring-{i:02d}": (0.03 if i < 3 else 0.0) for i in range(12)}
    rows = [
        r
        for r in run_config_tests(_runs(effect), ("vmf_init", "power_spherical_init"))
        if r["metric"] == "sampled_log_loss"
    ]
    q = benjamini_hochberg([r["p"] for r in rows])
    hits = {r["config_id"] for r, v in zip(rows, q, strict=True) if v <= 0.05}

    assert hits == {"ring-00", "ring-01", "ring-02"}


def test_pooled_family_test_pools_configs_and_required_draws_scales_with_variance() -> (
    None
):
    effect = {f"ring-{i:02d}": 0.03 for i in range(8)}
    pooled = pooled_family_tests(_runs(effect), ("vmf_init", "power_spherical_init"))
    loss = next(r for r in pooled if r["metric"] == "sampled_log_loss")

    assert loss["configs"] == 8 and loss["p"] < 0.01
    assert required_draws(0.02, 0.01, 1e-3) > required_draws(0.01, 0.01, 1e-3)


def test_rejected_pairs_selects_the_bh_significant_ones_in_order() -> None:
    rows = [
        {"config_id": "b-01", "metric": "m1", "q_bh": 0.01},
        {"config_id": "a-00", "metric": "m2", "q_bh": 0.2},
        {"config_id": "a-00", "metric": "m1", "q_bh": 0.05},
    ]

    assert rejected_pairs(rows, 0.05) == [("a-00", "m1"), ("b-01", "m1")]
