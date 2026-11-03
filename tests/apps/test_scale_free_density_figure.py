"""Tests for apps.eval.scale_free_density_figure."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr

from apps.eval import scale_free_density_figure as sdf
from apps.synthetic.graph_search import EXTENDED_STAGES, SEED_BASE, build_manifest


def _cell(
    gain: float, count: int = 24, seed: int = 0
) -> dict[str, dict[str, np.ndarray]]:
    """TNBBeta better by ``gain`` on every metric."""
    rng = np.random.default_rng(seed)
    cell: dict[str, dict[str, np.ndarray]] = {}
    for condition in ("tnbbeta", *sdf.PRIMARY):
        cell[condition] = {}
        for metric, _ in sdf.BAR_METRICS:
            base = 0.5 + 0.01 * rng.standard_normal(count)
            if condition == "tnbbeta":
                base = base + (gain if sdf.METRICS[metric].higher else -gain)
            elif condition == "power_spherical_init":
                base = base + 0.05  # worse than vmf for lower-better, better for higher
            cell[condition][metric] = base
    return cell


def test_configs_differ_only_in_mean_degree() -> None:
    configs = sdf.density_configs()
    assert [c["mean_degree"] for c in configs] == [6.0, 15.0, 35.0]
    for key in ("family", "num_nodes", "contrast", "features", "heterogeneity"):
        assert len({c[key] for c in configs}) == 1
    assert configs[0]["family"] == "scale_free"
    assert [c["config_id"] for c in configs] == [
        "scale_free-low",
        "scale_free-med",
        "scale_free-high",
    ]


def test_advantage_is_positive_when_tnbbeta_is_better_for_every_metric() -> None:
    cell = _cell(gain=0.1)
    for metric, _ in sdf.BAR_METRICS:
        assert np.all(sdf.advantage(cell, "vmf_init", metric) > 0.05)
    worse = _cell(gain=-0.1)
    for metric, _ in sdf.BAR_METRICS:
        assert np.all(sdf.advantage(worse, "vmf_init", metric) < -0.05)


def test_comparator_is_the_better_one_by_its_own_level_mean() -> None:
    cell = _cell(gain=0.0)
    keep = np.ones(24, dtype=bool)
    # PowerSpherical is +0.05 on every metric: worse if lower is better, else better.
    assert sdf.choose_comparator(cell, "truth_kl", keep) == "vmf_init"
    assert sdf.choose_comparator(cell, "centre_auc", keep) == "power_spherical_init"


def test_bootstrap_interval_contains_the_mean_and_dz_has_the_right_sign() -> None:
    values = np.random.default_rng(1).normal(0.3, 1.0, 40)
    out = sdf.bootstrap_mean_dz(values, np.random.default_rng(2), boots=500)
    assert out["mean_lo"] < out["mean"] < out["mean_hi"]
    assert out["dz_lo"] < out["dz"] < out["dz_hi"]
    assert out["dz"] > 0


def test_wilcoxon_handles_all_zero_and_detects_a_shift() -> None:
    assert sdf.wilcoxon_p(np.zeros(10)) == 1.0
    shifted = np.random.default_rng(0).normal(1.0, 0.2, 30)
    assert sdf.wilcoxon_p(shifted) < 1e-3


def test_level_effects_flags_a_clear_win_and_a_clear_loss() -> None:
    rng = np.random.default_rng(3)
    win = sdf.level_effects(_cell(gain=0.1), "truth_kl", rng)
    loss = sdf.level_effects(_cell(gain=-0.1), "sampled_auc", rng)
    assert win["mean"] > 0 and win["comparator"] == "vmf_init"
    assert loss["mean"] < 0
    assert win["draws"] == 24 and win["dropped"] == 0
    records = [win, loss]
    sdf.label_cells(records)
    assert [r["label"] for r in records] == ["better", "worse"]


def test_nonfinite_pairs_are_dropped() -> None:
    cell = _cell(gain=0.1)
    cell["tnbbeta"]["ece"][0] = np.nan
    record = sdf.level_effects(cell, "ece", np.random.default_rng(0))
    assert record["draws"] == 23 and record["dropped"] == 1


def _results(gain: float) -> dict[str, Any]:
    rng = np.random.default_rng(4)
    levels = {}
    for level in sdf.LEVELS:
        records = [
            {"level": level, **sdf.level_effects(_cell(gain, seed=i), m, rng)}
            for i, (m, _) in enumerate(sdf.BAR_METRICS)
        ]
        sdf.label_cells(records)
        levels[level] = records
    return {
        "levels": levels,
        "summary": {lv: sdf.panel_summary(levels[lv]) for lv in sdf.LEVELS},
        "level_covariates": {
            lv: {"realised_degree": 6.0, "expected_degree_cv": 1.0, "clustering": 0.1}
            for lv in sdf.LEVELS
        },
        "pooled": {
            "points": [
                {
                    "config": f"{f}-00",
                    "family": f,
                    "degree": 10.0 + i,
                    "nominal_degree": 10.0,
                    "E": 0.01 * i,
                }
                for i, f in enumerate(sdf.FAMILIES)
            ],
            "h1_spearman": 0.462,
        },
    }


def test_panel_summary_counts_labels() -> None:
    results = _results(0.1)
    text = results["summary"]["low"]
    assert "probability metrics better 3/3" in text and "AUROC better 2/2" in text


def test_within_family_spearman_matches_plain_spearman_for_one_family() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.random(30), rng.random(30)
    families = np.array(["x"] * 30)
    expected = float(cast("Any", spearmanr(a, b))[0])
    assert abs(sdf.within_family_spearman(a, b, families) - expected) < 1e-9


def test_layout_is_deterministic_bounded_and_separates_nodes() -> None:
    adjacency = np.zeros((10, 10))
    for i in range(9):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    first = sdf.fruchterman_reingold(adjacency, seed=7)
    second = sdf.fruchterman_reingold(adjacency, seed=7)
    assert np.array_equal(first, second)
    assert first.shape == (10, 2) and np.abs(first).max() <= 1.0 + 1e-9
    gaps = np.linalg.norm(first[:, None] - first[None], axis=-1)
    assert gaps[~np.eye(10, dtype=bool)].min() > 1e-2


def test_drawn_graphs_get_denser_with_the_level() -> None:
    means = [sdf.drawn_graph(level)["degree"].mean() for level in sdf.LEVELS]
    assert means[0] < means[1] < means[2]
    assert all(
        len(sdf.drawn_graph(lv)["degree"]) == sdf.DRAWN_NODES for lv in sdf.LEVELS
    )


def test_figure_layout_matches_the_preregistration() -> None:
    figure = sdf.build_figure(_results(0.1))
    layout: Any = figure.layout
    assert layout.legend.orientation == "h"
    assert layout.legend.y < 0
    assert layout.template.layout.font.family == sdf.FONT_FAMILY
    bars = [t for t in cast("list[Any]", list(figure.data)) if t.type == "bar"]
    assert len(bars) == 6  # two groups x three levels
    assert sum(len(b.x) for b in bars) == 15
    scatters = [
        t
        for t in cast("list[Any]", list(figure.data))
        if t.type == "scatter" and t.mode == "markers"
    ]
    assert len(scatters) == 3 + len(sdf.FAMILIES)  # drawn graphs + one per family
    text = " ".join(a.text for a in layout.annotations)
    for phrase in ("selection", "NOT varied", "NOT from the drawn graphs", "unbiased"):
        assert phrase.lower() in text.lower() or phrase in text


def test_density_stage_is_extended_and_has_its_own_seed_range() -> None:
    assert "density" in EXTENDED_STAGES
    assert SEED_BASE["density"] not in {
        v for k, v in SEED_BASE.items() if k != "density"
    }
    configs = sdf.density_configs()
    tuned = {
        "scale_free": {"tnbbeta": 0.01, "vmf_init": 0.03, "power_spherical_init": 0.003}
    }
    items = build_manifest(
        "density",
        configs,
        draws=2,
        conditions=["tnbbeta", "vmf_init", "power_spherical_init"],
        tuned_lr=tuned,
    )
    assert len(items) == 3 * 2 * 3
    assert all(item["extended"] for item in items)
    seeds = {(i["config_id"], i["draw"]): i["seed"] for i in items}
    assert seeds[("scale_free-low", 1)] == SEED_BASE["density"] + 1
    assert seeds[("scale_free-high", 0)] == SEED_BASE["density"] + 2000
    pairs = {(i["config_id"], i["draw"], i["seed"]) for i in items}
    assert len(pairs) == 6  # all conditions of a (config, draw) share the seed
