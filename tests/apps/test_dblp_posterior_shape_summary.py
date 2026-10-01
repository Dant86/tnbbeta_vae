"""Tests for the com-DBLP posterior shape summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

from apps.eval.dblp_posterior_shape_summary import aggregate, load_results


def _write_record(
    directory: Path,
    *,
    bridge_stats: dict[str, float],
    non_bridge_stats: dict[str, float],
) -> None:
    directory.mkdir(parents=True)
    record = {
        "bridge_nodes": {"posterior_stats": bridge_stats},
        "non_bridge_nodes": {"posterior_stats": non_bridge_stats},
    }
    (directory / "bridge_diagnostic_final.json").write_text(json.dumps(record))


def test_load_results_collects_every_seed_per_family(tmp_path: Path) -> None:
    """Each seed's bridge/non-bridge statistics land in the right family's lists."""
    for seed in (0, 1):
        _write_record(
            tmp_path / f"dblp_tnbbeta_d16_seed{seed}",
            bridge_stats={"r_bar": 0.5 + seed, "p_mean": 0.6},
            non_bridge_stats={"r_bar": 0.4 + seed, "p_mean": 0.3},
        )
    _write_record(
        tmp_path / "dblp_vmf_d16_seed0",
        bridge_stats={"entropy_mean": 1.2, "r_bar": 0.1},
        non_bridge_stats={"entropy_mean": 1.3, "r_bar": 0.1},
    )

    seeds = load_results(tmp_path)

    assert seeds["tnbbeta"]["r_bar"]["bridge"] == [0.5, 1.5]
    assert seeds["tnbbeta"]["r_bar"]["non_bridge"] == [0.4, 1.4]
    assert seeds["tnbbeta"]["p_mean"]["bridge"] == [0.6, 0.6]
    # vMF never reports p_mean -- its list stays empty, not filled with anything.
    assert seeds["vmf"]["p_mean"]["bridge"] == []
    assert seeds["vmf"]["entropy_mean"]["bridge"] == [1.2]


def test_aggregate_skips_statistics_no_seed_reported() -> None:
    """A family missing a statistic entirely (e.g. (p, q, epsilon) for vMF) gets no
    row for it, rather than a row full of placeholders."""
    family_seeds = {
        "entropy_mean": {"bridge": [1.2, 1.3], "non_bridge": [1.1, 1.2]},
        "r_bar": {"bridge": [], "non_bridge": []},
        "p_mean": {"bridge": [], "non_bridge": []},
        "q_mean": {"bridge": [], "non_bridge": []},
        "epsilon_mean": {"bridge": [], "non_bridge": []},
        "m_mean": {"bridge": [], "non_bridge": []},
        "frac_bimodal": {"bridge": [], "non_bridge": []},
    }

    rows = aggregate(family_seeds, alpha=0.01)

    assert [row["statistic"] for row in rows] == ["entropy_mean"]


def test_aggregate_skips_an_all_nan_statistic() -> None:
    """TNBBeta's entropy_mean is always NaN (no closed form) -- confirms it's
    dropped rather than shown as a meaningless "nan +/- nan" row."""
    family_seeds = {
        "entropy_mean": {"bridge": [math.nan, math.nan], "non_bridge": [math.nan]},
        "r_bar": {"bridge": [0.5, 0.6], "non_bridge": [0.3, 0.35]},
        "p_mean": {"bridge": [], "non_bridge": []},
        "q_mean": {"bridge": [], "non_bridge": []},
        "epsilon_mean": {"bridge": [], "non_bridge": []},
        "m_mean": {"bridge": [], "non_bridge": []},
        "frac_bimodal": {"bridge": [], "non_bridge": []},
    }

    rows = aggregate(family_seeds, alpha=0.01)

    assert [row["statistic"] for row in rows] == ["r_bar"]


def test_aggregate_reports_n_a_for_a_zero_variance_statistic() -> None:
    """A statistic identical across every seed in both groups (TNBBeta's
    frac_bimodal was exactly 1.0 everywhere in the real run) can't support a
    Welch t-test (zero combined variance) -- reported as "n/a", not a crash or a
    fabricated p-value."""
    family_seeds = {
        "entropy_mean": {"bridge": [], "non_bridge": []},
        "r_bar": {"bridge": [], "non_bridge": []},
        "p_mean": {"bridge": [], "non_bridge": []},
        "q_mean": {"bridge": [], "non_bridge": []},
        "epsilon_mean": {"bridge": [], "non_bridge": []},
        "m_mean": {"bridge": [], "non_bridge": []},
        "frac_bimodal": {"bridge": [1.0, 1.0], "non_bridge": [1.0, 1.0]},
    }

    rows = aggregate(family_seeds, alpha=0.01)

    assert rows[0]["statistic"] == "frac_bimodal"
    assert rows[0]["p_value"] == "n/a"


def test_aggregate_bolds_a_significant_p_value() -> None:
    """A p-value below alpha is bolded in the display string."""
    family_seeds = {
        "entropy_mean": {"bridge": [], "non_bridge": []},
        "r_bar": {"bridge": [0.9, 0.91, 0.92], "non_bridge": [0.1, 0.11, 0.12]},
        "p_mean": {"bridge": [], "non_bridge": []},
        "q_mean": {"bridge": [], "non_bridge": []},
        "epsilon_mean": {"bridge": [], "non_bridge": []},
        "m_mean": {"bridge": [], "non_bridge": []},
        "frac_bimodal": {"bridge": [], "non_bridge": []},
    }

    rows = aggregate(family_seeds, alpha=0.01)

    assert rows[0]["p_value"].startswith("**")
