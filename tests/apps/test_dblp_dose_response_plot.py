"""Tests for apps.eval.dblp_dose_response_plot."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from apps.eval import dblp_dose_response_plot as plot


def _trace_count(figure: go.Figure) -> int:
    """Returns the number of traces on ``figure`` (plotly's stubs don't type ``.data``
    as ``Sized``)."""
    return len(figure.data)  # pyright: ignore[reportArgumentType]


def _write_diagnostic(
    results_dir: Path, family: str, seed: int, by_count: dict[str, dict[str, float]]
) -> None:
    run_dir = results_dir / f"dblp_{family}_d16_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bridge_diagnostic_final.json").write_text(
        json.dumps({"link_prediction_by_community_count": by_count})
    )


def _fixture_by_count(auc: float, ap: float, count: int) -> dict[str, dict[str, float]]:
    return {
        bucket: {"auc": auc, "ap": ap, "count": count}
        for bucket in ["0", "1", "2", "3", "4", "5+"]
    }


def test_load_results_collects_every_seed_per_family(tmp_path: Path) -> None:
    for seed, auc in enumerate([0.8, 0.9]):
        _write_diagnostic(tmp_path, "tnbbeta", seed, _fixture_by_count(auc, 0.7, 100))
    _write_diagnostic(
        tmp_path, "power_spherical", 0, _fixture_by_count(0.75, 0.65, 100)
    )
    _write_diagnostic(tmp_path, "vmf", 0, _fixture_by_count(0.7, 0.6, 100))

    per_family, counts = plot.load_results(tmp_path)

    assert per_family["tnbbeta"]["auc"]["0"] == [0.8, 0.9]
    assert per_family["power_spherical"]["auc"]["0"] == [0.75]
    assert per_family["vmf"]["ap"]["5+"] == [0.6]
    assert counts["0"] == 100


def test_build_figure_has_one_band_set_plus_line_per_family_per_metric_plus_bars(
    tmp_path: Path,
) -> None:
    for family, auc, ap in [
        ("tnbbeta", 0.9, 0.8),
        ("power_spherical", 0.85, 0.75),
        ("vmf", 0.84, 0.74),
    ]:
        _write_diagnostic(tmp_path, family, 0, _fixture_by_count(auc, ap, 100))
        _write_diagnostic(tmp_path, family, 1, _fixture_by_count(auc, ap, 100))

    per_family, counts = plot.load_results(tmp_path)
    figure = plot.build_figure(per_family, counts)

    # Per metric (2): 3 families x 3 traces (lower band, upper band, line) = 9.
    # Edge count is a single third panel, not duplicated per metric: 9*2 + 1 = 19.
    assert _trace_count(figure) == 19
    legend_entries = [
        trace.name  # pyright: ignore[reportAttributeAccessIssue]
        for trace in figure.data
        if trace.showlegend  # pyright: ignore[reportAttributeAccessIssue]
    ]
    assert legend_entries == ["TNBBeta", "Power Spherical", "vMF"]


def test_main_writes_a_png(tmp_path: Path) -> None:
    for family in ["tnbbeta", "power_spherical", "vmf"]:
        _write_diagnostic(tmp_path, family, 0, _fixture_by_count(0.9, 0.8, 50))

    output = tmp_path / "dose_response.png"
    plot.main(["--results-dir", str(tmp_path), "--output", str(output)])

    assert output.exists() and output.stat().st_size > 0
