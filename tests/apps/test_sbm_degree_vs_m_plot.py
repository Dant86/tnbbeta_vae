"""Tests for apps.eval.sbm_degree_vs_m_plot."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from apps.eval import sbm_degree_vs_m_plot


def _trace_count(figure: go.Figure) -> int:
    """Returns the number of traces on ``figure`` (plotly's stubs don't type ``.data``
    as ``Sized``)."""
    return len(figure.data)  # pyright: ignore[reportArgumentType]


def _write_raw(tmp_path: Path, raw: dict[str, dict[str, list[float]]]) -> Path:
    path = tmp_path / "degree_vs_m_raw.json"
    path.write_text(json.dumps(raw))
    return path


def test_load_raw_round_trips_the_json(tmp_path: Path) -> None:
    raw = {"0.005": {"degree": [1.0, 2.0], "m": [-4.0, -3.0]}}
    path = _write_raw(tmp_path, raw)

    assert sbm_degree_vs_m_plot.load_raw(path) == raw


def test_writes_a_png(tmp_path: Path) -> None:
    raw = {
        "0.005": {"degree": [10.0, 12.0, 14.0, 16.0], "m": [-4.5, -4.3, -4.1, -3.9]},
        "0.05": {"degree": [30.0, 33.0, 36.0, 39.0], "m": [-5.2, -5.1, -5.0, -4.9]},
    }
    input_path = _write_raw(tmp_path, raw)
    output = tmp_path / "sbm_degree_vs_m.png"

    sbm_degree_vs_m_plot.main(["--input", str(input_path), "--output", str(output)])

    assert output.exists() and output.stat().st_size > 0


def test_build_figure_has_a_scatter_and_a_fit_line_per_p_out() -> None:
    raw = {
        "0.005": {"degree": [10.0, 12.0, 14.0, 16.0], "m": [-4.5, -4.3, -4.1, -3.9]},
        "0.05": {"degree": [30.0, 33.0, 36.0, 39.0], "m": [-5.2, -5.1, -5.0, -4.9]},
    }

    figure = sbm_degree_vs_m_plot.build_figure(raw, p_out_order=["0.005", "0.05"])

    # One scatter + one fit-line trace per p_out level present.
    assert _trace_count(figure) == 4


def test_legend_label_includes_the_pearson_r() -> None:
    # A perfectly linear relationship: r should round to 1.00.
    raw = {"0.005": {"degree": [1.0, 2.0, 3.0, 4.0], "m": [-4.0, -3.0, -2.0, -1.0]}}

    figure = sbm_degree_vs_m_plot.build_figure(raw, p_out_order=["0.005"])

    scatter = figure.data[0]
    assert scatter.name == "p_out=0.005 (r=1.00)"  # pyright: ignore[reportAttributeAccessIssue]


def test_skips_p_out_levels_not_present_in_the_data() -> None:
    raw = {"0.005": {"degree": [1.0, 2.0], "m": [-4.0, -3.0]}}

    figure = sbm_degree_vs_m_plot.build_figure(
        raw, p_out_order=["0.005", "0.01", "0.05"]
    )

    assert _trace_count(figure) == 2
