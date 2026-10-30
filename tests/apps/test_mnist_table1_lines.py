"""Tests for apps.eval.mnist_table1_lines."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from apps.eval import mnist_table1_lines


def _trace_count(figure: go.Figure) -> int:
    """Returns the number of traces on ``figure`` (plotly's stubs don't type ``.data``
    as ``Sized``)."""
    return len(figure.data)  # pyright: ignore[reportArgumentType]


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "mnist_table1_lines.png"

    mnist_table1_lines.main(["--output", str(output)])

    assert output.exists() and output.stat().st_size > 0


def _toy_dim(value: float, std: float) -> dict[str, tuple[float, float]]:
    """All 4 metric keys, each (value, std) -- build_figure's grid is fixed at 4."""
    return {
        "ll": (value, std),
        "elbo": (value, std),
        "re": (value, std),
        "kl": (value, std),
    }


def test_build_figure_has_one_trace_set_per_model_per_panel() -> None:
    data = {
        "gaussian": {2: _toy_dim(-1.0, 0.1), 5: _toy_dim(-2.0, 0.2)},
        "vmf": {2: _toy_dim(-1.5, 0.1)},  # dimension 5 intentionally omitted
    }
    models = [
        ("gaussian", "Gaussian", "#7f7f7f", True, "circle"),
        ("vmf", "vMF", "#1f77b4", False, "square"),
    ]

    figure = mnist_table1_lines.build_figure(data, models=models, dims=[2, 5])

    # 3 traces per model per panel (lower band, upper band, line); 4 panels, 2 models.
    assert _trace_count(figure) == 4 * 2 * 3


def test_vmf_line_skips_dimensions_it_has_no_data_for() -> None:
    data = {
        "gaussian": {2: _toy_dim(-1.0, 0.1), 5: _toy_dim(-2.0, 0.2)},
        "vmf": {2: _toy_dim(-1.5, 0.1)},
    }
    models = [
        ("gaussian", "Gaussian", "#7f7f7f", True, "circle"),
        ("vmf", "vMF", "#1f77b4", False, "square"),
    ]

    figure = mnist_table1_lines.build_figure(data, models=models, dims=[2, 5])

    # First panel's vMF line trace (3rd trace of its group: lower, upper, line).
    vmf_line = figure.data[5]
    assert vmf_line.name == "vMF"  # pyright: ignore[reportAttributeAccessIssue]
    assert list(vmf_line.x) == ["2"]  # pyright: ignore[reportAttributeAccessIssue]
    assert vmf_line.marker.symbol == "square"  # pyright: ignore[reportAttributeAccessIssue]
