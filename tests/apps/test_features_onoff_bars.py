"""Tests for apps.eval.features_onoff_bars."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from apps.eval import features_onoff_bars


def _trace_count(figure: go.Figure) -> int:
    """Returns the number of traces on ``figure`` (plotly's stubs don't type ``.data``
    as ``Sized``)."""
    return len(figure.data)  # pyright: ignore[reportArgumentType]


def test_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "features_onoff_bars.png"

    features_onoff_bars.main(["--output", str(output)])

    assert output.exists() and output.stat().st_size > 0


def test_build_figure_has_one_bar_trace_per_family_per_panel() -> None:
    data = {
        "auc": {
            "on": {
                "cora": {"gaussian": (90.0, 1.0), "vmf": (91.0, 1.0)},
                "citeseer": {"gaussian": (92.0, 1.0), "vmf": (93.0, 1.0)},
            },
        },
    }
    panels = [("auc", "on", "AUC — with features")]
    families = [("gaussian", "N-VGAE", "#7f7f7f"), ("vmf", "S-VGAE (vMF)", "#1f77b4")]
    datasets = ["cora", "citeseer"]

    figure = features_onoff_bars.build_figure(
        data,  # pyright: ignore[reportArgumentType]
        panels=panels,  # pyright: ignore[reportArgumentType]
        families=families,
        datasets=datasets,
        y_range=[80.0, 100.0],
    )

    assert _trace_count(figure) == len(families)


def test_bars_use_the_given_datasets_mean_and_label_order() -> None:
    data = {
        "ap": {
            "off": {
                "cora": {"gaussian": (85.0, 2.0)},
                "citeseer": {"gaussian": (80.0, 1.0)},
                "pubmed": {"gaussian": (82.0, 1.0)},
                "mag_cs": {"gaussian": (90.0, 1.0)},
            },
        },
    }
    panels = [("ap", "off", "AP — without features")]
    families = [("gaussian", "N-VGAE", "#7f7f7f")]

    figure = features_onoff_bars.build_figure(
        data,  # pyright: ignore[reportArgumentType]
        panels=panels,  # pyright: ignore[reportArgumentType]
        families=families,
        y_range=[70.0, 100.0],
    )

    bar = figure.data[0]
    assert list(bar.y) == [85.0, 80.0, 82.0, 90.0]  # pyright: ignore[reportAttributeAccessIssue]
    assert list(bar.x) == ["Cora", "Citeseer", "Pubmed", "MAG CS"]  # pyright: ignore[reportAttributeAccessIssue]


def test_error_bars_use_the_given_stds() -> None:
    data = {
        "auc": {
            "on": {
                "cora": {"gaussian": (90.0, 3.5)},
                "citeseer": {"gaussian": (91.0, 1.0)},
                "pubmed": {"gaussian": (92.0, 1.0)},
                "mag_cs": {"gaussian": (93.0, 1.0)},
            },
        },
    }
    panels = [("auc", "on", "AUC — with features")]
    families = [("gaussian", "N-VGAE", "#7f7f7f")]

    figure = features_onoff_bars.build_figure(
        data,  # pyright: ignore[reportArgumentType]
        panels=panels,  # pyright: ignore[reportArgumentType]
        families=families,
        y_range=[80.0, 100.0],
    )

    bar = figure.data[0]
    assert list(bar.error_y.array) == [3.5, 1.0, 1.0, 1.0]  # pyright: ignore[reportAttributeAccessIssue]
