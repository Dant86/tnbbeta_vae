"""Tests for the edge-thinning line figure."""

from typing import Any

import numpy as np

from apps.real_graph.thinning_lines import (
    DATASETS,
    FAMILIES,
    FEATURES,
    KEEPS,
    build_figure,
    build_per_graph_figure,
)


def _arrays() -> dict[tuple[Any, ...], dict[str, Any]]:
    values = np.linspace(0.2, 0.3, 10)
    return {
        (d, f, fam, None if keep == 1.0 else keep, None, None): {"log_loss": values}
        for d in DATASETS
        for f in FEATURES
        for fam in FAMILIES
        for keep in KEEPS
    }


def test_one_line_per_family_per_panel() -> None:
    figure = build_figure(_arrays())
    assert len(figure.data) == len(FAMILIES) * len(FEATURES)
    assert sum(trace.showlegend for trace in figure.data) == len(FAMILIES)
    assert list(figure.data[0].x) == list(KEEPS)


def test_no_bold_and_column_titles() -> None:
    layout: Any = build_figure(_arrays()).layout
    texts = [annotation.text for annotation in layout.annotations]
    assert texts == ["With Node Features", "Without Node Features"]


def test_per_graph_columns_are_graphs() -> None:
    figure = build_per_graph_figure(_arrays())
    layout: Any = figure.layout
    titles = [a.text for a in layout.annotations if a.textangle != 90]
    assert titles == ["Amazon Computers", "Amazon Photo", "Squirrel"]
    assert len(figure.data) == len(FAMILIES) * len(FEATURES) * len(DATASETS)
    assert sum(trace.showlegend for trace in figure.data) == len(FAMILIES)
