"""Tests for the combined real-graph bar figure."""

from typing import Any

from apps.real_graph.combined_bars import (
    FAMILIES,
    FEATURES,
    MEAN_DEGREE,
    METRICS,
    build_figure,
)


def _cells() -> dict[str, Any]:
    stat = {"mean": 0.5, "ci": [0.4, 0.6]}
    return {
        f"{d}|{f}": {
            "families": {fam: {m: stat for m, _ in METRICS} for fam in FAMILIES}
        }
        for d in MEAN_DEGREE
        for f in FEATURES
    }


def test_panels_traces_and_order() -> None:
    figure = build_figure(_cells())
    assert len(figure.data) == len(FAMILIES) * len(FEATURES) * len(METRICS)
    first = list(figure.data[0].x)
    assert first[0].startswith("Citeseer") and first[-1].startswith("Squirrel")
    assert sum(trace.showlegend for trace in figure.data) == len(FAMILIES)


def test_no_bold_and_no_caption_text() -> None:
    layout: Any = build_figure(_cells()).layout
    for annotation in layout.annotations:
        assert "<b>" not in annotation.text
    assert "<b>" not in layout.title.text
