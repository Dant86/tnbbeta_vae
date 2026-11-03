"""Smoke test for the linear-decoder ring figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.synthetic import linear_decoder_ring as study
from apps.synthetic import linear_decoder_ring_plot as plot


def test_metric_figure_builds_from_tiny_runs(tmp_path: Path) -> None:
    for arm in study.ARMS:
        study.main(
            [
                "train", "--decoder", "L", "--dim", "3", "--sigma", "0.2",
                "--arm", arm, "--seed", "0", "--epochs", "1", "--num-train", "128",
                "--num-test", "50", "--device", "cpu", "--out-dir", str(tmp_path),
            ]
        )  # fmt: skip
    runs = study.load_runs(tmp_path)
    figure = plot.metric_figure(runs, "gap_elbo", "Test ELBO gap (nats)", "log")
    assert len(tuple(figure.data)) == 2 * len(study.ARMS) * len(study.SIGMAS)
    layout: Any = figure.layout
    assert layout.legend.orientation == "h"
