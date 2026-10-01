"""Tests for apps.eval.posterior_trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from apps.eval import posterior_trajectories as pt


def _trace_count(figure: go.Figure) -> int:
    """Returns the number of traces on ``figure`` (plotly's stubs don't type ``.data``
    as ``Sized``)."""
    return len(figure.data)  # pyright: ignore[reportArgumentType]


def _write_metrics(run_dir: Path, records: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _tnbbeta_record(step: int, p: float, q: float, epsilon: float) -> dict[str, object]:
    return {
        "step": step,
        "posterior_p_mean": p,
        "posterior_q_mean": q,
        "posterior_epsilon_mean": epsilon,
    }


def test_discover_seeds_finds_only_matching_prefix(tmp_path: Path) -> None:
    (tmp_path / "mnist_tnbs_d5_seed0").mkdir()
    (tmp_path / "mnist_tnbs_d5_seed2").mkdir()
    (tmp_path / "mnist_tnbs_d5_seed10").mkdir()
    (tmp_path / "mnist_tnbs_d10_seed0").mkdir()  # different dim, must not match
    (tmp_path / "mnist_tnbs_d5_seedx").mkdir()  # not a seed, must not match

    assert pt.discover_seeds(tmp_path, "mnist_tnbs_d5") == [0, 2, 10]


def test_discover_seeds_empty_when_nothing_matches(tmp_path: Path) -> None:
    assert pt.discover_seeds(tmp_path, "mnist_tnbs_d5") == []


def test_read_metric_series_filters_and_extracts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metrics(
        run_dir,
        [
            _tnbbeta_record(0, 0.5, 0.1, 1.0),
            {"step": 1, "epoch": 0, "val_loss": 100.0},  # no posterior fields
            _tnbbeta_record(2, 0.6, 0.2, 1.1),
        ],
    )

    steps, values = pt.read_metric_series(run_dir / "metrics.jsonl", "posterior_p_mean")

    assert steps.tolist() == [0, 2]
    assert values.tolist() == [0.5, 0.6]


def test_read_metric_series_missing_file_returns_empty(tmp_path: Path) -> None:
    steps, values = pt.read_metric_series(
        tmp_path / "missing" / "metrics.jsonl", "posterior_p_mean"
    )
    assert steps.size == 0
    assert values.size == 0


def test_read_metric_series_missing_field_returns_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metrics(run_dir, [{"step": 0, "val_loss": 1.0}])

    steps, values = pt.read_metric_series(run_dir / "metrics.jsonl", "posterior_p_mean")
    assert steps.size == 0
    assert values.size == 0


def test_read_metric_series_decimates_to_max_points(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metrics(
        run_dir, [_tnbbeta_record(step, 0.5, 0.1, 1.0) for step in range(1000)]
    )

    steps, values = pt.read_metric_series(
        run_dir / "metrics.jsonl", "posterior_p_mean", max_points=50
    )

    assert len(steps) == 50
    assert len(values) == 50
    # Decimation keeps the first and last points so the full span is still visible.
    assert steps[0] == 0
    assert steps[-1] == 999


def _seed_runs(
    runs_dir: Path, prefix: str, seeds: list[int], field_values: dict[str, float]
) -> None:
    for seed in seeds:
        record = {"step": 0, **{name: value for name, value in field_values.items()}}
        _write_metrics(runs_dir / f"{prefix}_seed{seed}", [record])


def test_build_tnbbeta_figure_has_one_trace_per_seed_per_cell(tmp_path: Path) -> None:
    dims = [2, 5]
    for dim in dims:
        _seed_runs(
            tmp_path,
            f"mnist_tnbs_d{dim}",
            [0, 1, 2],
            {
                "posterior_p_mean": 0.9,
                "posterior_q_mean": 0.1,
                "posterior_epsilon_mean": 1.0,
            },
        )

    figure = pt.build_tnbbeta_figure(tmp_path, max_points=500, dims=dims)

    # 3 rows (p, q, epsilon) x 2 dims x 3 seeds.
    assert _trace_count(figure) == 3 * len(dims) * 3


def test_build_kappa_figure_has_one_trace_per_model_per_seed_per_cell(
    tmp_path: Path,
) -> None:
    dims = [2, 5]
    for dim in dims:
        for model in ("vmfs", "pss", "vmfks"):
            _seed_runs(
                tmp_path, f"mnist_{model}_d{dim}", [0, 1], {"posterior_kappa_mean": 5.0}
            )

    figure = pt.build_kappa_figure(tmp_path, max_points=500, dims=dims)

    # 2 dims x 3 models x 2 seeds.
    assert _trace_count(figure) == len(dims) * 3 * 2
    legend_entries = [
        trace.name  # pyright: ignore[reportAttributeAccessIssue]
        for trace in figure.data
        if trace.showlegend  # pyright: ignore[reportAttributeAccessIssue]
    ]
    assert legend_entries == ["vmfs", "pss", "vmfks"]


def test_build_ablation_figures_trace_counts(tmp_path: Path) -> None:
    for prefix in ("mnistfix_eps0p5_d5", "mnistfix_eps1p0_d5", "mnistfix_eps1p5_d5"):
        _seed_runs(tmp_path, prefix, [0, 1, 2], {"posterior_q_mean": 0.5})
    _seed_runs(
        tmp_path,
        "mnistfixmu_tnb_d5",
        [0, 1, 2],
        {"posterior_p_mean": 0.7, "posterior_q_mean": 0.3},
    )

    eps_figure, mu_figure = pt.build_ablation_figures(tmp_path, max_points=500)

    # 3 epsilon groups x 3 seeds, x2 for each line's endpoint marker.
    assert _trace_count(eps_figure) == 3 * 3 * 2
    # p and q, 3 seeds each, x2 for each line's endpoint marker.
    assert _trace_count(mu_figure) == 2 * 3 * 2


def test_build_figures_skip_missing_runs_without_error(tmp_path: Path) -> None:
    # No runs at all under tmp_path -- every figure should still build, just empty.
    tnbbeta_figure = pt.build_tnbbeta_figure(tmp_path, max_points=500, dims=[2])
    kappa_figure = pt.build_kappa_figure(tmp_path, max_points=500, dims=[2])
    eps_figure, mu_figure = pt.build_ablation_figures(tmp_path, max_points=500)

    assert _trace_count(tnbbeta_figure) == 0
    assert _trace_count(kappa_figure) == 0
    assert _trace_count(eps_figure) == 0
    assert _trace_count(mu_figure) == 0


def test_main_writes_all_four_pngs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    for dim in [2]:
        _seed_runs(
            runs_dir,
            f"mnist_tnbs_d{dim}",
            [0],
            {
                "posterior_p_mean": 0.9,
                "posterior_q_mean": 0.1,
                "posterior_epsilon_mean": 1.0,
            },
        )
        for model in ("vmfs", "pss", "vmfks"):
            _seed_runs(
                runs_dir, f"mnist_{model}_d{dim}", [0], {"posterior_kappa_mean": 5.0}
            )
    for prefix in ("mnistfix_eps0p5_d5", "mnistfix_eps1p0_d5", "mnistfix_eps1p5_d5"):
        _seed_runs(runs_dir, prefix, [0], {"posterior_q_mean": 0.5})
    _seed_runs(
        runs_dir,
        "mnistfixmu_tnb_d5",
        [0],
        {"posterior_p_mean": 0.7, "posterior_q_mean": 0.3},
    )

    output_dir = tmp_path / "output"
    pt.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(output_dir),
            "--dims",
            "2",
        ]
    )

    for name in [
        "tnbbeta_posterior_trajectories.png",
        "kappa_trajectories.png",
        "fixed_epsilon_ablation_trajectories.png",
        "fixed_mu_ablation_trajectories.png",
    ]:
        path = output_dir / name
        assert path.exists() and path.stat().st_size > 0
