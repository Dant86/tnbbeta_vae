"""Density figure: scale_free graphs at three mean degrees, paired TNBBeta effects, pooled scatter.

Usage (pre-registration: ``writeup/results/scale_free_density_figure_preregistration_*.md``):
    uv run python -m apps.eval.scale_free_density_figure configs
    uv run python -m apps.synthetic.graph_search manifest --stage density --draws 32 \
        --conditions tnbbeta vmf_init power_spherical_init
    # train via SLURM (scripts/slurm/graph_search_cpu.sbatch with STAGE=density), then
    uv run python -m apps.eval.scale_free_density_figure analyze --workers 8
    uv run python -m apps.eval.scale_free_density_figure plot --fig-dir writeup/results/figures

``configs`` writes ``graph_search/configs_density.json`` (three ``scale_free`` configs that
differ only in mean degree). ``analyze`` reads the ``density`` stage runs and the cached
``fresh`` stage and writes ``density_results.json`` (and a markdown report) under
``$TNBBETA_CHECKPOINT_DIR/graph_search/metric_first``. ``plot`` renders the figure from that
file (login node: kaleido does not work on compute nodes).
"""
# ruff: noqa: E501  (long caption and report strings)

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import rankdata, spearmanr, wilcoxon

from apps.eval.graph_search_metric_first import (
    METRICS,
    PRIMARY,
    build_cache,
    family_of,
    load_covariates,
    load_table,
    output_dir,
)
from apps.eval.graph_search_stats import benjamini_hochberg
from apps.synthetic.graph_search import SEED_BASE
from tnbbeta_vae.data.graph_covariates import draw_covariates
from tnbbeta_vae.data.graph_space import (
    FAMILIES,
    GraphSpec,
    expected_oracle_auc,
    sample_graph,
    true_probability,
)
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

Array = npt.NDArray[np.float64]

LEVELS = ("low", "med", "high")
LEVEL_NAMES = {"low": "Low", "med": "Medium", "high": "High"}
RUN_DEGREES = {"low": 6.0, "med": 15.0, "high": 35.0}
DRAWN_DEGREES = {"low": 2.0, "med": 8.0, "high": 20.0}
NUM_NODES = 600
CONTRAST = 13.0
DRAWN_NODES = 36
DRAWING_SEED = 7
GRAVITY = 0.6
BOOTSTRAP = 10_000
SEED = 20261103
ALPHA = 0.05
BAR_METRICS = (
    ("truth_kl", "truth KL"),
    ("ece", "ECE"),
    ("sampled_log_loss", "sampled log loss"),
    ("sampled_auc", "sampled AUROC"),
    ("centre_auc", "centre AUROC"),
)
PROBABILITY_METRICS = ("truth_kl", "ece", "sampled_log_loss")
STAMP = "2026-11-03"
INK = "#1a1a1a"
GRID = "#e4e4e4"
OTHER_GREY = "#b8b8b8"
GROUP_COLOURS = {"probability": "#1baf7a", "ranking": "#eb6834"}
FAMILY_COLOURS = dict(
    zip(
        FAMILIES,
        (
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#8e5bd0",
            "#c9a227",
            "#d1478c",
            "#4a4a4a",
            "#17becf",
            "#b5251d",
            "#7f9c3a",
        ),
        strict=True,
    )
)


def main(argv: list[str] | None = None) -> None:
    """Dispatches ``configs``, ``analyze`` and ``plot``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("configs")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--workers", type=int, default=4)
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "configs":
        path = checkpoint_dir() / "graph_search" / "configs_density.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(density_configs(), indent=2))
        print(f"Wrote {path}")
    elif args.command == "analyze":
        report = analyze(args.workers)
        print(report)
    else:
        results = json.loads((output_dir() / "density_results.json").read_text())
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        figure = build_figure(results)
        path = args.fig_dir / f"scale_free_density_figure_{STAMP}.png"
        figure.write_image(path, scale=3)
        print(f"Wrote {path}")


# ----------------------------------------------------------------------- configs


def density_configs() -> list[dict[str, Any]]:
    """Returns the three ``scale_free`` configs, differing only in mean degree.

    The contrast (hence the Pareto tail index), node count, features and heterogeneity
    are shared; ``achieved_auc`` is the expected oracle AUC of the seed-0 draw (it is
    recorded, not calibrated to).
    """
    configs = []
    for level in LEVELS:
        spec = GraphSpec(
            family="scale_free",
            num_nodes=NUM_NODES,
            mean_degree=RUN_DEGREES[level],
            num_groups=2,
            heterogeneity="none",
            features="none",
            contrast=CONTRAST,
        )
        auc = expected_oracle_auc(true_probability(spec, np.random.default_rng(0)))
        spec = GraphSpec(**{**asdict(spec), "achieved_auc": auc, "target_auc": auc})
        configs.append({"config_id": f"scale_free-{level}", **asdict(spec)})
    return configs


# ------------------------------------------------------------------------ effects


def advantage(
    table_cell: dict[str, dict[str, Array]], comparator: str, metric: str
) -> Array:
    """Per-draw paired difference, positive when TNBBeta is better (any metric)."""
    treated = table_cell["tnbbeta"][metric]
    other = table_cell[comparator][metric]
    return treated - other if METRICS[metric].higher else other - treated


def choose_comparator(
    table_cell: dict[str, dict[str, Array]], metric: str, keep: npt.NDArray[np.bool_]
) -> str:
    """Picks the comparator with the better level-mean of ``metric`` (TNBBeta unseen)."""
    means = {c: float(np.mean(table_cell[c][metric][keep])) for c in PRIMARY}
    pick = max if METRICS[metric].higher else min
    return pick(means, key=lambda c: means[c])


def bootstrap_mean_dz(
    values: Array, rng: np.random.Generator, boots: int = BOOTSTRAP
) -> dict[str, float]:
    """Mean, dz and 95% percentile bootstrap intervals of a paired-difference vector."""
    count = values.shape[0]
    index = rng.integers(0, count, (boots, count))
    resampled = values[index]
    means = resampled.mean(1)
    spreads = resampled.std(1, ddof=1)
    dzs = np.where(spreads > 0, means / np.where(spreads > 0, spreads, 1.0), 0.0)
    spread = float(values.std(ddof=1)) if count > 1 else 0.0
    mean = float(values.mean())
    return {
        "mean": mean,
        "mean_lo": float(np.percentile(means, 2.5)),
        "mean_hi": float(np.percentile(means, 97.5)),
        "dz": mean / spread if spread > 0 else 0.0,
        "dz_lo": float(np.percentile(dzs, 2.5)),
        "dz_hi": float(np.percentile(dzs, 97.5)),
    }


def wilcoxon_p(values: Array) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (1 if every difference is 0)."""
    if not np.any(values != 0):
        return 1.0
    return float(wilcoxon(values).pvalue)  # type: ignore[union-attr]


def level_effects(
    table_cell: dict[str, dict[str, Array]], metric: str, rng: np.random.Generator
) -> dict[str, Any]:
    """The effect record of one (level, metric); see the pre-registration."""
    arrays = [table_cell["tnbbeta"][metric]] + [table_cell[c][metric] for c in PRIMARY]
    keep = np.all(np.stack([np.isfinite(a) for a in arrays]), axis=0)
    chosen = choose_comparator(table_cell, metric, keep)
    values = advantage(table_cell, chosen, metric)[keep]
    record: dict[str, Any] = {
        "metric": metric,
        "comparator": chosen,
        "draws": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "p": wilcoxon_p(values),
        **bootstrap_mean_dz(values, rng),
    }
    each = {c: advantage(table_cell, c, metric)[keep] for c in PRIMARY}
    record["per_comparator"] = {
        c: {"mean": float(v.mean()), "dz": _dz(v)} for c, v in each.items()
    }
    per_draw_min = np.minimum(each[PRIMARY[0]], each[PRIMARY[1]])
    record["per_draw_min"] = {
        "mean": float(per_draw_min.mean()),
        "dz": _dz(per_draw_min),
    }
    return record


def label_cells(records: list[dict[str, Any]]) -> None:
    """Adds BH-adjusted p-values and a better/worse/neither label in place."""
    adjusted = benjamini_hochberg([r["p"] for r in records])
    for record, q in zip(records, adjusted, strict=True):
        record["q"] = float(q)
        significant = q <= ALPHA
        record["label"] = (
            "better"
            if significant and record["mean"] > 0
            else "worse"
            if significant and record["mean"] < 0
            else "neither"
        )


def panel_summary(records: list[dict[str, Any]]) -> str:
    """The auto-generated "BH-significant" line of one level's bar panel."""
    by_metric = {r["metric"]: r["label"] for r in records}
    prob = [by_metric[m] for m in PROBABILITY_METRICS]
    rank = [by_metric[m] for m in ("sampled_auc", "centre_auc")]
    return (
        f"BH-sig.: probability metrics better {prob.count('better')}/3, worse {prob.count('worse')}/3; "
        f"AUROC better {rank.count('better')}/2, worse {rank.count('worse')}/2"
    )


def spearman_ci(
    x: Array, y: Array, rng: np.random.Generator, boots: int = 2_000
) -> dict[str, float]:
    """Spearman of ``x`` and ``y`` with a 95% bootstrap interval (descriptive)."""
    rho = float(spearmanr(x, y)[0])  # type: ignore[index]
    index = rng.integers(0, x.shape[0], (boots, x.shape[0]))
    values = np.array([_rho(x[i], y[i]) for i in index])
    values = values[np.isfinite(values)]
    return {
        "rho": rho,
        "lo": float(np.percentile(values, 2.5)),
        "hi": float(np.percentile(values, 97.5)),
    }


def within_family_spearman(a: Array, b: Array, families: npt.NDArray[Any]) -> float:
    """Spearman of ``a`` and ``b`` after centring the ranks within each family."""
    ra, rb = rankdata(a), rankdata(b)
    for family in np.unique(families):
        pick = families == family
        ra[pick] -= ra[pick].mean()
        rb[pick] -= rb[pick].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def pooled_points() -> list[dict[str, Any]]:
    """Per fresh config: realised degree, nominal degree, and the truth-KL outcome ``E_c``.

    ``E_c`` is the pre-registered H1 outcome: the mean over draws of the paired truth-KL
    difference versus the less favorable of ``vmf_init`` / ``power_spherical_init``
    (positive = TNBBeta better).
    """
    table = load_table("fresh")
    covariates = load_covariates("fresh")
    nominal = {
        c["config_id"]: c["mean_degree"]
        for c in json.loads(
            (checkpoint_dir() / "graph_search" / "configs_fresh.json").read_text()
        )
    }
    points = []
    for config in sorted(table):
        stacked = np.stack(
            [advantage(table[config], comp, "truth_kl") for comp in PRIMARY]
        )
        keep = np.all(np.isfinite(stacked), axis=0)
        points.append(
            {
                "config": config,
                "family": family_of(config),
                "degree": covariates[config]["realised_degree"],
                "nominal_degree": float(nominal[config]),
                "E": float(stacked[:, keep].mean(1).min()),
                "kl0": covariates[config]["kl0"],
            }
        )
    return points


# ----------------------------------------------------------------------- analysis


def analyze(workers: int) -> str:
    """Runs the pre-registered analysis, writes ``density_results.json``, returns markdown."""
    build_cache("density", workers)
    table = load_table("density")
    configs = json.loads(
        (checkpoint_dir() / "graph_search" / "density" / "configs.json").read_text()
    )
    by_id = {c["config_id"]: c for c in configs}
    rng = np.random.default_rng(SEED)
    cells: dict[str, list[dict[str, Any]]] = {}
    flat: list[dict[str, Any]] = []
    for level in LEVELS:
        records = [
            {"level": level, **level_effects(table[f"scale_free-{level}"], m, rng)}
            for m, _ in BAR_METRICS
        ]
        cells[level] = records
        flat.extend(records)
    label_cells(flat)
    draw_info = _draw_covariates(by_id, table, workers)
    descriptive = _descriptive(table, draw_info, cells, rng)
    points = pooled_points()
    families = np.array([p["family"] for p in points])
    log_degree = np.log([p["nominal_degree"] for p in points])
    energy = np.array([p["E"] for p in points])
    skill = energy / np.array([p["kl0"] for p in points])
    h1 = within_family_spearman(log_degree, skill, families)
    h1_nats = within_family_spearman(log_degree, energy, families)
    results = {
        "levels": cells,
        "summary": {level: panel_summary(cells[level]) for level in LEVELS},
        "level_covariates": {
            level: _level_covariates(draw_info[level]) for level in LEVELS
        },
        "descriptive": descriptive,
        "pooled": {
            "points": points,
            "h1_spearman": h1,
            "h1_spearman_nats": h1_nats,
        },
    }
    (output_dir() / "density_results.json").write_text(json.dumps(results))
    report = render_report(results, by_id)
    (output_dir() / "density_report.md").write_text(report)
    return report


def render_report(results: dict[str, Any], configs: dict[str, dict[str, Any]]) -> str:
    """Markdown tables of the analysis."""
    lines = [
        "## Level covariates (mean over draws; realised from the sampled graphs)\n"
    ]
    lines.append(
        "| level | nominal degree | realised degree | expected-degree CV | clustering | oracle AUC (calibration draw) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for level in LEVELS:
        c = results["level_covariates"][level]
        config = configs[f"scale_free-{level}"]
        lines.append(
            f"| {level} | {config['mean_degree']:g} | {c['realised_degree']:.2f} | {c['expected_degree_cv']:.3f} | {c['clustering']:.3f} | {config['achieved_auc']:.3f} |"
        )
    lines.append(
        "\n## Paired effects (positive = TNBBeta better; comparator = better level-mean of vmf_init / power_spherical_init)\n"
    )
    lines.append(
        "| level | metric | comparator | draws (dropped) | mean diff [95% CI] | dz [95% CI] | Wilcoxon p | BH q | label | dz vmf / dz PS | per-draw-min mean (dz) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for level in LEVELS:
        for r in results["levels"][level]:
            pc = r["per_comparator"]
            lines.append(
                f"| {level} | {r['metric']} | {r['comparator']} | {r['draws']} ({r['dropped']}) | {r['mean']:+.4f} [{r['mean_lo']:+.4f}, {r['mean_hi']:+.4f}] | {r['dz']:+.2f} [{r['dz_lo']:+.2f}, {r['dz_hi']:+.2f}] | {r['p']:.4f} | {r['q']:.4f} | {r['label']} | {pc['vmf_init']['dz']:+.2f} / {pc['power_spherical_init']['dz']:+.2f} | {r['per_draw_min']['mean']:+.4f} ({r['per_draw_min']['dz']:+.2f}) |"
            )
    lines.append("\n## Panel summaries (auto-generated figure text)\n")
    for level in LEVELS:
        lines.append(f"- {level}: {results['summary'][level]}")
    lines.append(
        "\n## Descriptive trends (NOT confirmatory; Spearman of per-draw advantage vs per-draw realised degree and vs level index, over all draws)\n"
    )
    lines.append(
        "| metric | rho vs degree [95% CI] | rho vs level [95% CI] | within low | within med | within high |"
    )
    lines.append("|---|---|---|---|---|---|")
    for metric, row in results["descriptive"].items():
        w = row["within"]
        lines.append(
            f"| {metric} | {row['degree']['rho']:+.2f} [{row['degree']['lo']:+.2f}, {row['degree']['hi']:+.2f}] | {row['level']['rho']:+.2f} [{row['level']['lo']:+.2f}, {row['level']['hi']:+.2f}] | {w['low']:+.2f} | {w['med']:+.2f} | {w['high']:+.2f} |"
        )
    pooled = results["pooled"]
    lines.append(
        f"\n## Pooled panel check\n\nWithin-family Spearman of S_c = E_c / KL0 (the registered H1 outcome) vs log nominal degree over {len(pooled['points'])} fresh configs: {pooled['h1_spearman']:+.3f} (registered: +0.462); with E_c in nats instead: {pooled['h1_spearman_nats']:+.3f}."
    )
    return "\n".join(lines)


def _draw_covariates(
    configs: dict[str, dict[str, Any]], table: Any, workers: int
) -> dict[str, list[dict[str, float]]]:
    """Regenerates each draw's graph and returns its realised statistics per level."""
    draws = max(len(v) for v in table["scale_free-low"]["tnbbeta"].values())
    jobs = [
        (level, d, configs[f"scale_free-{level}"], index)
        for index, level in enumerate(LEVELS)
        for d in range(draws)
    ]

    def one(job: tuple[str, int, dict[str, Any], int]) -> dict[str, float]:
        _, draw, config, index = job
        spec = GraphSpec(**{k: v for k, v in config.items() if k != "config_id"})
        seed = SEED_BASE["density"] + 1000 * index + draw
        return draw_covariates(spec, seed)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    out: dict[str, list[dict[str, float]]] = {level: [] for level in LEVELS}
    for job, result in zip(jobs, results, strict=True):
        out[job[0]].append(result)
    return out


def _level_covariates(draws: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([d[key] for d in draws]))
        for key in ("realised_degree", "expected_degree_cv", "clustering")
    }


def _descriptive(
    table: Any,
    draw_info: dict[str, list[dict[str, float]]],
    cells: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """Per metric: Spearman of the per-draw advantage vs realised degree and level."""
    out: dict[str, dict[str, Any]] = {}
    for position, (metric, _) in enumerate(BAR_METRICS):
        xs, ls, ys, within = [], [], [], {}
        for index, level in enumerate(LEVELS):
            record = cells[level][position]
            values = advantage(
                table[f"scale_free-{level}"], record["comparator"], metric
            )
            degrees = np.array([d["realised_degree"] for d in draw_info[level]])
            keep = np.isfinite(values)
            xs.append(degrees[keep])
            ys.append(values[keep])
            ls.append(np.full(int(keep.sum()), float(index)))
            within[level] = _rho(degrees[keep], values[keep])
        x, y, lv = np.concatenate(xs), np.concatenate(ys), np.concatenate(ls)
        out[metric] = {
            "degree": spearman_ci(x, y, rng),
            "level": spearman_ci(lv, y, rng),
            "within": within,
        }
    return out


# ---------------------------------------------------------------------- drawing


def fruchterman_reingold(adjacency: Array, seed: int, iterations: int = 300) -> Array:
    """A seeded spring layout (Fruchterman-Reingold) rescaled to ``[-1, 1]^2``.

    Args:
        adjacency: Symmetric 0/1 matrix.
        seed: Seed of the initial positions.
        iterations: Number of cooling steps.

    Returns:
        Positions, shape ``(n, 2)``.
    """
    count = adjacency.shape[0]
    rng = np.random.default_rng(seed)
    position = rng.uniform(-1.0, 1.0, (count, 2))
    ideal = np.sqrt(4.0 / count)
    temperature = 0.2
    for step in range(iterations):
        delta = position[:, None, :] - position[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        repulsion = (ideal**2 / distance**2)[:, :, None] * delta
        attraction = -(adjacency * distance / ideal)[:, :, None] * delta
        force = repulsion.sum(1) + attraction.sum(1) - GRAVITY * position
        length = np.maximum(np.linalg.norm(force, axis=1, keepdims=True), 1e-9)
        position = position + force / length * np.minimum(length, temperature)
        position = np.clip(position, -1.0, 1.0)
        temperature = 0.2 * (1.0 - (step + 1) / iterations)
    position = position - position.mean(0)
    return position / np.abs(position).max()


def drawn_graph(level: str) -> dict[str, Any]:
    """The small graph drawn above one level: layout, edges and degrees."""
    spec = GraphSpec(
        family="scale_free",
        num_nodes=DRAWN_NODES,
        mean_degree=DRAWN_DEGREES[level],
        num_groups=2,
        heterogeneity="none",
        features="none",
        contrast=CONTRAST,
    )
    adjacency = np.asarray(sample_graph(spec, DRAWING_SEED).adjacency.todense())
    position = fruchterman_reingold(adjacency, DRAWING_SEED)
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    return {
        "position": position,
        "edges": list(zip(rows.tolist(), cols.tolist(), strict=True)),
        "degree": adjacency.sum(1),
    }


# ----------------------------------------------------------------------- figure


def build_figure(results: dict[str, Any]) -> go.Figure:
    """Builds the density figure from the analysis results (layout fixed in the pre-registration)."""
    figure = make_subplots(
        rows=2,
        cols=4,
        column_widths=[0.2, 0.2, 0.2, 0.4],
        row_heights=[0.4, 0.6],
        specs=[
            [{}, {}, {}, {"rowspan": 2}],
            [{}, {}, {}, None],
        ],
        horizontal_spacing=0.045,
        vertical_spacing=0.1,
    )
    bounds = _bar_bounds(results)
    for column, level in enumerate(LEVELS, start=1):
        _add_drawing(figure, level, column, results)
        _add_bars(figure, level, column, results, bounds, legend=False)
    _add_pooled(figure, results["pooled"])
    figure.update_layout(
        template=TEMPLATE_NAME,
        font={"family": FONT_FAMILY, "color": INK, "size": 13},
        width=1750,
        height=800,
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.14,
            "x": 0.8,
            "xanchor": "center",
        },
        margin={"l": 70, "r": 30, "t": 110, "b": 120},
        barmode="overlay",
    )
    return figure


def _bar_bounds(results: dict[str, Any]) -> tuple[float, float]:
    lows = [r["dz_lo"] for lv in LEVELS for r in results["levels"][lv]]
    highs = [r["dz_hi"] for lv in LEVELS for r in results["levels"][lv]]
    pad = 0.15 * (max(highs) - min(lows) + 1e-9)
    return min(0.0, min(lows)) - pad, max(0.0, max(highs)) + pad


def _add_drawing(
    figure: go.Figure, level: str, column: int, results: dict[str, Any]
) -> None:
    drawing = drawn_graph(level)
    position = drawing["position"]
    xs: list[float | None] = []
    ys: list[float | None] = []
    for i, j in drawing["edges"]:
        xs += [float(position[i, 0]), float(position[j, 0]), None]
        ys += [float(position[i, 1]), float(position[j, 1]), None]
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "#b0b0b0", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=column,
    )
    degree = drawing["degree"]
    figure.add_trace(
        go.Scatter(
            x=position[:, 0],
            y=position[:, 1],
            mode="markers",
            marker={
                "size": 6 + 4 * np.sqrt(degree),
                "color": "#2a78d6",
                "line": {"color": "white", "width": 1},
                "opacity": 0.9,
            },
            text=[f"degree {int(d)}" for d in degree],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=column,
    )
    covariates = results["level_covariates"][level]
    figure.update_xaxes(visible=False, range=[-1.15, 1.15], row=1, col=column)
    figure.update_yaxes(
        visible=False,
        range=[-1.15, 1.15],
        scaleanchor=f"x{column if column > 1 else ''}",
        row=1,
        col=column,
    )
    figure.add_annotation(
        text=(
            f"{LEVEL_NAMES[level]} Density<br>Drawn Mean Degree {degree.mean():.1f}"
            f"<br>(Runs: {RUN_DEGREES[level]:g} Nominal, {covariates['realised_degree']:.1f} Realized)"
        ),
        xref=f"x{column if column > 1 else ''} domain",
        yref=f"y{column if column > 1 else ''} domain",
        x=0.5,
        y=1.02,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font={"size": 13},
    )


def _add_bars(
    figure: go.Figure,
    level: str,
    column: int,
    results: dict[str, Any],
    bounds: tuple[float, float],
    *,
    legend: bool,
) -> None:
    records = results["levels"][level]
    for group, metrics in (
        ("probability", ("truth_kl",)),
        ("ranking", ("sampled_auc",)),
    ):
        chosen = [r for r in records if r["metric"] in metrics]
        names = {"truth_kl": "Truth KL", "sampled_auc": "Sampled AUROC"}
        figure.add_trace(
            go.Bar(
                x=[names[r["metric"]] for r in chosen],
                y=[r["dz"] for r in chosen],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [r["dz_hi"] - r["dz"] for r in chosen],
                    "arrayminus": [r["dz"] - r["dz_lo"] for r in chosen],
                    "color": INK,
                    "thickness": 1.2,
                },
                marker={"color": GROUP_COLOURS[group]},
                text=["*" if r["label"] != "neither" else "" for r in chosen],
                textposition="outside",
                textfont={"size": 18},
                customdata=[
                    [r["mean"], r["mean_lo"], r["mean_hi"], r["q"], r["comparator"]]
                    for r in chosen
                ],
                hovertemplate=(
                    "dz %{y:.2f}<br>raw diff %{customdata[0]:+.4f} "
                    "[%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]<br>BH q %{customdata[3]:.3f}"
                    "<br>vs %{customdata[4]}<extra></extra>"
                ),
                name=(
                    "probability-quality metrics (TNBBeta tends to win)"
                    if group == "probability"
                    else "ranking metrics, AUROC (TNBBeta tends to lose)"
                ),
                legendgroup=group,
                showlegend=legend,
            ),
            row=2,
            col=column,
        )
    figure.update_yaxes(
        range=list(bounds),
        zeroline=True,
        zerolinecolor=INK,
        gridcolor=GRID,
        title_text="Paired Effect dz (Up = TNBBeta Better)" if column == 1 else None,
        row=2,
        col=column,
    )
    figure.update_xaxes(
        tickangle=-40, linecolor=INK, tickfont={"size": 11}, row=2, col=column
    )


def _add_pooled(figure: go.Figure, pooled: dict[str, Any]) -> None:
    points = pooled["points"]
    others = [p for p in points if p["family"] != "scale_free"]
    drawn = [p for p in points if p["family"] == "scale_free"]
    for chosen, name, marker in (
        (
            others,
            "Other Families",
            {
                "size": 7,
                "color": OTHER_GREY,
                "opacity": 0.7,
                "line": {"color": "white", "width": 0.5},
            },
        ),
        (
            drawn,
            "Scale Free (Drawn Family)",
            {
                "size": 10,
                "color": FAMILY_COLOURS["scale_free"],
                "opacity": 0.85,
                "line": {"color": INK, "width": 1.6},
            },
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=[p["degree"] for p in chosen],
                y=[p["E"] for p in chosen],
                mode="markers",
                marker=marker,
                text=[p["config"] for p in chosen],
                hovertemplate="%{text}<br>degree %{x:.1f}<br>E_c %{y:+.4f}<extra></extra>",
                name=name,
            ),
            row=1,
            col=4,
        )
    figure.add_hline(
        y=0.0,
        line={"color": INK, "width": 1},
        row=1,  # pyright: ignore[reportArgumentType]
        col=4,  # pyright: ignore[reportArgumentType]
    )
    figure.update_xaxes(
        type="log",
        title_text="Realized Mean Degree (Log Scale)",
        tickvals=[4, 6, 10, 15, 25, 40],
        ticktext=["4", "6", "10", "15", "25", "40"],
        gridcolor=GRID,
        linecolor=INK,
        row=1,
        col=4,
    )
    figure.update_yaxes(
        title_text="Truth-KL Advantage E_c (Nats; Up = TNBBeta Better)",
        gridcolor=GRID,
        linecolor=INK,
        row=1,
        col=4,
    )
    figure.add_annotation(
        text=(
            f"Pooled Evidence: All {len(points)} Fresh Configs<br>"
            f"within-family Spearman {pooled['h1_spearman']:+.3f} (pre-registered H1, one-sided p = 0.0001)<br>"
            "per config vs the less favorable of tuned vMF / PowerSpherical;<br>"
            "Spearman is on S_c = E_c / KL0 (registered outcome), y shows E_c"
        ),
        xref="x4 domain",
        yref="y4 domain",
        x=0.0,
        y=1.03,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
        font={"size": 12},
    )


# ---------------------------------------------------------------------- helpers


def _dz(values: Array) -> float:
    spread = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
    return float(values.mean()) / spread if spread > 0 else 0.0


def _rho(x: Array, y: Array) -> float:
    if x.shape[0] < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y)[0])  # type: ignore[index]


if __name__ == "__main__":
    main()
