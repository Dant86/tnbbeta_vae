"""Does a tuned fixed kappa, or a recalibration, remove TNBBeta's graph-search wins?

Analysis of the ``kappa_grid`` stage, pre-registered in
``writeup/results/graph_search_kappa_grid_preregistration_2026-10-31.md``:

    uv run python -m apps.eval.graph_search_kappa report --output report.md

Two confirmatory families, both intersection-union tests over vMF and power-spherical
with the margins of :mod:`apps.eval.graph_search_stats`, Holm-corrected within the
family:

* ``grid``: for each confirmed (config, metric) pair, TNBBeta against the fixed-kappa
  model whose kappa was best on the *other* half of the draws (cross-fitted), per
  family.
* ``recal``: TNBBeta against the trained comparators after every model's mean
  prediction is passed through a Platt map fitted on its validation pairs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr

from apps.eval.graph_search_stats import (
    COMPARATOR_SETS,
    MARGINS,
    holm,
    load_runs,
    one_sided_p,
    rejected_pairs,
    run_config_tests,
)
from apps.synthetic.graph_search import KAPPA_GRID, stage_dir

ALPHA = 0.05
FAMILIES = {"vmf": "vmf", "power_spherical": "power_spherical"}
RECAL_METRICS = {
    "recal_log_loss": (False, MARGINS["sampled_log_loss"][1]),
    "recal_truth_kl": (False, MARGINS["truth_kl"][1]),
}


def main(argv: list[str] | None = None) -> None:
    """Runs the ``report`` subcommand.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="Print the full kappa-grid report.")
    report.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    text = render_report(ALPHA)
    if args.output:
        args.output.write_text(text)
    print(text)


def confirmed_pairs(alpha: float = ALPHA) -> list[tuple[str, str]]:
    """Returns the (config, metric) pairs confirmed on the confirmation stage (Holm)."""
    discovery = json.loads((stage_dir("discovery") / "test_rows.json").read_text())
    wanted = set(rejected_pairs(discovery["primary"], alpha))
    rows = [
        r
        for r in run_config_tests(load_runs("confirmation"), COMPARATOR_SETS["primary"])
        if (r["config_id"], r["metric"]) in wanted
    ]
    adjusted = holm([r["p"] for r in rows])
    return sorted(
        (r["config_id"], r["metric"])
        for r, p in zip(rows, adjusted, strict=True)
        if p <= alpha
    )


def value_table(runs: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict]:
    """Indexes each run's metrics by (config, draw, condition)."""
    return {(r["config_id"], r["draw"], r["condition"]): r["metrics"] for r in runs}


def orientation(metric: str) -> tuple[bool, float]:
    """Returns (higher is better, margin) for a standard or recalibrated metric."""
    return {**MARGINS, **RECAL_METRICS}[metric]


def oriented(table: dict, key: tuple[str, int, str], metric: str) -> float:
    """Returns the metric oriented so that larger is better (NaN if missing)."""
    metrics = table.get(key)
    if metrics is None:
        return math.nan
    value = metrics[metric]
    return value if orientation(metric)[0] else -value


def best_kappa(
    table: dict, config_id: str, family: str, metric: str, draws: list[int]
) -> str:
    """Returns the fixed-kappa condition of ``family`` best on ``draws``."""
    scores = {}
    for kappa in KAPPA_GRID:
        name = f"{family}_k{kappa:g}"
        values = [oriented(table, (config_id, d, name), metric) for d in draws]
        values = [v for v in values if math.isfinite(v)]
        if values:
            scores[name] = float(np.mean(values))
    return max(scores, key=lambda name: scores[name])


def crossfit_differences(
    table: dict, config_id: str, family: str, metric: str, draws: list[int]
) -> tuple[list[float], list[str]]:
    """TNBBeta minus the cross-fitted best-kappa model of ``family``, per draw.

    The kappa used on one half of the draws is picked on the other half, so the
    comparator's choice never sees the draws it is scored on.
    """
    differences: list[float] = []
    chosen: list[str] = []
    halves = (draws[0::2], draws[1::2])
    for tested, selected in (halves, halves[::-1]):
        name = best_kappa(table, config_id, family, metric, selected)
        chosen.append(name)
        for d in tested:
            mine = oriented(table, (config_id, d, "tnbbeta"), metric)
            theirs = oriented(table, (config_id, d, name), metric)
            if math.isfinite(mine) and math.isfinite(theirs):
                differences.append(mine - theirs)
    return differences, chosen


def grid_tests(
    table: dict, pairs: list[tuple[str, str]], draws: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """Cross-fitted best-kappa intersection-union test for every confirmed pair."""
    rows = []
    for config_id, metric in pairs:
        margin = orientation(metric)[1]
        by_family = {
            f: crossfit_differences(table, config_id, f, metric, draws[config_id])
            for f in FAMILIES
        }
        p_values = {f: one_sided_p(d, margin) for f, (d, _) in by_family.items()}
        rows.append(
            {
                "config_id": config_id,
                "metric": metric,
                "p": max(p_values.values()),
                "mean_diff": {
                    f: float(np.mean(d)) if d else math.nan
                    for f, (d, _) in by_family.items()
                },
                "chosen": {f: sorted(set(c)) for f, (_, c) in by_family.items()},
            }
        )
    return rows


def recal_tests(
    table: dict, configs: list[str], draws: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """TNBBeta against the trained comparators after Platt recalibration."""
    rows = []
    for config_id in configs:
        for metric, (_, margin) in RECAL_METRICS.items():
            samples = {}
            for comparator in COMPARATOR_SETS["primary"]:
                values = []
                for d in draws[config_id]:
                    mine = oriented(table, (config_id, d, "tnbbeta"), metric)
                    theirs = oriented(table, (config_id, d, comparator), metric)
                    if math.isfinite(mine) and math.isfinite(theirs):
                        values.append(mine - theirs)
                samples[comparator] = values
            rows.append(
                {
                    "config_id": config_id,
                    "metric": metric,
                    "p": max(one_sided_p(v, margin) for v in samples.values()),
                    "mean_diff": {c: float(np.mean(v)) for c, v in samples.items()},
                }
            )
    return rows


def kappa_vs_posterior(
    runs: list[dict[str, Any]], configs: list[str]
) -> dict[str, Any]:
    """Compares each config's best fixed-kappa r_bar with TNBBeta's own."""
    table = value_table(runs)
    draws = _draws(runs)
    by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (config_id, _, condition), metrics in table.items():
        by_condition[(config_id, condition)].append(metrics["post_r_bar"])
    tnb, best = [], []
    for config_id in configs:
        mine = by_condition.get((config_id, "tnbbeta"))
        if not mine:
            continue
        name = best_kappa(table, config_id, "vmf", "sampled_log_loss", draws[config_id])
        tnb.append(float(np.mean(mine)))
        best.append(float(np.mean(by_condition[(config_id, name)])))
    rho = cast_rho(tnb, best)
    return {"configs": len(tnb), "spearman": rho, "tnb": tnb, "vmf_best": best}


def cast_rho(x: list[float], y: list[float]) -> float:
    """Returns the Spearman correlation, NaN when undefined."""
    if len(x) < 3:
        return math.nan
    return float(cast("Any", spearmanr(x, y))[0])


def render_report(alpha: float) -> str:
    """Builds the markdown report of the confirmatory families and diagnostics."""
    runs = load_runs("kappa_grid")
    table = value_table(runs)
    draws = _draws(runs)
    pairs = [p for p in confirmed_pairs(alpha) if p[0] in draws]
    configs = sorted({c for c, _ in pairs})
    grid = grid_tests(table, pairs, draws)
    for row, value in zip(grid, holm([r["p"] for r in grid]), strict=True):
        row["p_holm"] = value
    recal = recal_tests(table, configs, draws)
    for row, value in zip(recal, holm([r["p"] for r in recal]), strict=True):
        row["p_holm"] = value
    out = [
        f"## kappa-grid stage: {len(runs)} runs, {len(configs)} configs, "
        f"{len(pairs)} confirmed pairs",
        "",
        f"### Family grid: TNBBeta vs cross-fitted best kappa (Holm, alpha={alpha})",
        f"- still better: {sum(r['p_holm'] <= alpha for r in grid)} of {len(grid)}",
    ]
    out += _by_metric(grid, alpha)
    out += [
        "",
        "| config | metric | p (Holm) | mean diff vMF-k / PS-k | chosen kappa |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(grid, key=lambda r: r["p_holm"]):
        d = r["mean_diff"]
        out.append(
            f"| {r['config_id']} | {r['metric']} | {r['p_holm']:.2e} | "
            f"{d['vmf']:+.4f} / {d['power_spherical']:+.4f} | "
            f"{','.join(map(_k, r['chosen']['vmf']))}; "
            f"{','.join(map(_k, r['chosen']['power_spherical']))} |"
        )
    out += [
        "",
        f"### Family recal: TNBBeta vs recalibrated vMF/PS (Holm over {len(recal)})",
        f"- still better: {sum(r['p_holm'] <= alpha for r in recal)} of {len(recal)}",
    ]
    out += _by_metric(recal, alpha)
    reverse = sum(all(v < 0 for v in r["mean_diff"].values()) for r in recal)
    out += [f"- configs-metrics where TNBBeta is worse on average than both: {reverse}"]
    link = kappa_vs_posterior(runs, configs)
    out += [
        "",
        "### Concentration check (descriptive)",
        f"- configs: {link['configs']}; Spearman of TNBBeta r_bar vs the r_bar of the "
        f"best vMF kappa (by log loss): {link['spearman']:+.2f}",
    ]
    return "\n".join(out)


def _by_metric(rows: list[dict[str, Any]], alpha: float) -> list[str]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["metric"]][1] += 1
        counts[row["metric"]][0] += row["p_holm"] <= alpha
    return [f"  - {m}: {a} of {b}" for m, (a, b) in sorted(counts.items())]


def _k(name: str) -> str:
    return name.rsplit("_k", 1)[1]


def _draws(runs: list[dict[str, Any]]) -> dict[str, list[int]]:
    found: dict[str, set[int]] = defaultdict(set)
    for r in runs:
        found[r["config_id"]].add(r["draw"])
    return {c: sorted(d) for c, d in found.items()}


if __name__ == "__main__":
    main()
