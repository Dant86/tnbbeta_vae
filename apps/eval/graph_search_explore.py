"""Exploratory characterisation of where TNBBeta wins in the wide graph search.

Usage:
    uv run python -m apps.eval.graph_search_explore aggregate --stage discovery
    uv run python -m apps.eval.graph_search_explore report --stage discovery

Everything here is *post hoc*: it reads finished runs and describes them, with no
pre-registered hypothesis, so its associations are candidates to be tested on a fresh
set of configs, not findings. ``aggregate`` reduces a stage's runs to one record per
config (means of every metric per condition, and the paired standardized effect of
TNBBeta against each comparator) and writes ``<stage dir>/config_aggregates.json``;
``report`` prints markdown tables from it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from apps.eval.graph_search_stats import MARGINS, benjamini_hochberg, load_runs
from apps.synthetic.graph_search import stage_dir
from tnbbeta_vae.data.graph_space import FAMILIES

COMPARATORS = (
    "vmf_init",
    "power_spherical_init",
    "gaussian",
    "vmf_pin",
    "power_spherical_pin",
)
PRIMARY = ("vmf_init", "power_spherical_init")
# Oriented effects are computed for these (the pre-registered six plus sampled AUC).
EFFECT_METRICS = {**MARGINS, "sampled_auc": (True, 0.0)}
ATTRIBUTES = (
    "log_num_nodes",
    "log_mean_degree",
    "log_num_groups",
    "heterogeneity",
    "informative_features",
    "achieved_auc",
)
# A paired standardized effect beyond +/- CLEAR counts as a clear win or loss.
CLEAR = 0.5
HEADLINE = ("centre_auc", "sampled_log_loss", "cal_slope_dev", "truth_kl")


def main(argv: list[str] | None = None) -> None:
    """Runs the ``aggregate`` or ``report`` subcommand.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["aggregate", "report"])
    parser.add_argument("--stage", default="discovery")
    args = parser.parse_args(argv)
    path = stage_dir(args.stage) / "config_aggregates.json"
    if args.command == "aggregate":
        configs = json.loads((stage_dir(args.stage) / "configs.json").read_text())
        aggregates = aggregate(load_runs(args.stage), configs)
        path.write_text(json.dumps(aggregates))
        print(f"Wrote {len(aggregates)} config aggregates to {path}")
    else:
        print(report(json.loads(path.read_text())))


def aggregate(
    runs: list[dict[str, Any]], configs: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Reduces runs to one record per config.

    Args:
        runs: Finished runs of a stage.
        configs: The stage's configs (their attributes are copied into the record).

    Returns:
        ``{config_id: {"spec", "mean", "effect", "diverged"}}``: ``mean`` maps each
        condition to its per-metric mean over draws, ``effect`` maps each comparator
        and metric to the oriented paired difference ``tnbbeta - comparator`` (positive
        means TNBBeta is better) as ``mean``, ``sd``, ``dz`` (mean over SD) and ``n``.
    """
    by_draw: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    diverged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for run in runs:
        by_draw[run["config_id"]][run["draw"]][run["condition"]] = run["metrics"]
        diverged[run["config_id"]][run["condition"]] += int(run.get("diverged", 0) > 0)
    records: dict[str, dict[str, Any]] = {}
    for config in configs:
        draws = by_draw.get(config["config_id"])
        if not draws:
            continue
        conditions = sorted({c for d in draws.values() for c in d})
        mean = {
            c: {
                m: float(np.nanmean([d[c][m] for d in draws.values() if c in d]))
                for m in next(iter(draws.values()))[c]
            }
            for c in conditions
        }
        effect = {
            comparator: {
                metric: _paired_effect(draws, comparator, metric, higher)
                for metric, (higher, _) in EFFECT_METRICS.items()
            }
            for comparator in COMPARATORS
            if comparator in conditions
        }
        records[config["config_id"]] = {
            "spec": config,
            "mean": mean,
            "effect": effect,
            "diverged": dict(diverged[config["config_id"]]),
        }
    return records


def report(aggregates: dict[str, dict[str, Any]]) -> str:
    """Returns the markdown tables of the exploratory analysis."""
    sections = [
        _metric_table(aggregates),
        _profile_table(aggregates),
        _metric_correlations(aggregates),
        _calibration_table(aggregates),
        _attribute_table(aggregates),
        _regime_table(aggregates),
        _regime_detail(aggregates),
    ]
    return "\n\n".join(sections)


def advantage(record: dict[str, Any], metric: str) -> float:
    """Returns TNBBeta's standardized advantage on ``metric``.

    The *smaller* of its paired effects against the two primary comparators, since it
    must beat both to count (as in the pre-registered tests).
    """
    return min(record["effect"][c][metric]["dz"] for c in PRIMARY)


def disadvantage(record: dict[str, Any], metric: str) -> float:
    """Returns the *larger* paired effect against the primary comparators.

    Below ``-CLEAR`` TNBBeta is clearly worse than both.
    """
    return max(record["effect"][c][metric]["dz"] for c in PRIMARY)


def _paired_effect(
    draws: dict[int, dict[str, dict[str, float]]],
    comparator: str,
    metric: str,
    higher: bool,
) -> dict[str, float]:
    """Mean, SD and standardized effect of the oriented per-draw difference."""
    sign = 1.0 if higher else -1.0
    diffs = [
        sign * (d["tnbbeta"][metric] - d[comparator][metric])
        for d in draws.values()
        if "tnbbeta" in d and comparator in d
    ]
    diffs = [x for x in diffs if math.isfinite(x)]
    nan = float("nan")
    if len(diffs) < 3:
        return {"mean": nan, "sd": nan, "dz": nan, "n": len(diffs)}
    mean, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    return {"mean": mean, "sd": sd, "dz": mean / sd if sd > 0 else 0.0, "n": len(diffs)}


def _family(config_id: str) -> str:
    return config_id.rsplit("-", 1)[0]


def _metric_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Per metric: configs with a clear TNBBeta win or loss against both comparators."""
    lines = [
        "### 1. Per metric: configs where TNBBeta is clearly better / worse than both",
        "",
        f"(paired standardized effect beyond +/-{CLEAR}, both primary comparators)",
        "",
        "| metric | clear win | clear loss | median advantage (dz) |",
        "|---|---|---|---|",
    ]
    for metric in EFFECT_METRICS:
        values = [advantage(r, metric) for r in aggregates.values()]
        wins = sum(v > CLEAR for v in values)
        losses = sum(disadvantage(r, metric) < -CLEAR for r in aggregates.values())
        median = float(np.nanmedian(values))
        lines.append(f"| {metric} | {wins} | {losses} | {median:+.2f} |")
    return "\n".join(lines)


def _profile_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Win profiles over the four headline metrics."""
    counts: dict[str, int] = defaultdict(int)
    for record in aggregates.values():
        marks = []
        for metric in HEADLINE:
            if advantage(record, metric) > CLEAR:
                marks.append("+")
            elif disadvantage(record, metric) < -CLEAR:
                marks.append("-")
            else:
                marks.append("0")
        counts["".join(marks)] += 1
    lines = [
        "### 2. Win profiles over (centre AUC, log loss, calibration slope, truth KL)",
        "",
        "`+` clear win over both comparators, `-` clear loss to both, `0` neither.",
        "",
        "| profile | configs |",
        "|---|---|",
    ]
    for profile, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"| `{profile}` | {count} |")
    return "\n".join(lines)


def _metric_correlations(aggregates: dict[str, dict[str, Any]]) -> str:
    """Spearman correlation across configs of TNBBeta's advantage between metrics."""
    metrics = list(EFFECT_METRICS)
    table = np.array([[advantage(r, m) for m in metrics] for r in aggregates.values()])
    lines = [
        "### 3. Correlation across configs of TNBBeta's advantage between metrics",
        "",
        "Spearman rho of the standardized advantage over the configs.",
        "",
        "| | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for i, a in enumerate(metrics):
        cells = []
        for j in range(len(metrics)):
            ok = np.isfinite(table[:, i]) & np.isfinite(table[:, j])
            cells.append(f"{spearmanr(table[ok, i], table[ok, j])[0]:+.2f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _calibration_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Calibration direction and 'worse than the base rate' by family and condition."""
    conditions = ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
    lines = [
        "### 4. Calibration slope and worse-than-chance share, by family",
        "",
        "Each cell is `median slope / share of configs with sampled log loss > ln 2`.",
        "A slope above 1 is under-confident, below 1 over-confident; a log loss above",
        "ln 2 is worse than predicting 50% for every pair.",
        "",
        "| family | " + " | ".join(conditions) + " |",
        "|---|" + "---|" * len(conditions),
    ]
    for family in FAMILIES:
        records = [r for k, r in aggregates.items() if _family(k) == family]
        cells = []
        for condition in conditions:
            slopes = [r["mean"][condition]["cal_slope"] for r in records]
            losses = [r["mean"][condition]["sampled_log_loss"] for r in records]
            share = float(np.mean([x > math.log(2) for x in losses]))
            cells.append(f"{np.median(slopes):.2f} / {share:.2f}")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _attribute_values(spec: dict[str, Any]) -> dict[str, float]:
    heterogeneity = {"none": 0.0, "mild": 1.0, "heavy": 2.0}
    return {
        "log_num_nodes": math.log(spec["num_nodes"]),
        "log_mean_degree": math.log(spec["mean_degree"]),
        "log_num_groups": math.log(max(2, spec["num_groups"])),
        "heterogeneity": heterogeneity[spec["heterogeneity"]],
        "informative_features": float(spec["features"] == "informative"),
        "achieved_auc": spec["achieved_auc"],
    }


def _attribute_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Spearman association of TNBBeta's advantage with each config attribute.

    Pooled over all configs and *within family* (both sides demeaned by family, so graph
    family itself cannot produce the association). BH over every test in the table.
    """
    ids = list(aggregates)
    families = np.array([_family(k) for k in ids])
    attrs = {k: _attribute_values(aggregates[k]["spec"]) for k in ids}
    entries = []
    for metric in HEADLINE:
        y = np.array([advantage(aggregates[k], metric) for k in ids])
        ok = np.isfinite(y)
        for attribute in ATTRIBUTES:
            x = np.array([attrs[k][attribute] for k in ids])
            pooled = spearmanr(x[ok], y[ok])
            xw, yw = x.copy(), y.copy()
            for family in FAMILIES:
                group = (families == family) & ok
                xw[group] -= x[group].mean()
                yw[group] -= y[group].mean()
            within = spearmanr(xw[ok], yw[ok])
            entries.append(
                (metric, attribute, pooled[0], pooled[1], within[0], within[1])
            )
    q_pooled = benjamini_hochberg([e[3] for e in entries])
    q_within = benjamini_hochberg([e[5] for e in entries])
    lines = [
        "### 5. Association of TNBBeta's advantage with graph attributes",
        "",
        "Spearman rho; q is BH over all "
        f"{2 * len(entries)} tests; *within family* removes family means. Only rows",
        "with q <= 0.1 in either column are shown.",
        "",
        "| metric | attribute | rho pooled | q | rho within family | q |",
        "|---|---|---|---|---|---|",
    ]
    for e, qp, qw in zip(entries, q_pooled, q_within, strict=True):
        if min(qp, qw) <= 0.1:
            lines.append(
                f"| {e[0]} | {e[1]} | {e[2]:+.2f} | {qp:.3f} | {e[4]:+.2f} | {qw:.3f} |"
            )
    return "\n".join(lines)


def _regime_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Win counts by TNBBeta's own posterior regime (tight, middle or diffuse)."""
    bins = (
        ("tight (r_bar > 0.8)", 0.8, 2.0),
        ("middle (0.5 - 0.8)", 0.5, 0.8),
        ("diffuse (< 0.5)", -1.0, 0.5),
    )
    lines = [
        "### 6. By TNBBeta posterior regime (config mean of r_bar)",
        "",
        "Clear wins over both comparators per headline metric.",
        "",
        "| regime | configs | TNB r_bar | vMF r_bar | second-mode mass | "
        + " | ".join(f"win {m}" for m in HEADLINE)
        + " | families |",
        "|---|---|---|---|---|" + "---|" * (len(HEADLINE) + 1),
    ]
    for label, low, high in bins:
        records = [
            (k, r)
            for k, r in aggregates.items()
            if low <= r["mean"]["tnbbeta"]["post_r_bar"] < high
        ]
        if not records:
            continue
        tnb = [r["mean"]["tnbbeta"] for _, r in records]
        vmf = [r["mean"]["vmf_init"] for _, r in records]
        wins = [sum(advantage(r, m) > CLEAR for _, r in records) for m in HEADLINE]
        names = sorted({_family(k) for k, _ in records})
        shown = ", ".join(
            f"{f} ({sum(_family(k) == f for k, _ in records)})" for f in names
        )
        tnb_r = np.mean([t["post_r_bar"] for t in tnb])
        vmf_r = np.mean([v["post_r_bar"] for v in vmf])
        lines.append(
            f"| {label} | {len(records)} | {tnb_r:.2f} | {vmf_r:.2f}"
            f" | {np.nanmean([t['second_mode_mass'] for t in tnb]):.3f}"
            f" | {' | '.join(str(w) for w in wins)} | {shown} |"
        )
    return "\n".join(lines)


def regime_of(record: dict[str, Any]) -> str:
    """Returns ``tight``, ``middle`` or ``diffuse`` from TNBBeta's config-mean r_bar."""
    r_bar = record["mean"]["tnbbeta"]["post_r_bar"]
    return "tight" if r_bar > 0.8 else "diffuse" if r_bar < 0.5 else "middle"


def _regime_detail(aggregates: dict[str, dict[str, Any]]) -> str:
    """Absolute log loss, calibration slope and learnability by posterior regime.

    The regime is TNBBeta's own *outcome* of training, so it is a descriptive
    stratification (partly a proxy for how learnable the graph is), not a cause.
    """
    conditions = ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
    lines = [
        "### 7. By regime: learnability, absolute log loss and calibration slope",
        "",
        "Means over the regime's configs. `>ln2` is the share with sampled log loss",
        "above ln 2 (worse than predicting 50%).",
        "",
        "| regime | configs | oracle AUC | "
        + " | ".join(f"log loss {c}" for c in conditions)
        + " | slope tnbbeta / vmf / ps | >ln2 tnbbeta / vmf / ps |",
        "|---|---|---|" + "---|" * (len(conditions) + 2),
    ]
    for label in ("tight", "middle", "diffuse"):
        records = [r for r in aggregates.values() if regime_of(r) == label]
        if not records:
            continue

        auc = np.mean([r["spec"]["achieved_auc"] for r in records])
        three = ("tnbbeta", "vmf_init", "power_spherical_init")
        slopes = " / ".join(
            f"{_statistic(records, c, 'cal_slope', np.median):.2f}" for c in three
        )
        shares = " / ".join(f"{_share_above_chance(records, c):.2f}" for c in three)
        losses = " | ".join(
            f"{_statistic(records, c, 'sampled_log_loss', np.mean):.3f}"
            for c in conditions
        )
        lines.append(
            f"| {label} | {len(records)} | {auc:.3f} | {losses} | {slopes} | {shares} |"
        )
    r_bars = [r["mean"]["tnbbeta"]["post_r_bar"] for r in aggregates.values()]
    aucs = [r["spec"]["achieved_auc"] for r in aggregates.values()]
    lines += [
        "",
        f"Spearman(TNBBeta r_bar, oracle AUC) over all configs: "
        f"{spearmanr(r_bars, aucs)[0]:+.2f}.",
    ]
    for label, metric, other in (
        ("diffuse", "sampled_log_loss", "second_mode_mass"),
        ("diffuse", "sampled_log_loss", "oracle AUC"),
        ("tight", "cal_slope_dev", "oracle AUC"),
    ):
        records = [r for r in aggregates.values() if regime_of(r) == label]
        y = [advantage(r, metric) for r in records]
        x = [
            r["spec"]["achieved_auc"]
            if other == "oracle AUC"
            else r["mean"]["tnbbeta"][other]
            for r in records
        ]
        lines.append(
            f"Within the {label} regime, Spearman({other}, {metric} advantage) = "
            f"{spearmanr(x, y)[0]:+.2f} (n = {len(records)})."
        )
    three = ("tnbbeta", "vmf_init", "power_spherical_init")
    for label, metric in (("tight", "cal_slope_dev"), ("diffuse", "sampled_log_loss")):
        wins = [
            r
            for r in aggregates.values()
            if regime_of(r) == label and advantage(r, metric) > CLEAR
        ]
        if wins:
            slopes = " / ".join(
                f"{_statistic(wins, c, 'cal_slope', np.median):.2f}" for c in three
            )
            losses = " / ".join(
                f"{_statistic(wins, c, 'sampled_log_loss', np.mean):.3f}" for c in three
            )
            lines.append(
                f"The {len(wins)} {label}-regime configs with a clear {metric} win: "
                f"median calibration slope {slopes} and mean log loss {losses} "
                "(tnbbeta / vmf / ps)."
            )
    return "\n".join(lines)


def _statistic(
    records: list[dict[str, Any]], condition: str, metric: str, reduce: Any
) -> float:
    """Reduces ``metric`` of ``condition`` over the records' config means."""
    return float(reduce([r["mean"][condition][metric] for r in records]))


def _share_above_chance(records: list[dict[str, Any]], condition: str) -> float:
    """Share of records whose sampled log loss exceeds ln 2."""
    losses = [r["mean"][condition]["sampled_log_loss"] for r in records]
    return float(np.mean([x > math.log(2) for x in losses]))


if __name__ == "__main__":
    main()
