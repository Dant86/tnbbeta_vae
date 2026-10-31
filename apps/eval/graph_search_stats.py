"""Tuning, power and hypothesis tests for the wide graph-type search.

Usage:
    uv run python -m apps.eval.graph_search_stats tune      # pilot -> tuned_lr.json
    uv run python -m apps.eval.graph_search_stats variance  # pilot SDs and draws needed
    uv run python -m apps.eval.graph_search_stats test --stage discovery

Per (config, metric) the null is *TNBBeta is no better than comparator A, or no better
than comparator B* (A, B = vMF, power-spherical): an intersection-union test whose
p-value is the larger of the two one-sided paired Wilcoxon signed-rank p-values, each
testing ``oriented difference > margin`` over independent graph draws (``MARGINS``).
Discovery p-values are corrected over every (config, metric) test with
Benjamini-Hochberg, and confirmation p-values with Holm; Bonferroni is reported
alongside. Pooled per-family tests (config as the replicate) and the reverse direction
(TNBBeta significantly *worse*) are reported as separate families. Nothing here looks at
test outcomes to choose a margin.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from typing import Any, cast

import numpy as np
from scipy.stats import norm, wilcoxon

from apps.synthetic.graph_search import LEARNING_RATES, stage_dir
from tnbbeta_vae.data.graph_space import FAMILIES

# metric -> (higher is better, practical margin delta in the metric's own units).
# Margins are fixed a priori from interpretability (what difference would matter), not
# from the data.
MARGINS: dict[str, tuple[bool, float]] = {
    "centre_auc": (True, 0.005),
    "sampled_log_loss": (False, 0.005),
    "cal_slope_dev": (False, 0.05),
    "uncertainty_auroc": (True, 0.02),
    "truth_kl": (False, 0.005),
    "rank_corr": (True, 0.01),
}
COMPARATOR_SETS = {
    "primary": ("vmf_init", "power_spherical_init"),
    "pinned": ("vmf_pin", "power_spherical_pin"),
    "default": ("vmf_default", "power_spherical_default"),
}
TUNED_CONDITIONS = ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
TUNING_METRIC = "sampled_log_loss"


def main(argv: list[str] | None = None) -> None:
    """Runs the ``tune``, ``variance`` or ``test`` subcommand.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tune")
    sub.add_parser("variance")
    test = sub.add_parser("test")
    test.add_argument("--stage", choices=["discovery", "confirmation"], required=True)
    test.add_argument("--set", choices=sorted(COMPARATOR_SETS), default="primary")
    test.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.command == "tune":
        print(json.dumps(tune_learning_rates(load_runs("pilot")), indent=2))
    elif args.command == "variance":
        print(variance_report(load_runs("pilot")))
    elif args.stage == "confirmation":
        print(confirmation_report(args.alpha))
    else:
        print(render_report(args.stage, args.set, args.alpha))


def load_runs(stage: str) -> list[dict[str, Any]]:
    """Reads every finished run of ``stage``."""
    directory = stage_dir(stage) / "runs"
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def tune_learning_rates(
    runs: list[dict[str, Any]], *, write: bool = True
) -> dict[str, dict[str, float]]:
    """Picks, per graph family and tuned condition, the learning rate with the lowest
    mean :data:`TUNING_METRIC` over all pilot configs and draws of that family.

    Pilot graphs are disjoint from discovery and confirmation graphs, so selecting on
    their test metrics does not leak into the tests. Writes ``pilot/tuned_lr.json``.
    """
    scores: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for run in runs:
        if run["condition"] in TUNED_CONDITIONS:
            key = (run["family"], run["condition"], run["lr"])
            scores[key].append(run["metrics"][TUNING_METRIC])
    tuned: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        for condition in TUNED_CONDITIONS:
            candidates = {
                lr: float(np.mean(scores[(family, condition, lr)]))
                for lr in LEARNING_RATES
                if scores.get((family, condition, lr))
            }
            if candidates:
                tuned.setdefault(family, {})[condition] = min(
                    candidates, key=lambda lr: candidates[lr]
                )
    if write:
        (stage_dir("pilot") / "tuned_lr.json").write_text(json.dumps(tuned, indent=2))
    return tuned


def paired_differences(
    runs: list[dict[str, Any]],
    treatment: str,
    comparator: str,
    metric: str,
    lr_for: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    """Per config, the oriented ``treatment - comparator`` differences over draws.

    Positive means the treatment is better. Draws where either side's metric is not
    finite are dropped. ``lr_for`` restricts pilot runs to each condition's tuned rate.
    """
    higher = MARGINS[metric][0]
    by_key: dict[tuple[str, int, str], float] = {}
    for run in runs:
        if lr_for is not None:
            wanted = lr_for.get(run["family"], {}).get(run["condition"])
            if wanted is not None and run["lr"] != wanted:
                continue
        value = run["metrics"][metric]
        by_key[(run["config_id"], run["draw"], run["condition"])] = (
            value if higher else -value
        )
    differences: dict[str, list[float]] = defaultdict(list)
    for config_id, draw, condition in sorted(by_key):
        if condition != treatment:
            continue
        other = by_key.get((config_id, draw, comparator))
        mine = by_key[(config_id, draw, condition)]
        if other is not None and math.isfinite(mine) and math.isfinite(other):
            differences[config_id].append(mine - other)
    return dict(differences)


def one_sided_p(
    differences: list[float], margin: float, *, better: bool = True
) -> float:
    """Returns the one-sided Wilcoxon signed-rank p-value of a paired effect.

    Tests ``difference > margin`` (``better``) or ``difference < -margin`` (the reverse
    direction); 1.0 if nothing can be tested.
    """
    shifted = np.asarray(differences, dtype=np.float64)
    shifted = shifted - margin if better else -shifted - margin
    if shifted.size < 3 or np.all(shifted == 0):
        return 1.0
    return float(cast("Any", wilcoxon(shifted, alternative="greater")).pvalue)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Returns BH-adjusted p-values (q-values), in the input order."""
    order = np.argsort(p_values)
    count = len(p_values)
    adjusted = np.empty(count)
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def holm(p_values: list[float]) -> list[float]:
    """Returns Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    count = len(p_values)
    adjusted = np.empty(count)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def run_config_tests(
    runs: list[dict[str, Any]],
    comparators: tuple[str, ...],
    *,
    treatment: str = "tnbbeta",
) -> list[dict[str, Any]]:
    """Runs the intersection-union test for every (config, metric).

    Returns:
        One row per test with ``p`` (the max over comparators), the per-comparator ``p``
        values, ``p_reverse`` (TNBBeta significantly worse than *either* comparator, the
        mirrored intersection-union test), the mean oriented difference against each
        comparator, and the number of draws used.
    """
    rows = []
    for metric, (_, margin) in MARGINS.items():
        per_comparator = {
            c: paired_differences(runs, treatment, c, metric) for c in comparators
        }
        for config_id in sorted(
            set.intersection(*(set(d) for d in per_comparator.values()))
        ):
            samples = {c: per_comparator[c][config_id] for c in comparators}
            forward = {c: one_sided_p(v, margin) for c, v in samples.items()}
            reverse = {
                c: one_sided_p(v, margin, better=False) for c, v in samples.items()
            }
            rows.append(
                {
                    "config_id": config_id,
                    "family": config_id.rsplit("-", 1)[0],
                    "metric": metric,
                    "draws": min(len(v) for v in samples.values()),
                    "p": max(forward.values()),
                    "p_reverse": max(reverse.values()),
                    "p_by_comparator": forward,
                    "mean_diff": {c: float(np.mean(v)) for c, v in samples.items()},
                }
            )
    return rows


def pooled_family_tests(
    runs: list[dict[str, Any]], comparators: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Per (family, metric): does TNBBeta beat every comparator across configs?

    The replicate is the config (its draw-mean difference), so this answers "does
    TNBBeta help on this graph family" with far fewer tests than the per-config family.
    """
    rows = []
    for metric, (_, margin) in MARGINS.items():
        means = {
            c: {
                config: float(np.mean(v))
                for config, v in paired_differences(runs, "tnbbeta", c, metric).items()
            }
            for c in comparators
        }
        for family in FAMILIES:
            configs = sorted(
                set.intersection(*(set(m) for m in means.values()))
                & {k for m in means.values() for k in m if k.startswith(family + "-")}
            )
            if len(configs) < 4:
                continue
            p = max(
                one_sided_p([means[c][k] for k in configs], margin) for c in comparators
            )
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "configs": len(configs),
                    "p": p,
                    "mean_diff": {
                        c: float(np.mean([means[c][k] for k in configs]))
                        for c in comparators
                    },
                }
            )
    return rows


def required_draws(sd: float, margin: float, alpha: float, power: float = 0.8) -> int:
    """Draws for a one-sided paired test of a ``margin`` effect (normal approx.)."""
    if sd <= 0 or margin <= 0:
        return 2
    return math.ceil(((norm.ppf(1 - alpha) + norm.ppf(power)) * sd / margin) ** 2)


def variance_report(runs: list[dict[str, Any]], alpha: float = 1e-3) -> str:
    """Markdown table of pilot paired-difference SDs and the draws each metric needs."""
    tuned = tune_learning_rates(runs, write=False)
    lines = [
        "| metric | margin | median SD of paired diff | "
        f"draws for 80% power at alpha={alpha} |",
        "|---|---|---|---|",
    ]
    for metric, (_, margin) in MARGINS.items():
        sds = [
            float(np.std(v, ddof=1))
            for comparator in COMPARATOR_SETS["primary"]
            for v in paired_differences(
                runs, "tnbbeta", comparator, metric, tuned
            ).values()
            if len(v) >= 3
        ]
        if sds:
            sd = float(np.median(sds))
            draws = required_draws(sd, margin, alpha)
            lines.append(f"| {metric} | {margin} | {sd:.4f} | {draws} |")
    return "\n".join(lines)


def render_report(stage: str, comparator_set: str, alpha: float) -> str:
    """Markdown report: per-config tests (BH and Holm), pooled family tests, reverse."""
    runs = load_runs(stage)
    comparators = COMPARATOR_SETS[comparator_set]
    rows = run_config_tests(runs, comparators)
    if not rows:
        return "No complete (config, metric) pairs yet."
    for name, fn in (("q_bh", benjamini_hochberg), ("p_holm", holm)):
        for row, value in zip(rows, fn([r["p"] for r in rows]), strict=True):
            row[name] = value
    reverse_q = benjamini_hochberg([r["p_reverse"] for r in rows])
    bonferroni = alpha / len(rows)
    rejected = sorted((r for r in rows if r["q_bh"] <= alpha), key=lambda r: r["q_bh"])
    worse = [r for r, q in zip(rows, reverse_q, strict=True) if q <= alpha]
    pooled = pooled_family_tests(runs, comparators)
    for row, value in zip(pooled, holm([r["p"] for r in pooled]), strict=True):
        row["p_holm"] = value
    out = [
        f"## {stage}, comparators {comparators} (alpha = {alpha})",
        f"- tests: {len(rows)}; BH rejections: {len(rejected)}; "
        f"Holm rejections: {sum(r['p_holm'] <= alpha for r in rows)}; "
        f"Bonferroni threshold: {bonferroni:.2e}",
        "- reverse direction (TNBBeta significantly worse, BH): "
        f"{len(worse)} of {len(rows)}",
        "",
        "### Rejected nulls (TNBBeta better than every comparator), BH",
        "",
        "| config | metric | draws | q (BH) | p (Holm) | mean diff vs comparators |",
        "|---|---|---|---|---|---|",
    ]
    for r in rejected:
        diffs = ", ".join(f"{c}: {d:+.4f}" for c, d in r["mean_diff"].items())
        out.append(
            f"| {r['config_id']} | {r['metric']} | {r['draws']} | {r['q_bh']:.2e} | "
            f"{r['p_holm']:.2e} | {diffs} |"
        )
    out += [
        "",
        "### Pooled per graph family (config as replicate), Holm",
        "",
        "| family | metric | configs | p | p (Holm) | mean diff vs each comparator |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(pooled, key=lambda r: r["p"]):
        diffs = ", ".join(f"{c}: {d:+.4f}" for c, d in r["mean_diff"].items())
        out.append(
            f"| {r['family']} | {r['metric']} | {r['configs']} | {r['p']:.2e} | "
            f"{r['p_holm']:.2e} | {diffs} |"
        )
    return "\n".join(out)


def rejected_pairs(rows: list[dict[str, Any]], alpha: float) -> list[tuple[str, str]]:
    """Returns the (config, metric) pairs with discovery BH q-value at most alpha."""
    return sorted((r["config_id"], r["metric"]) for r in rows if r["q_bh"] <= alpha)


def confirmation_report(alpha: float) -> str:
    """Tests exactly the discovery-rejected (config, metric) pairs on fresh draws.

    The pairs come from ``discovery/test_rows.json`` (primary comparator set, BH
    q <= alpha); the confirmation p-values are Holm-corrected over just those pairs. A
    pair is *confirmed* when its Holm p-value is at most alpha. Also reports each pair's
    mean difference in discovery and in confirmation, to show shrinkage.
    """
    discovery = json.loads((stage_dir("discovery") / "test_rows.json").read_text())
    primary = discovery["primary"]
    wanted = set(rejected_pairs(primary, alpha))
    first = {(r["config_id"], r["metric"]): r for r in primary}
    rows = [
        r
        for r in run_config_tests(load_runs("confirmation"), COMPARATOR_SETS["primary"])
        if (r["config_id"], r["metric"]) in wanted
    ]
    if not rows:
        return "No complete confirmation pairs yet."
    for row, value in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = value
    confirmed = [r for r in rows if r["p_holm"] <= alpha]
    out = [
        f"## confirmation of {len(wanted)} discovery-rejected pairs "
        f"(Holm, alpha = {alpha})",
        f"- pairs tested: {len(rows)} of {len(wanted)}; confirmed: {len(confirmed)}",
        "",
        "| config | metric | confirmed | p (Holm) | mean diff: discovery -> confirm |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["p_holm"]):
        before = first[(r["config_id"], r["metric"])]["mean_diff"]
        after = r["mean_diff"]
        shown = "; ".join(f"{c}: {before[c]:+.4f} -> {after[c]:+.4f}" for c in after)
        out.append(
            f"| {r['config_id']} | {r['metric']} | "
            f"{'yes' if r['p_holm'] <= alpha else 'no'} | {r['p_holm']:.2e} | {shown} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    main()
