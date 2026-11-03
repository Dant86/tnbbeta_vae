"""Analysis of the metric-first graph-search study, exactly as pre-registered.

Usage (see ``writeup/results/graph_search_metric_first_preregistration_2026-11-02.md``):
    uv run python -m apps.eval.graph_search_metric_first cache --stage discovery
    uv run python -m apps.eval.graph_search_metric_first explore    # discovery, exploratory
    uv run python -m apps.eval.graph_search_metric_first fresh      # fresh, confirmatory

``cache`` reads every run JSON of a stage with a thread pool and writes one compact
per-draw table; ``explore`` runs Step 0-5 on the discovery configs (writing
``prediction_rule.json``); ``fresh`` evaluates the frozen rule and the config x metric
matrix on the fresh stage. Everything is written under
``$TNBBETA_CHECKPOINT_DIR/graph_search/metric_first/`` and printed as markdown.
"""
# ruff: noqa: E501  (long markdown report strings)

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata, spearmanr

from apps.eval.graph_search_covariates import covariate_path
from apps.eval.graph_search_stats import benjamini_hochberg, holm, one_sided_p
from apps.synthetic.graph_search import stage_dir
from tnbbeta_vae.data.graph_covariates import COVARIATE_NAMES
from tnbbeta_vae.data.graph_space import FAMILIES
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.regularised import (
    ElasticNetFit,
    block_permutation,
    fit_elastic_net,
    lambda_grid,
    leave_family_out,
    select_lambda,
)

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
PRIMARY = ("vmf_init", "power_spherical_init")
PINNED = ("vmf_pin", "power_spherical_pin")
BOOTSTRAP = 10_000
PERMUTATIONS = 2_000
PLAIN_PERMUTATIONS = 1_000
ALPHA = 0.05
HEADLINE_MARGIN = 0.005
SELECTION_SHARE = 0.8
CLUSTER_RHO = 0.8
M_NAMES = (
    "post_p_mean",
    "post_q_mean",
    "post_epsilon_mean",
    "post_m_mean",
    "post_p_std",
    "post_epsilon_std",
    "post_frac_bimodal",
    "post_r_bar",
    "second_mode_mass",
    "frac_nodes_second_mode",
    "_best_epoch",
    "_diverged",
)
_LOG_TRANSFORMED = ("expected_degree_cv", "spectral_gap", "affinity_erank")


@dataclass(frozen=True)
class MetricSpec:
    """Orientation, practical margin and tier of one metric."""

    higher: bool
    margin: float
    confirmatory: bool = False


METRICS: dict[str, MetricSpec] = {
    "truth_kl": MetricSpec(False, 0.005, True),
    "sampled_log_loss": MetricSpec(False, 0.005, True),
    "truth_kl_w": MetricSpec(False, 0.005),
    "recal_truth_kl": MetricSpec(False, 0.005),
    "truth_kl_eps1e4": MetricSpec(False, 0.005),
    "cal_slope_dev": MetricSpec(False, 0.05),
    "ece": MetricSpec(False, 0.01),
    "hard_hits10": MetricSpec(True, 0.02),
    "hard_hits50": MetricSpec(True, 0.02),
    "hardc_hits10": MetricSpec(True, 0.02),
    "hardc_hits50": MetricSpec(True, 0.02),
    "hard_mrr": MetricSpec(True, 0.01),
    "hardc_mrr": MetricSpec(True, 0.01),
    "gs_degree_w1": MetricSpec(False, 0.05),
    "gs_clustering_err": MetricSpec(False, 0.05),
    "gs_triangle_err": MetricSpec(False, 0.05),
    "latent_gram_rho": MetricSpec(True, 0.02),
    "latent_nmi": MetricSpec(True, 0.02),
    "latent_procrustes": MetricSpec(False, 0.02),
    "centre_auc": MetricSpec(True, 0.005),
    "sampled_auc": MetricSpec(True, 0.005),
    "rank_corr": MetricSpec(True, 0.01),
    "uncertainty_auroc": MetricSpec(True, 0.02),
}
STORED_DISCOVERY = (
    "truth_kl",
    "sampled_log_loss",
    "cal_slope_dev",
    "centre_auc",
    "sampled_auc",
    "rank_corr",
    "uncertainty_auroc",
)

Table = dict[str, dict[str, dict[str, Array]]]


def main(argv: list[str] | None = None) -> None:
    """Dispatches ``cache``, ``explore`` or ``fresh``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--stage", required=True)
    cache.add_argument("--workers", type=int, default=32)
    explore = sub.add_parser("explore")
    explore.add_argument("--workers", type=int, default=8)
    sub.add_parser("fresh")
    args = parser.parse_args(argv)
    if args.command == "cache":
        print(build_cache(args.stage, args.workers))
    elif args.command == "explore":
        print(explore_report(args.workers))
    else:
        print(fresh_report())


def output_dir() -> Path:
    """Returns (creating it) the directory of this study's derived files."""
    path = checkpoint_dir() / "graph_search" / "metric_first"
    path.mkdir(parents=True, exist_ok=True)
    return path


def family_of(config_id: str) -> str:
    """Returns the graph family of a config id (``ring-03`` -> ``ring``)."""
    return config_id.rsplit("-", 1)[0]


# --------------------------------------------------------------------------- data


def build_cache(stage: str, workers: int) -> Path:
    """Reads every run JSON of ``stage`` and writes the compact per-draw table."""
    paths = sorted((stage_dir(stage) / "runs").glob("*.json"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(_read, paths))
    cache: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for run in runs:
        cell = cache.setdefault(run["config_id"], {}).setdefault(run["condition"], {})
        values = {
            **run["metrics"],
            "_best_epoch": run["best_epoch"],
            "_diverged": run["diverged"],
        }
        for key, value in values.items():
            cell.setdefault(key, {})[run["draw"]] = float(value)
    path = output_dir() / f"runs_{stage}.json"
    path.write_text(json.dumps(cache))
    return path


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_table(stage: str) -> Table:
    """Loads the cache as ``{config: {condition: {metric: array over draws}}}``."""
    raw = json.loads((output_dir() / f"runs_{stage}.json").read_text())
    draws = 1 + max(
        int(d) for c in raw.values() for m in c.values() for v in m.values() for d in v
    )
    table: Table = {}
    for config, conditions in raw.items():
        for condition, metrics in conditions.items():
            for metric, values in metrics.items():
                array = np.full(draws, np.nan)
                for draw, value in values.items():
                    array[int(draw)] = value
                table.setdefault(config, {}).setdefault(condition, {})[metric] = array
    return table


def load_covariates(stage: str) -> dict[str, dict[str, float]]:
    """Per config, the mean over draws of every Set G covariate (and ``kl0``)."""
    raw = json.loads(covariate_path(stage).read_text())
    return {
        config: {
            key: float(np.mean([d[key] for d in draws.values()]))
            for key in next(iter(draws.values()))
        }
        for config, draws in raw.items()
    }


def design_matrix(
    covariates: dict[str, dict[str, float]], configs: list[str]
) -> tuple[Array, list[str]]:
    """Set G as a matrix with the pre-registered log transforms (g8, g11, g12)."""
    columns = []
    for name in COVARIATE_NAMES:
        values = np.array([covariates[c][name] for c in configs])
        if name in _LOG_TRANSFORMED:
            values = np.log(np.maximum(values, 1e-6))
        columns.append(values)
    return np.stack(columns, axis=1), list(COVARIATE_NAMES)


def model_side(table: Table, configs: list[str]) -> dict[str, Array]:
    """Set M: TNBBeta's config-mean posterior statistics and the comparators' r_bar."""
    out: dict[str, Array] = {}
    for name in M_NAMES:
        out[name] = np.array(
            [_nanmean(table[c]["tnbbeta"].get(name)) for c in configs], dtype=np.float64
        )
    for condition in PRIMARY:
        out[f"{condition}_r_bar"] = np.array(
            [_nanmean(table[c][condition].get("post_r_bar")) for c in configs]
        )
    return out


def _nanmean(values: Array | None) -> float:
    if values is None or not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


# ------------------------------------------------------------------- effects


def oriented_difference(
    table: Table, config: str, comparator: str, metric: str
) -> Array:
    """Per draw, ``comparator - tnbbeta`` (lower-better) or ``tnbbeta - comparator``."""
    treated = table[config]["tnbbeta"][metric]
    other = table[config][comparator][metric]
    return treated - other if METRICS[metric].higher else other - treated


def config_metric_record(
    table: Table,
    config: str,
    metric: str,
    comparators: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """The effect record of one (config, metric): means, dz, E, interval, IU p-values."""
    try:
        diffs = np.stack(
            [oriented_difference(table, config, c, metric) for c in comparators]
        )
    except KeyError:
        return None
    keep = np.all(np.isfinite(diffs), axis=0)
    if int(keep.sum()) < 3:
        return None
    diffs = diffs[:, keep]
    margin = METRICS[metric].margin
    means = diffs.mean(1)
    sds = diffs.std(1, ddof=1)
    dz = np.where(sds > 0, means / np.where(sds > 0, sds, 1.0), 0.0)
    draws = diffs.shape[1]
    index = rng.integers(0, draws, (BOOTSTRAP, draws))
    boot = diffs[:, index].mean(-1).min(0)
    forward = [one_sided_p(list(d), margin) for d in diffs]
    reverse = [one_sided_p(list(d), margin, better=False) for d in diffs]
    return {
        "config_id": config,
        "family": family_of(config),
        "metric": metric,
        "draws": draws,
        "mean_d": dict(zip(comparators, means.tolist(), strict=True)),
        "dz": dict(zip(comparators, dz.tolist(), strict=True)),
        "E": float(means.min()),
        "E_lo": float(np.percentile(boot, 2.5)),
        "E_hi": float(np.percentile(boot, 97.5)),
        "dz_min": float(dz.min()),
        "p": max(forward),
        "p_reverse": max(reverse),
    }


def all_records(
    table: Table,
    metrics: tuple[str, ...],
    comparators: tuple[str, ...],
    seed: int = 20261102,
) -> list[dict[str, Any]]:
    """Effect records for every config and every metric with data."""
    rng = np.random.default_rng(seed)
    records = []
    for metric in metrics:
        for config in sorted(table):
            record = config_metric_record(table, config, metric, comparators, rng)
            if record is not None:
                records.append(record)
    return records


def classify(records: list[dict[str, Any]], *, fresh: bool) -> None:
    """Adds ``adj``, ``adj_reverse`` and ``label`` (better / worse / neither) in place.

    Fresh stage: Holm over the confirmatory metrics' tests, BH over all other tests.
    Discovery (exploratory): BH over every test. Forward and reverse p-values are
    adjusted separately within the same families.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            "confirmatory"
            if fresh and METRICS[record["metric"]].confirmatory
            else "rest"
        )
        groups[key].append(record)
    for key, rows in groups.items():
        method = holm if (fresh and key == "confirmatory") else benjamini_hochberg
        for field, name in (("p", "adj"), ("p_reverse", "adj_reverse")):
            for row, value in zip(rows, method([r[field] for r in rows]), strict=True):
                row[name] = value
    for record in records:
        if record["adj"] <= ALPHA:
            record["label"] = "better"
        elif record["adj_reverse"] <= ALPHA:
            record["label"] = "worse"
        else:
            record["label"] = "neither"


def family_metric_tables(
    records: list[dict[str, Any]], metrics: tuple[str, ...]
) -> str:
    """Markdown: counts better/worse/neither and median dz per family x metric."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[(r["family"], r["metric"])].append(r)
    shown = [m for m in metrics if any(r["metric"] == m for r in records)]
    header = "| family | " + " | ".join(shown) + " |\n|---|" + "---|" * len(shown)
    counts, medians = [header], [header]
    for family in (*FAMILIES, "all"):
        row_counts, row_medians = [], []
        for metric in shown:
            rows = (
                [r for r in records if r["metric"] == metric]
                if family == "all"
                else cells.get((family, metric), [])
            )
            if not rows:
                row_counts.append("-")
                row_medians.append("-")
                continue
            tally = [
                sum(r["label"] == k for r in rows)
                for k in ("better", "worse", "neither")
            ]
            row_counts.append("/".join(map(str, tally)))
            row_medians.append(f"{np.median([r['dz_min'] for r in rows]):+.2f}")
        counts.append(f"| {family} | " + " | ".join(row_counts) + " |")
        medians.append(f"| {family} | " + " | ".join(row_medians) + " |")
    return (
        "(cells: better / worse / neither configs)\n\n"
        + "\n".join(counts)
        + "\n\nMedian standardised effect dz of the less favourable comparator:\n\n"
        + "\n".join(medians)
    )


# ------------------------------------------------------- outcomes and step 0


def outcome_vectors(
    records: list[dict[str, Any]],
    covariates: dict[str, dict[str, float]],
    configs: list[str],
) -> tuple[Array, Array, Array, Array]:
    """``E`` (nats), ``S = E / KL0``, and the interval ends of ``E`` per config."""
    by_config = {r["config_id"]: r for r in records if r["metric"] == "truth_kl"}
    energy = np.array([by_config[c]["E"] for c in configs])
    low = np.array([by_config[c]["E_lo"] for c in configs])
    high = np.array([by_config[c]["E_hi"] for c in configs])
    kl0 = np.array([covariates[c]["kl0"] for c in configs])
    return energy, energy / kl0, low, high


def split_half_reliability(table: Table, configs: list[str]) -> tuple[float, float]:
    """Spearman correlation of ``E_c`` between draw halves and its Spearman-Brown value."""
    firsts, seconds = [], []
    for config in configs:
        diffs = np.stack(
            [oriented_difference(table, config, c, "truth_kl") for c in PRIMARY]
        )
        draws = diffs.shape[1]
        halves = (np.arange(draws) < draws // 2, np.arange(draws) >= draws // 2)
        values = []
        for half in halves:
            sub = diffs[:, half]
            ok = np.all(np.isfinite(sub), axis=0)
            values.append(float(sub[:, ok].mean(1).min()) if ok.any() else float("nan"))
        firsts.append(values[0])
        seconds.append(values[1])
    a, b = np.array(firsts), np.array(seconds)
    ok = np.isfinite(a) & np.isfinite(b)
    rho = float(cast("Any", spearmanr(a[ok], b[ok]))[0])
    return rho, 2 * rho / (1 + rho)


# --------------------------------------------------------- exploratory steps


def clusters_of(x: Array, names: list[str]) -> list[list[int]]:
    """Connected components of the graph on columns with ``|Spearman| > 0.8``."""
    p = x.shape[1]
    rho = np.abs(cast("Any", spearmanr(x))[0]) if p > 1 else np.ones((1, 1))
    rho = np.nan_to_num(rho, nan=0.0)
    seen: set[int] = set()
    out = []
    for start in range(p):
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            component.append(i)
            stack.extend(
                j for j in range(p) if j not in seen and rho[i, j] > CLUSTER_RHO
            )
        out.append(sorted(component))
    return out


def _perm_worker(args: tuple[int, Array, Array, IntArray, bool]) -> float:
    seed, x, y, groups, blocks = args
    rng = np.random.default_rng(seed)
    order = block_permutation(groups, rng) if blocks else rng.permutation(len(y))
    return leave_family_out(x, y[order], groups)[0]


def permutation_p(
    x: Array,
    y: Array,
    groups: IntArray,
    observed: float,
    count: int,
    blocks: bool,
    workers: int,
    seed: int,
) -> float:
    """One-sided permutation p-value of an observed leave-family-out R^2."""
    tasks = [(seed + i, x, y, groups, blocks) for i in range(count)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = np.array(list(pool.map(_perm_worker, tasks, chunksize=8)))
    return float((1 + (values >= observed).sum()) / (1 + count))


def within_family_r2(
    x: Array, y: Array, groups: IntArray, repeats: int = 20, seed: int = 0
) -> float:
    """Repeated 4-fold CV, stratified by family: covariates beyond family means."""
    rng = np.random.default_rng(seed)
    centred = x.copy()
    for g in np.unique(groups):
        centred[groups == g] -= x[groups == g].mean(0)
    scores = []
    for _ in range(repeats):
        fold = np.zeros(len(y), dtype=int)
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            fold[rng.permutation(rows)] = np.arange(rows.size) % 4
        sse_model = sse_base = 0.0
        for k in range(4):
            train, test = fold != k, fold == k
            means = {g: y[train & (groups == g)].mean() for g in np.unique(groups)}
            base = np.array([means[g] for g in groups])
            residual = y - base
            inner_groups = rng.integers(0, 3, int(train.sum()))
            lam = select_lambda(
                centred[train], residual[train], inner_groups, rule="min"
            )
            fit = fit_elastic_net(centred[train], residual[train], np.array([lam]))[0]
            prediction = base[test] + fit.predict(centred[test])
            sse_model += float(((y[test] - prediction) ** 2).sum())
            sse_base += float(((y[test] - base[test]) ** 2).sum())
        scores.append(1.0 - sse_model / sse_base)
    return float(np.mean(scores))


def explore_report(workers: int) -> str:
    """Steps 0-5 on the discovery configs (exploratory); writes the prediction rule."""
    table = load_table("discovery")
    configs_all = json.loads((stage_dir("discovery") / "configs.json").read_text())
    configs = [c["config_id"] for c in configs_all]
    covariates = load_covariates("discovery")
    families = np.array([family_of(c) for c in configs])
    groups = np.array([FAMILIES.index(f) for f in families], dtype=np.int64)
    records = all_records(table, STORED_DISCOVERY, PRIMARY)
    classify(records, fresh=False)
    energy, skill, low, high = outcome_vectors(records, covariates, configs)
    x, names = design_matrix(covariates, configs)
    model = model_side(table, configs)
    rho_half, reliability = split_half_reliability(table, configs)
    out = ["# Exploratory analysis of the discovery configs", ""]
    out += _step0(
        configs, families, energy, skill, low, high, rho_half, reliability, x, names
    )
    informative = np.std(x, axis=0) > 0
    xs, used = (
        x[:, informative],
        [n for n, k in zip(names, informative, strict=True) if k],
    )
    result = _step1(xs, skill, groups, workers)
    result_nats = leave_family_out(xs, energy, groups)[0]
    out += [
        "## Step 1: predictability (outcome S_c, elastic net, leave-one-family-out)",
        "",
        f"- R^2_LOFO = {result['r2']:+.3f}; block-permutation p = {result['p_block']:.4f} "
        f"({PERMUTATIONS} permutations); plain-permutation p (optimistic) = "
        f"{result['p_plain']:.4f} ({PLAIN_PERMUTATIONS})",
        f"- reliability ceiling (Spearman-Brown split-half) = {reliability:.3f}; "
        f"25% of it = {0.25 * reliability:.3f}",
        f"- secondary: outcome E_c in nats R^2_LOFO = {result_nats:+.3f}",
    ]
    keep = families != "bipartite"
    out.append(
        "- secondary: excluding bipartite R^2_LOFO = "
        f"{leave_family_out(xs[keep], skill[keep], groups[keep])[0]:+.3f}"
    )
    diverged = model["_diverged"] > 0.1
    out.append(
        f"- secondary: excluding {int(diverged.sum())} configs with TNBBeta divergence rate > 10% "
        f"R^2_LOFO = {leave_family_out(xs[~diverged], skill[~diverged], groups[~diverged])[0]:+.3f}"
    )
    out.append(
        f"- secondary: within-family increment (repeated 4-fold, 20x) R^2 = "
        f"{within_family_r2(xs, skill, groups):+.3f}"
    )
    out += ["", *_steps23(xs, used, skill, groups, result, configs)]
    rule_path = _write_rule(xs, used, skill, groups)
    out += [
        "",
        f"- prediction rule written to `{rule_path}`; sha256 "
        f"`{hashlib.sha256(rule_path.read_bytes()).hexdigest()}`",
    ]
    out += [
        "",
        *_step4(records, configs, families, covariates, model, skill, low, high),
    ]
    out += ["", *_step5(model, skill, families)]
    out += [
        "",
        "## Config x metric matrix (discovery, stored metrics, BH over all tests)",
        "",
        family_metric_tables(records, STORED_DISCOVERY),
    ]
    summary = {
        "result": result,
        "reliability": reliability,
        "records": records,
        "configs": configs,
        "E": energy.tolist(),
        "S": skill.tolist(),
    }
    (output_dir() / "results_exploratory.json").write_text(json.dumps(summary))
    return "\n".join(out)


def _step0(
    configs: list[str],
    families: Array,
    energy: Array,
    skill: Array,
    low: Array,
    high: Array,
    rho_half: float,
    reliability: float,
    x: Array,
    names: list[str],
) -> list[str]:
    out = [
        "## Step 0: outcomes and covariate marginals (no outcome-covariate relation)",
        "",
        f"- configs: {len(configs)}; split-half Spearman of E_c = {rho_half:.3f}, "
        f"Spearman-Brown reliability = {reliability:.3f}",
        "",
        "| family | median E_c (nats) | median S_c | min E_c | max E_c |",
        "|---|---|---|---|---|",
    ]
    for family in FAMILIES:
        pick = families == family
        out.append(
            f"| {family} | {np.median(energy[pick]):+.4f} | {np.median(skill[pick]):+.3f} | "
            f"{energy[pick].min():+.4f} | {energy[pick].max():+.4f} |"
        )
    out += ["", "| covariate | median | min | max |", "|---|---|---|---|"]
    for j, name in enumerate(names):
        out.append(
            f"| {name} | {np.median(x[:, j]):.3f} | {x[:, j].min():.3f} | {x[:, j].max():.3f} |"
        )
    return out


def _step1(x: Array, y: Array, groups: IntArray, workers: int) -> dict[str, Any]:
    r2, predictions, fits = leave_family_out(x, y, groups)
    p_block = permutation_p(x, y, groups, r2, PERMUTATIONS, True, workers, 1_000)
    p_plain = permutation_p(x, y, groups, r2, PLAIN_PERMUTATIONS, False, workers, 9_000)
    return {
        "r2": r2,
        "p_block": p_block,
        "p_plain": p_plain,
        "fit_coefficients": [f.coefficients.tolist() for f in fits],
        "predictions": predictions.tolist(),
    }


def _steps23(
    x: Array,
    names: list[str],
    y: Array,
    groups: IntArray,
    result: dict[str, Any],
    configs: list[str],
) -> list[str]:
    coefficients = np.array(result["fit_coefficients"])
    lam = select_lambda(x, y, groups)
    full = fit_elastic_net(x, y, np.array([lam]))[0]
    clusters = clusters_of(x, names)
    out = [
        "## Step 2: selection and stability",
        "",
        f"- full-data penalty (one-SE rule): {lam:.4g}; leave-one-family-out fits: {len(coefficients)}",
        "",
        "| cluster (|rho| > 0.8) | member | full-fit coefficient | nonzero share | constant sign | stable |",
        "|---|---|---|---|---|---|",
    ]
    stable_clusters = []
    for number, members in enumerate(clusters):
        stable_any = False
        for j in members:
            column = coefficients[:, j]
            share = float(np.mean(column != 0))
            signs = np.sign(column[column != 0])
            constant = signs.size > 0 and bool(np.all(signs == signs[0]))
            stable = share >= SELECTION_SHARE and constant
            stable_any = stable_any or stable
            out.append(
                f"| {number} | {names[j]} | {full.coefficients[j]:+.4f} | {share:.2f} | "
                f"{constant} | {stable} |"
            )
        if stable_any:
            stable_clusters.append(members)
    out += [
        "",
        f"- selected clusters: {len(stable_clusters)}",
        "",
        "## Step 3: partial dependence",
        "",
    ]
    ranked = sorted(
        stable_clusters, key=lambda m: -max(abs(full.coefficients[j]) for j in m)
    )[:3]
    rng = np.random.default_rng(3)
    for members in ranked:
        j = max(members, key=lambda k: abs(full.coefficients[k]))
        boots = []
        for _ in range(300):
            pick = rng.integers(0, len(y), len(y))
            boots.append(
                fit_elastic_net(x[pick], y[pick], np.array([lam]))[0].coefficients[j]
            )
        z = (x[:, j] - full.mean[j]) / full.scale[j]
        partial = (
            y
            - full.intercept
            - (((x - full.mean) / full.scale) @ full.coefficients)
            + full.coefficients[j] * z
        )
        edges = np.quantile(x[:, j], [1 / 3, 2 / 3])
        tert = np.digitize(x[:, j], edges)
        means = [float(partial[tert == t].mean()) for t in range(3)]
        out.append(
            f"- `{names[j]}`: slope {full.coefficients[j]:+.4f} S_c per SD "
            f"(bootstrap 95% [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]); "
            f"partial-residual means by tertile (low/mid/high) {means[0]:+.3f} / {means[1]:+.3f} / {means[2]:+.3f}; "
            f"tertile trend sign {'agrees' if np.sign(means[2] - means[0]) == np.sign(full.coefficients[j]) else 'disagrees'}"
        )
    if not ranked:
        out.append("- no covariate cluster met the 80% stability rule")
    return out


def _step4(
    records: list[dict[str, Any]],
    configs: list[str],
    families: Array,
    covariates: dict[str, dict[str, float]],
    model: dict[str, Array],
    skill: Array,
    low: Array,
    high: Array,
) -> list[str]:
    tiers = np.where(
        low > HEADLINE_MARGIN,
        "win",
        np.where(high < -HEADLINE_MARGIN, "loss", "inconclusive"),
    )
    out = [
        "## Step 4: descriptive characterisation by tier (labels are descriptive only)",
        "",
    ]
    out.append("| family | win | inconclusive | loss |\n|---|---|---|---|")
    for family in FAMILIES:
        pick = families == family
        out.append(
            f"| {family} | "
            + " | ".join(
                str(int(np.sum(tiers[pick] == t)))
                for t in ("win", "inconclusive", "loss")
            )
            + " |"
        )
    out += [
        "",
        "| covariate | median (win) | median (inconclusive) | median (loss) |",
        "|---|---|---|---|",
    ]
    for name in (*COVARIATE_NAMES, *model):
        if name in model:
            values = model[name]
        else:
            values = np.array([covariates[c][name] for c in configs])
        cells = []
        for t in ("win", "inconclusive", "loss"):
            sub = values[tiers == t]
            sub = sub[np.isfinite(sub)]
            cells.append(f"{np.median(sub):.3f}" if sub.size else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def _step5(model: dict[str, Array], skill: Array, families: Array) -> list[str]:
    out = [
        "## Step 5: model-side description (co-occurrence only)",
        "",
        "| model-side variable | Spearman with S_c | within-family (family means removed from ranks) |",
        "|---|---|---|",
    ]
    for name, values in model.items():
        ok = np.isfinite(values)
        if ok.sum() < 5 or np.std(values[ok]) == 0:
            continue
        pooled = cast("Any", spearmanr(values[ok], skill[ok]))[0]
        ra, rb = rankdata(values[ok]), rankdata(skill[ok])
        for f in np.unique(families[ok]):
            pick = families[ok] == f
            ra[pick] -= ra[pick].mean()
            rb[pick] -= rb[pick].mean()
        within = float(np.corrcoef(ra, rb)[0, 1])
        out.append(f"| {name} | {pooled:+.2f} | {within:+.2f} |")
    return out


def _write_rule(x: Array, names: list[str], y: Array, groups: IntArray) -> Path:
    lam = select_lambda(x, y, groups)
    fit = fit_elastic_net(x, y, np.array([lam]))[0]
    path = output_dir() / "prediction_rule.json"
    path.write_text(
        json.dumps(
            {
                "names": names,
                "mean": fit.mean.tolist(),
                "scale": fit.scale.tolist(),
                "coefficients": fit.coefficients.tolist(),
                "intercept": fit.intercept,
                "lam": fit.lam,
                "discovery_mean_S": float(y.mean()),
            },
            indent=1,
        )
    )
    return path


# ------------------------------------------------------------------- fresh


def _rule_fit(rule: dict[str, Any]) -> ElasticNetFit:
    return ElasticNetFit(
        np.array(rule["mean"]),
        np.array(rule["scale"]),
        np.array(rule["coefficients"]),
        float(rule["intercept"]),
        float(rule["lam"]),
    )


def _spearman(a: Array, b: Array) -> float:
    return float(cast("Any", spearmanr(a, b))[0])


def fresh_report() -> str:
    """Confirmatory evaluation on the fresh stage, exactly as pre-registered."""
    table = load_table("fresh")
    configs_all = json.loads((stage_dir("fresh") / "configs.json").read_text())
    configs = [c["config_id"] for c in configs_all]
    families = np.array([family_of(c) for c in configs])
    covariates = load_covariates("fresh")
    rule = json.loads((output_dir() / "prediction_rule.json").read_text())
    x_all, names = design_matrix(covariates, configs)
    x = x_all[:, [names.index(n) for n in rule["names"]]]
    fit = _rule_fit(rule)
    prediction = fit.predict(x)
    metrics = tuple(m for m in METRICS if any(m in table[c]["tnbbeta"] for c in table))
    records = all_records(table, metrics, PRIMARY)
    classify(records, fresh=True)
    energy, skill, low, high = outcome_vectors(records, covariates, configs)
    base = float(rule["discovery_mean_S"])
    r2 = 1.0 - float(((skill - prediction) ** 2).sum() / ((skill - base) ** 2).sum())
    rho = _spearman(prediction, skill)
    rng = np.random.default_rng(77)
    boots = []
    index_by_family = {f: np.flatnonzero(families == f) for f in np.unique(families)}
    for _ in range(BOOTSTRAP):
        pick = np.concatenate(
            [rng.choice(rows, rows.size) for rows in index_by_family.values()]
        )
        boots.append(_spearman(prediction[pick], skill[pick]))
    lower = float(np.percentile(boots, 5))
    within = _within_family_spearman(prediction, skill, families)
    degree = x_all[:, names.index("log_nominal_degree")]
    h1_stat, h1_p = _h1(degree, skill, families, rng)
    coefficients = np.array(rule["coefficients"])
    top = int(np.argmax(np.abs(coefficients)))
    top_rho = _spearman(x[:, top], skill)
    sign_match = bool(np.sign(top_rho) == np.sign(coefficients[top]))
    exploratory = json.loads((output_dir() / "results_exploratory.json").read_text())[
        "result"
    ]
    out = [
        "# Confirmatory evaluation on the fresh stage",
        "",
        f"- fresh configs: {len(configs)}; draws per config: {table[configs[0]]['tnbbeta']['truth_kl'].size}",
        f"- rule sha256: `{hashlib.sha256((output_dir() / 'prediction_rule.json').read_bytes()).hexdigest()}`",
        "",
        "## Prediction rule on fresh configs (outcome S_c)",
        "",
        f"- R^2_fresh = {r2:+.3f} (baseline: mean discovery S_c = {base:+.3f})",
        f"- rho_fresh (Spearman predicted vs observed) = {rho:+.3f}; lower 95% bound "
        f"(5th percentile of {BOOTSTRAP} within-family bootstrap resamples) = {lower:+.3f}",
        f"- within-fresh-family Spearman (family means removed) = {within:+.3f}",
        f"- top coefficient `{rule['names'][top]}`: fresh Spearman with S_c = {top_rho:+.3f} "
        f"(sign {'matches' if sign_match else 'does not match'})",
        f"- H1 (log_nominal_degree, within family): statistic {h1_stat:+.3f}, one-sided permutation "
        f"p = {h1_p:.4f}",
        "",
        "## Decision rule",
        "",
        f"1. discovery R^2_LOFO > 0, block p <= 0.05, > 25% of reliability: "
        f"R^2_LOFO = {exploratory['r2']:+.3f}, p = {exploratory['p_block']:.4f} -> "
        f"{exploratory['r2'] > 0 and exploratory['p_block'] <= ALPHA}",
        f"3. rho_fresh >= 0.30, lower bound > 0, R^2_fresh > 0: "
        f"{rho >= 0.30 and lower > 0 and r2 > 0}",
        f"4. top-cluster sign matches: {sign_match}",
        "",
        "## Confirmatory family-level statement (median E_c by family, bootstrap over configs)",
        "",
        "| family | configs | median E_c (nats, truth_kl) | 95% interval | above 0 |",
        "|---|---|---|---|---|",
    ]
    for family in FAMILIES:
        pick = energy[families == family]
        medians = [
            float(np.median(rng.choice(pick, pick.size))) for _ in range(BOOTSTRAP)
        ]
        lo, hi = np.percentile(medians, [2.5, 97.5])
        out.append(
            f"| {family} | {pick.size} | {np.median(pick):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {lo > 0} |"
        )
    out += [
        "",
        "## Config x metric matrix (fresh; truth_kl and sampled_log_loss Holm, others BH)",
        "",
        family_metric_tables(records, metrics),
    ]
    pinned = all_records(table, ("truth_kl", "sampled_log_loss"), PINNED)
    classify(pinned, fresh=True)
    out += [
        "",
        "## Same two confirmatory metrics against the pinned-kappa pair (descriptive)",
        "",
        family_metric_tables(pinned, ("truth_kl", "sampled_log_loss")),
    ]
    out += ["", *_supplement(x, rule["names"], skill, families, base)]
    (output_dir() / "results_fresh.json").write_text(
        json.dumps(
            {
                "records": records,
                "configs": configs,
                "E": energy.tolist(),
                "S": skill.tolist(),
                "prediction": prediction.tolist(),
                "r2": r2,
                "rho": rho,
                "lower": lower,
                "within": within,
                "h1": [h1_stat, h1_p],
                "top_rho": top_rho,
                "sign_match": sign_match,
            }
        )
    )
    return "\n".join(out)


def _supplement(
    x_fresh: Array,
    names: list[str],
    skill: Array,
    families: Array,
    base: float,
) -> list[str]:
    """Post hoc (not pre-registered): family means plus a within-family elastic net.

    Fresh configs come from the same ten families, so a model that uses family
    identity (unavailable to the leave-family-out rule) can be scored on them. Fitted
    on discovery only: discovery family mean of ``S_c`` plus an elastic net on the
    family-centred covariates, penalty by the minimum-MSE rule over random folds.
    """
    exploratory = json.loads((output_dir() / "results_exploratory.json").read_text())
    configs = exploratory["configs"]
    covariates = load_covariates("discovery")
    x_all, columns = design_matrix(covariates, configs)
    x = x_all[:, [columns.index(n) for n in names]]
    y = np.asarray(exploratory["S"], dtype=np.float64)
    labels = np.array([family_of(c) for c in configs])
    y_mean = {f: float(y[labels == f].mean()) for f in FAMILIES}
    x_mean = {f: x[labels == f].mean(0) for f in FAMILIES}
    residual = y - np.array([y_mean[f] for f in labels])
    centred = x - np.stack([x_mean[f] for f in labels])
    folds = np.random.default_rng(5).integers(0, 4, len(y)).astype(np.int64)
    lam = select_lambda(centred, residual, folds, rule="min")
    fit = fit_elastic_net(centred, residual, np.array([lam]))[0]
    family_part = np.array([y_mean[f] for f in families])
    shifted = x_fresh - np.stack([x_mean[f] for f in families])
    full = family_part + fit.predict(shifted)
    total = float(((skill - base) ** 2).sum())

    def r2(prediction: Array) -> float:
        return 1.0 - float(((skill - prediction) ** 2).sum()) / total

    return [
        "## Post hoc supplement (NOT pre-registered): family means plus within-family covariates",
        "",
        "Fitted on discovery only; scored on fresh configs against the same baseline "
        "(mean discovery S_c). The pre-registered leave-family-out rule cannot use family "
        "identity; fresh configs share the ten families, so this is a different, easier question.",
        "",
        f"- discovery family means alone: R^2_fresh = {r2(family_part):+.3f}; "
        f"Spearman with observed S_c = {_spearman(family_part, skill):+.3f}",
        f"- family means + within-family elastic net (penalty {lam:.4g}, "
        f"{int(np.sum(fit.coefficients != 0))} nonzero coefficients): "
        f"R^2_fresh = {r2(full):+.3f}; Spearman = {_spearman(full, skill):+.3f}",
        "- nonzero within-family coefficients (per SD of the family-centred covariate): "
        + (
            ", ".join(
                f"{names[j]} {fit.coefficients[j]:+.4f}"
                for j in np.flatnonzero(fit.coefficients)
            )
            or "none"
        ),
    ]


def _within_family_spearman(a: Array, b: Array, families: Array) -> float:
    ra, rb = rankdata(a), rankdata(b)
    for f in np.unique(families):
        pick = families == f
        ra[pick] -= ra[pick].mean()
        rb[pick] -= rb[pick].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def _h1(
    covariate: Array, outcome: Array, families: Array, rng: np.random.Generator
) -> tuple[float, float]:
    observed = _within_family_spearman(covariate, outcome, families)
    count = 0
    for _ in range(BOOTSTRAP):
        shuffled = outcome.copy()
        for f in np.unique(families):
            pick = np.flatnonzero(families == f)
            shuffled[pick] = outcome[rng.permutation(pick)]
        count += _within_family_spearman(covariate, shuffled, families) >= observed
    return observed, float((1 + count) / (1 + BOOTSTRAP))


__all__ = [
    "METRICS",
    "all_records",
    "classify",
    "config_metric_record",
    "family_metric_tables",
    "lambda_grid",
    "main",
]

if __name__ == "__main__":
    main()
