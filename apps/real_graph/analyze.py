"""Analysis of the real-graph revision, as pre-registered.

Usage:
    uv run python -m apps.real_graph.analyze [--stages main thin community pin]

Reads ``$TNBBETA_CHECKPOINT_DIR/real_graph/<stage>/`` and writes ``analysis.json`` (every
number the figures and the results note use) and ``analysis.md`` (the same as tables)
under ``$TNBBETA_CHECKPOINT_DIR/real_graph/``. Pre-registration:
``writeup/results/real_graph_revision_preregistration_2026-11-03.md``.

Sign convention: ``advantage > 0`` means TNBBeta is better (comparator minus TNBBeta for
log loss and ECE, TNBBeta minus comparator for AUROC and AP).
"""
# ruff: noqa: E501  (long report strings)

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from apps.real_graph.run import (
    DATASETS,
    FAMILIES,
    FEATURES,
    FINAL_SEEDS,
    PIN_FRACTIONS,
    root_dir,
    stage_dir,
)
from apps.real_graph.stats import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    holm,
    sign_flip_p,
    within_block_permutation_p,
)
from tnbbeta_vae.models.losses.ranking import roc_auc
from tnbbeta_vae.models.real_graph_eval import BUCKET_LABELS

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

METRICS = ("log_loss", "ece", "sampled_auc", "centre_auc", "ap", "centre_log_loss")
LOWER_IS_BETTER = ("log_loss", "ece", "centre_log_loss")
PRIMARY_METRIC = "log_loss"
SECONDARY_METRICS = ("sampled_auc", "centre_auc", "ap", "ece")
COMPARATORS = ("vmf", "power_spherical")
MARGIN = 0.05
MIN_PAIRS = 20
EDGE_BOOTSTRAP = 1000
LEVELS = (0.3, 0.5, 0.7, 1.0)
LOW_BUCKETS = (0, 1)
HIGH_BUCKETS = (2, 3, 4)

Cell = tuple[str, str]


def main(argv: list[str] | None = None) -> None:
    """Runs the analysis of the requested stages and writes the report files.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["main", "thin", "community", "pin"],
        choices=["main", "thin", "community", "pin"],
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=100_000)
    args = parser.parse_args(argv)
    results: dict[str, Any] = {
        "tuned": json.loads((root_dir() / "tuned.json").read_text())
    }
    report: list[str] = []
    runs = {stage: load_runs(stage, args.workers) for stage in args.stages}
    if "main" in runs:
        main_runs = runs["main"]
        results["main"] = analyze_main(main_runs)
        results["buckets"] = analyze_buckets(main_runs, args.workers)
        results["shape"] = analyze_shape(main_runs, runs.get("pin", []))
        report += render_main(results["main"]) + render_buckets(results["buckets"])
        report += render_shape(results["shape"])
        if "thin" in runs:
            results["thin"] = analyze_thinning(
                main_runs, runs["thin"], args.permutations
            )
            report += render_thinning(results["thin"])
        if "community" in runs:
            results["community"] = analyze_community(runs["community"])
            report += render_community(results["community"])
    root_dir().joinpath("analysis.json").write_text(json.dumps(results, indent=1))
    root_dir().joinpath("analysis.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {root_dir() / 'analysis.json'} and analysis.md")


def load_runs(stage: str, workers: int = 16) -> list[dict[str, Any]]:
    """Reads every finished run record of ``stage`` (a thread pool: slow network disk).

    Args:
        stage: Stage name.
        workers: Thread count.

    Returns:
        The run records that finished with a trained model.
    """
    paths = sorted((stage_dir(stage) / "runs").glob("*.json"))
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(lambda path: json.loads(path.read_text()), paths))
    return [record for record in records if "test" in record]


def metric_arrays(
    runs: list[dict[str, Any]], seeds: tuple[int, ...] = FINAL_SEEDS
) -> dict[tuple[Any, ...], dict[str, Array]]:
    """Groups runs by (dataset, features, family, keep, community, pin) into seed arrays.

    Args:
        runs: Run records.
        seeds: The seeds to lay out (missing ones are ``nan``).

    Returns:
        For each group, each test metric as an array over ``seeds``.
    """
    out: dict[tuple[Any, ...], dict[str, Array]] = {}
    for run in runs:
        job = run["job"]
        key = (
            job["dataset"],
            job["features"],
            job["family"],
            job.get("keep"),
            job.get("community"),
            job.get("pin"),
        )
        entry = out.setdefault(key, {m: np.full(len(seeds), np.nan) for m in METRICS})
        if job["seed"] in seeds:
            for metric in METRICS:
                entry[metric][seeds.index(job["seed"])] = run["test"][metric]
    return out


def orient(metric: str, comparator: Array, tnbbeta: Array) -> Array:
    """Per-seed advantage: positive when TNBBeta is better."""
    return comparator - tnbbeta if metric in LOWER_IS_BETTER else tnbbeta - comparator


def primary_comparator(
    metric: str, means: dict[str, float], candidates: tuple[str, ...] = COMPARATORS
) -> str:
    """The better of the candidates by cell mean (never looking at TNBBeta)."""
    available = [c for c in candidates if np.isfinite(means.get(c, np.nan))]
    sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
    return min(available, key=lambda c: sign * means[c])


def paired_summary(advantage: Array, *, margin: float | None = None) -> dict[str, Any]:
    """Effect size, interval and exact p-values of paired per-seed advantages.

    Args:
        advantage: Per-seed advantages (``nan`` seeds dropped).
        margin: If given, adds the one-sided win and loss tests against this margin.

    Returns:
        ``mean``, ``ci`` (95% percentile bootstrap), ``sd``, ``n``, ``p`` (two-sided
        exact sign flip) and, with a margin, ``p_win`` and ``p_loss``.
    """
    values = advantage[np.isfinite(advantage)]
    if values.size < 3:
        return {"mean": float("nan"), "n": int(values.size)}
    low, high = bootstrap_mean_ci(values)
    out: dict[str, Any] = {
        "mean": float(values.mean()),
        "ci": [low, high],
        "sd": float(values.std(ddof=1)),
        "n": int(values.size),
        "p": sign_flip_p(values),
    }
    if margin is not None:
        out["p_win"] = sign_flip_p(values, margin=margin, alternative="greater")
        out["p_loss"] = sign_flip_p(-values, margin=margin, alternative="greater")
    return out


def analyze_main(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 1: per-cell means and the pre-registered P1 and P2 tests."""
    arrays = metric_arrays(runs)
    cells: dict[str, Any] = {}
    for dataset in DATASETS:
        for features in FEATURES:
            cell = f"{dataset}|{features}"
            families = {
                f: arrays.get((dataset, features, f, None, None, None))
                for f in FAMILIES
            }
            if any(v is None for v in families.values()):
                continue
            stats: dict[str, Any] = {}
            for family, values in families.items():
                values = cast("dict[str, Array]", values)
                stats[family] = {
                    metric: {
                        "mean": float(np.nanmean(values[metric])),
                        "ci": list(
                            bootstrap_mean_ci(
                                values[metric][np.isfinite(values[metric])]
                            )
                        ),
                        "n": int(np.isfinite(values[metric]).sum()),
                        "values": values[metric].tolist(),
                    }
                    for metric in METRICS
                }
            comparisons: dict[str, Any] = {}
            for metric in METRICS:
                means = {f: stats[f][metric]["mean"] for f in FAMILIES}
                chosen = primary_comparator(metric, means)
                tnb = cast("dict[str, Array]", families["tnbbeta"])[metric]
                comparisons[metric] = {
                    "primary_comparator": chosen,
                    "primary": paired_summary(
                        orient(
                            metric,
                            cast("dict[str, Array]", families[chosen])[metric],
                            tnb,
                        ),
                        margin=MARGIN if metric == PRIMARY_METRIC else None,
                    ),
                    "by_comparator": {
                        c: paired_summary(
                            orient(
                                metric,
                                cast("dict[str, Array]", families[c])[metric],
                                tnb,
                            )
                        )
                        for c in (*COMPARATORS, "gaussian")
                    },
                }
            cells[cell] = {"families": stats, "comparisons": comparisons}
    _adjust_main(cells)
    return {"cells": cells, "margin": MARGIN}


def balanced_mean(values: Array, labels: Array, mask: Mask) -> float:
    """Label-balanced mean over a mask: half the positive mean, half the negative mean.

    ``nan`` when either label has fewer than :data:`MIN_PAIRS` pairs in the mask.
    """
    positive = mask & (labels > 0.5)
    negative = mask & (labels <= 0.5)
    if positive.sum() < MIN_PAIRS or negative.sum() < MIN_PAIRS:
        return float("nan")
    return 0.5 * float(values[positive].mean() + values[negative].mean())


def group_auc(prob: Array, labels: Array, mask: Mask) -> float:
    """AUROC of the pairs in a mask (``nan`` below :data:`MIN_PAIRS` of either label)."""
    positive = mask & (labels > 0.5)
    negative = mask & (labels <= 0.5)
    if positive.sum() < MIN_PAIRS or negative.sum() < MIN_PAIRS:
        return float("nan")
    return roc_auc(prob[positive].astype(np.float64), prob[negative].astype(np.float64))


def analyze_buckets(runs: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """Stage 2: per-bucket label-balanced advantages and the degree-gradient contrast."""
    arrays = metric_arrays(runs)
    index = {
        (
            r["job"]["dataset"],
            r["job"]["features"],
            r["job"]["family"],
            r["job"]["seed"],
        ): r
        for r in runs
    }

    def pair_path(dataset: str, features: str, family: str, seed: int) -> Path:
        run_id = index[(dataset, features, family, seed)]["job"]["run_id"]
        return stage_dir("main") / "pairs" / f"{run_id}.npz"

    cells: dict[str, Any] = {}
    work = []
    for dataset in DATASETS:
        for features in FEATURES:
            if (dataset, features, "tnbbeta", 0) not in index:
                continue
            work.append((dataset, features))
    for dataset, features in work:
        key = (dataset, features)
        seeds = [
            s
            for s in FINAL_SEEDS
            if all((dataset, features, f, s) in index for f in FAMILIES)
        ]
        means = {
            f: float(
                np.nanmean(arrays[(dataset, features, f, None, None, None)]["log_loss"])
            )
            for f in COMPARATORS
        }
        auc_means = {
            f: float(
                np.nanmean(
                    arrays[(dataset, features, f, None, None, None)]["sampled_auc"]
                )
            )
            for f in COMPARATORS
        }
        ll_comp = primary_comparator("log_loss", means)
        auc_comp = primary_comparator("sampled_auc", auc_means)
        needed = {"tnbbeta", ll_comp, auc_comp}
        with ThreadPoolExecutor(workers) as pool:
            loaded = {
                (f, s): pool.submit(_load_pairs, pair_path(dataset, features, f, s))
                for f in needed
                for s in seeds
            }
            data = {k: v.result() for k, v in loaded.items()}
        node_loaded = {}
        for s in seeds:
            run_id = index[(dataset, features, "tnbbeta", s)]["job"]["run_id"]
            with np.load(stage_dir("main") / "nodes" / f"{run_id}.npz") as nodes:
                node_loaded[s] = {k: nodes[k] for k in nodes.files}
        records = []
        for s in seeds:
            tnb, comp_ll, comp_auc = (
                data[("tnbbeta", s)],
                data[(ll_comp, s)],
                data[(auc_comp, s)],
            )
            labels = tnb["labels"].astype(np.float64)
            assert np.array_equal(labels, comp_ll["labels"])
            advantage = _pair_losses(comp_ll["sampled"], labels) - _pair_losses(
                tnb["sampled"], labels
            )
            records.append(
                {
                    "bucket": tnb["bucket"],
                    "labels": labels,
                    "advantage": advantage,
                    "tnb": tnb["sampled"].astype(np.float64),
                    "comp": comp_auc["sampled"].astype(np.float64),
                    "pairs": tnb["pairs"],
                    "degree": node_loaded[s]["degree"],
                    "m": node_loaded[s].get("m"),
                    "smm": node_loaded[s]["second_mode_mass"],
                }
            )
        buckets: list[dict[str, Any]] = []
        for b, label in enumerate(BUCKET_LABELS):
            ll_vals, auc_vals, n_pos, n_neg, frac_m, smm_vals = [], [], [], [], [], []
            for rec in records:
                mask = rec["bucket"] == b
                ll_vals.append(balanced_mean(rec["advantage"], rec["labels"], mask))
                auc_vals.append(
                    group_auc(rec["tnb"], rec["labels"], mask)
                    - group_auc(rec["comp"], rec["labels"], mask)
                )
                n_pos.append(int((mask & (rec["labels"] > 0.5)).sum()))
                n_neg.append(int((mask & (rec["labels"] <= 0.5)).sum()))
                lower = np.where(
                    rec["degree"][rec["pairs"][0]] <= rec["degree"][rec["pairs"][1]],
                    rec["pairs"][0],
                    rec["pairs"][1],
                )
                if mask.any():
                    smm_vals.append(float(rec["smm"][lower[mask]].mean()))
                    if rec["m"] is not None:
                        frac_m.append(float((rec["m"][lower[mask]] < 0).mean()))
            ll_arr = np.array(ll_vals)
            auc_arr = np.array(auc_vals)
            entry: dict[str, Any] = {
                "label": label,
                "pairs_pos_mean": float(np.mean(n_pos)),
                "pairs_neg_mean": float(np.mean(n_neg)),
                "log_loss": _bucket_summary(ll_arr),
                "auc": _bucket_summary(auc_arr),
                "tnb_frac_m_negative": float(np.mean(frac_m)) if frac_m else None,
                "tnb_second_mode_mass": float(np.mean(smm_vals)) if smm_vals else None,
            }
            entry["log_loss_edge_ci"] = _edge_bootstrap(records, b)
            buckets.append(entry)
        gradient: dict[str, Any] = {}
        for name, getter in (
            ("log_loss", lambda r, m: balanced_mean(r["advantage"], r["labels"], m)),
            (
                "auc",
                lambda r, m: (
                    group_auc(r["tnb"], r["labels"], m)
                    - group_auc(r["comp"], r["labels"], m)
                ),
            ),
        ):
            diffs = np.array(
                [
                    getter(r, np.isin(r["bucket"], HIGH_BUCKETS))
                    - getter(r, np.isin(r["bucket"], LOW_BUCKETS))
                    for r in records
                ]
            )
            gradient[name] = paired_summary(diffs)
        cells["|".join(key)] = {
            "seeds": len(seeds),
            "log_loss_comparator": ll_comp,
            "auc_comparator": auc_comp,
            "buckets": buckets,
            "gradient": gradient,
        }
    for name in ("log_loss", "auc"):
        cell_names = sorted(cells)
        p_values = [cells[c]["gradient"][name].get("p", 1.0) for c in cell_names]
        for cell, q in zip(cell_names, benjamini_hochberg(p_values), strict=True):
            cells[cell]["gradient"][name]["q_bh"] = q
    return {"cells": cells}


def analyze_shape(
    main_runs: list[dict[str, Any]], pin_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stage 4 (descriptive): free-TNBBeta shape statistics and the m-pinning curves."""
    shape: dict[str, Any] = {"free": {}, "pin": {}}
    for run in main_runs:
        job = run["job"]
        if job["family"] != "tnbbeta":
            continue
        cell = shape["free"].setdefault(f"{job['dataset']}|{job['features']}", [])
        cell.append({"seed": job["seed"], **run["shape"], "diverged": run["diverged"]})
    summary_keys = (
        "second_mode_mass",
        "frac_nodes_second_mode",
        "m_mean",
        "frac_m_negative",
        "m_degree_spearman",
        "second_mode_degree_spearman",
        "p_mean",
        "q_mean",
        "epsilon_mean",
    )
    shape["free"] = {
        cell: {k: _nanmean([r.get(k, np.nan) for r in rows]) for k in summary_keys}
        | {"n": len(rows)}
        for cell, rows in shape["free"].items()
    }
    arrays = metric_arrays(pin_runs)
    by_cell: dict[str, dict[str, Any]] = {}
    for run in pin_runs:
        job = run["job"]
        cell = by_cell.setdefault(f"{job['dataset']}|{job['features']}", {})
        row = cell.setdefault(f"{job['pin']:g}", {"smm": [], "m": []})
        row["smm"].append(run["shape"]["second_mode_mass"])
        row["m"].append(run["shape"].get("m_mean", np.nan))
    for (dataset, features, _fam, _keep, _comm, pin), values in arrays.items():
        if pin is None:
            continue
        row = by_cell[f"{dataset}|{features}"][f"{pin:g}"]
        for metric in ("log_loss", "ece", "sampled_auc", "centre_auc"):
            row[metric] = float(np.nanmean(values[metric]))
            row[f"{metric}_ci"] = list(
                bootstrap_mean_ci(values[metric][np.isfinite(values[metric])])
            )
        row["second_mode_mass"] = float(np.mean(row.pop("smm")))
        row["m_mean"] = float(np.nanmean(row.pop("m")))
        row["n"] = int(np.isfinite(values["log_loss"]).sum())
    shape["pin"] = by_cell
    return shape


def analyze_thinning(
    main_runs: list[dict[str, Any]], thin_runs: list[dict[str, Any]], permutations: int
) -> dict[str, Any]:
    """Stage 3: advantage versus realised density under edge thinning."""
    arrays = metric_arrays([*main_runs, *thin_runs])
    degree: dict[tuple[str, float, int], dict[str, float]] = {}
    for run in [*main_runs, *thin_runs]:
        job = run["job"]
        if job["family"] == "tnbbeta":
            keep = 1.0 if job.get("keep") is None else float(job["keep"])
            degree[(job["dataset"], keep, job["seed"])] = run["graph_stats"]
    out: dict[str, Any] = {"cells": {}, "tests": {}}
    for features in FEATURES:
        blocks_adv: dict[str, list[Array]] = {
            m: [] for m in ("log_loss", "sampled_auc")
        }
        blocks_cov: list[Array] = []
        for dataset in DATASETS:
            levels = {}
            for keep in LEVELS:
                key_keep = None if keep == 1.0 else keep
                levels[keep] = {
                    f: arrays.get((dataset, features, f, key_keep, None, None))
                    for f in FAMILIES
                }
            if any(v is None for lv in levels.values() for v in lv.values()):
                continue
            cell: dict[str, Any] = {"levels": {}}
            matrix = {
                m: np.full((len(FINAL_SEEDS), len(LEVELS)), np.nan) for m in blocks_adv
            }
            covariate = np.full((len(FINAL_SEEDS), len(LEVELS)), np.nan)
            for column, keep in enumerate(LEVELS):
                lv = cast("dict[str, dict[str, Array]]", levels[keep])
                stats_rows = [degree.get((dataset, keep, s)) for s in FINAL_SEEDS]
                realised = {
                    k: _nanmean([r[k] if r else np.nan for r in stats_rows])
                    for k in (
                        "mean_degree",
                        "degree_cv",
                        "transitivity",
                        "isolated_fraction",
                    )
                }
                entry: dict[str, Any] = {"graph": realised, "metrics": {}}
                for metric in ("log_loss", "sampled_auc", "ece"):
                    means = {f: float(np.nanmean(lv[f][metric])) for f in COMPARATORS}
                    chosen = primary_comparator(metric, means)
                    advantage = orient(
                        metric, lv[chosen][metric], lv["tnbbeta"][metric]
                    )
                    entry["metrics"][metric] = {
                        "comparator": chosen,
                        "advantage": paired_summary(advantage),
                        "tnbbeta_mean": float(np.nanmean(lv["tnbbeta"][metric])),
                    }
                    if metric in matrix:
                        matrix[metric][:, column] = advantage
                for seed_index, row in enumerate(stats_rows):
                    covariate[seed_index, column] = (
                        row["mean_degree"] if row else np.nan
                    )
                cell["levels"][f"{keep:g}"] = entry
            ok = np.isfinite(covariate).all(axis=1) & np.isfinite(
                matrix["log_loss"]
            ).all(axis=1)
            cell["slopes"] = _slopes(matrix, covariate, ok)
            for metric in blocks_adv:
                blocks_adv[metric].append(matrix[metric][ok])
            blocks_cov.append(covariate[ok])
            out["cells"][f"{dataset}|{features}"] = cell
        out["tests"][features] = {}
        for metric, blocks in blocks_adv.items():
            if not blocks:
                continue
            observed, p, per_block = within_block_permutation_p(
                blocks, blocks_cov, permutations=permutations
            )
            out["tests"][features][metric] = {
                "combined_spearman": observed,
                "p_one_sided": p,
                "per_dataset_spearman": per_block,
                "datasets": [d for d in DATASETS if f"{d}|{features}" in out["cells"]],
            }
    for metric in ("log_loss", "sampled_auc"):
        ps = [
            out["tests"][f][metric]["p_one_sided"]
            for f in FEATURES
            if metric in out["tests"].get(f, {})
        ]
        for feature, adjusted in zip(FEATURES, holm(ps), strict=False):
            out["tests"][feature][metric]["p_holm"] = adjusted
    return out


def analyze_community(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 3b (descriptive): advantage versus induced-subgraph density."""
    seeds = tuple(range(5))
    arrays = metric_arrays(runs, seeds)
    graphs: dict[tuple[str, str, int], dict[str, float]] = {}
    for run in runs:
        job = run["job"]
        graphs[(job["dataset"], job["features"], job["community"])] = run["graph_stats"]
    out: dict[str, Any] = {}
    for dataset, features in ((d, f) for d in ("mag_cs", "pubmed") for f in FEATURES):
        communities = sorted(
            {k[4] for k in arrays if k[0] == dataset and k[1] == features}
        )
        rows = []
        for community in communities:
            fams = {
                f: arrays.get((dataset, features, f, None, community, None))
                for f in FAMILIES
            }
            if any(v is None for v in fams.values()):
                continue
            row: dict[str, Any] = {
                "community": community,
                **graphs[(dataset, features, community)],
            }
            for metric in ("log_loss", "sampled_auc", "ece"):
                means = {
                    f: float(np.nanmean(cast("dict[str, Array]", fams[f])[metric]))
                    for f in COMPARATORS
                }
                chosen = primary_comparator(metric, means)
                advantage = orient(
                    metric,
                    cast("dict[str, Array]", fams[chosen])[metric],
                    cast("dict[str, Array]", fams["tnbbeta"])[metric],
                )
                row[f"{metric}_advantage"] = float(np.nanmean(advantage))
                row[f"{metric}_comparator"] = chosen
            rows.append(row)
        correlations: dict[str, Any] = {}
        if len(rows) >= 4:
            for metric in ("log_loss", "sampled_auc", "ece"):
                y = [r[f"{metric}_advantage"] for r in rows]
                correlations[metric] = {
                    covariate: float(
                        cast("Any", spearmanr([r[covariate] for r in rows], y))[0]
                    )
                    for covariate in (
                        "mean_degree",
                        "num_nodes",
                        "transitivity",
                        "degree_cv",
                    )
                }
        out[f"{dataset}|{features}"] = {"rows": rows, "spearman": correlations}
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Renders a markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_main(main: dict[str, Any]) -> list[str]:
    """Markdown of Stage 1."""
    lines = [
        "## Stage 1: main results (mean over 10 seeds, 95% bootstrap interval)",
        "",
    ]
    for metric in ("log_loss", "sampled_auc", "centre_auc", "ap", "ece"):
        rows = []
        for cell, content in main["cells"].items():
            row = [cell.replace("|", " / ")]
            for family in FAMILIES:
                m = content["families"][family][metric]
                row.append(f"{m['mean']:.4f} [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]")
            rows.append(row)
        lines += [f"### {metric}", ""] + md_table(["cell", *FAMILIES], rows)
    lines += ["### Paired advantages of TNBBeta over the primary comparator", ""]
    rows = []
    for cell, content in main["cells"].items():
        for metric in ("log_loss", *SECONDARY_METRICS):
            c = content["comparisons"][metric]
            p = c["primary"]
            if "ci" not in p:
                continue
            extra = p.get("label", "") if metric == "log_loss" else ""
            adjusted = (
                f"Holm {_fmt(p.get('p_holm'), 4)}"
                if metric == "log_loss"
                else f"BH {_fmt(p.get('q_bh'), 4)}"
            )
            rows.append(
                [
                    cell.replace("|", " / "),
                    metric,
                    c["primary_comparator"],
                    _interval(p, 4),
                    _fmt(p["p"], 4),
                    adjusted,
                    extra,
                ]
            )
    lines += md_table(
        [
            "cell",
            "metric",
            "comparator",
            "advantage [95% CI]",
            "p (exact)",
            "adjusted",
            "label",
        ],
        rows,
    )
    lines += [
        "### Log-loss advantage over each comparator separately (descriptive)",
        "",
    ]
    rows = []
    for cell, content in main["cells"].items():
        by = content["comparisons"]["log_loss"]["by_comparator"]
        rows.append(
            [cell.replace("|", " / ")]
            + [
                f"{_interval(by[c], 4)} (p={_fmt(by[c].get('p'), 3)})"
                for c in (*COMPARATORS, "gaussian")
            ]
        )
    lines += md_table(["cell", *COMPARATORS, "gaussian"], rows)
    return lines


def render_buckets(buckets: dict[str, Any]) -> list[str]:
    """Markdown of Stage 2."""
    lines = ["## Stage 2: degree buckets (lower endpoint's training degree)", ""]
    for cell, content in buckets["cells"].items():
        lines += [
            f"### {cell.replace('|', ' / ')} (log-loss comparator {content['log_loss_comparator']}, "
            f"AUROC comparator {content['auc_comparator']}, {content['seeds']} seeds)",
            "",
        ]
        rows = []
        for b in content["buckets"]:
            edge = b["log_loss_edge_ci"]
            rows.append(
                [
                    b["label"],
                    f"{b['pairs_pos_mean']:.0f} / {b['pairs_neg_mean']:.0f}",
                    _interval(b["log_loss"], 4),
                    "n/a" if edge is None else f"[{edge[0]:.4f}, {edge[1]:.4f}]",
                    _interval(b["auc"], 4),
                    _fmt(b["tnb_frac_m_negative"], 3),
                    _fmt(b["tnb_second_mode_mass"], 4),
                ]
            )
        lines += md_table(
            [
                "bucket",
                "pos / neg pairs per seed",
                "log-loss adv. (seed CI)",
                "edge-level CI (anti-conservative)",
                "AUROC adv. (seed CI)",
                "TNB frac m<0",
                "TNB 2nd-mode mass",
            ],
            rows,
        )
        g = content["gradient"]
        lines += [
            f"Degree-gradient contrast G (min degree >= 4 minus <= 3): log loss {_interval(g['log_loss'], 4)} "
            f"p={_fmt(g['log_loss'].get('p'), 4)} BH q={_fmt(g['log_loss'].get('q_bh'), 4)}; "
            f"AUROC {_interval(g['auc'], 4)} p={_fmt(g['auc'].get('p'), 4)} BH q={_fmt(g['auc'].get('q_bh'), 4)}",
            "",
        ]
    return lines


def render_shape(shape: dict[str, Any]) -> list[str]:
    """Markdown of Stage 4 (descriptive)."""
    lines = ["## Stage 4 (descriptive): free TNBBeta shape statistics", ""]
    keys = (
        "frac_m_negative",
        "m_mean",
        "second_mode_mass",
        "frac_nodes_second_mode",
        "m_degree_spearman",
        "second_mode_degree_spearman",
        "p_mean",
        "q_mean",
        "epsilon_mean",
    )
    rows = [
        [cell.replace("|", " / "), *[_fmt(v[k], 4) for k in keys]]
        for cell, v in shape["free"].items()
    ]
    lines += md_table(["cell", *keys], rows)
    lines += [
        "## Stage 4 (descriptive): m-pinning dose-response, f = m / ((d-1)/2)",
        "",
    ]
    for cell, content in shape["pin"].items():
        rows = []
        for fraction in PIN_FRACTIONS:
            row = content.get(f"{fraction:g}")
            if row is None:
                continue
            rows.append(
                [
                    f"{fraction:g}",
                    _fmt(row.get("m_mean"), 2),
                    _fmt(row.get("log_loss"), 4),
                    _fmt(row.get("ece"), 4),
                    _fmt(row.get("sampled_auc"), 4),
                    _fmt(row.get("second_mode_mass"), 4),
                    str(row.get("n")),
                ]
            )
        lines += [f"### {cell.replace('|', ' / ')}", ""] + md_table(
            ["f", "m", "log loss", "ECE", "sampled AUROC", "second-mode mass", "n"],
            rows,
        )
    return lines


def render_thinning(thin: dict[str, Any]) -> list[str]:
    """Markdown of Stage 3."""
    lines = ["## Stage 3: edge thinning", ""]
    for features in FEATURES:
        lines += [
            f"### Features {features}: realised graph and advantage by retained fraction",
            "",
        ]
        rows = []
        for cell, content in thin["cells"].items():
            if not cell.endswith(f"|{features}"):
                continue
            for level, entry in content["levels"].items():
                g = entry["graph"]
                ll = entry["metrics"]["log_loss"]
                auc = entry["metrics"]["sampled_auc"]
                rows.append(
                    [
                        cell.split("|")[0],
                        level,
                        f"{g['mean_degree']:.2f}",
                        f"{g['degree_cv']:.2f}",
                        f"{g['transitivity']:.3f}",
                        f"{g['isolated_fraction']:.3f}",
                        f"{_interval(ll['advantage'], 4)} ({ll['comparator']})",
                        f"{_interval(auc['advantage'], 4)} ({auc['comparator']})",
                    ]
                )
        lines += md_table(
            [
                "dataset",
                "kept",
                "mean degree",
                "degree CV",
                "transitivity",
                "isolated",
                "log-loss adv.",
                "AUROC adv.",
            ],
            rows,
        )
        for metric, test in thin["tests"].get(features, {}).items():
            lines += [
                f"Features {features}, {metric}: combined Spearman {test['combined_spearman']:.3f}, "
                f"one-sided permutation p = {test['p_one_sided']:.4f} (Holm {_fmt(test.get('p_holm'), 4)}); "
                f"per dataset {dict(zip(test['datasets'], [round(x, 3) for x in test['per_dataset_spearman']], strict=True))}",
                "",
            ]
        rows = []
        for cell, content in thin["cells"].items():
            if cell.endswith(f"|{features}"):
                for metric, s in content["slopes"].items():
                    rows.append(
                        [
                            cell.split("|")[0],
                            metric,
                            f"{s['slope_per_log_degree']:.4f} [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}]",
                            f"{s['spearman']:.3f}",
                        ]
                    )
        lines += md_table(
            [
                "dataset",
                "metric",
                "OLS slope per ln(mean degree) [seed-bootstrap CI]",
                "Spearman",
            ],
            rows,
        )
    return lines


def render_community(community: dict[str, Any]) -> list[str]:
    """Markdown of Stage 3b."""
    lines = ["## Stage 3b (descriptive): Louvain community subgraphs", ""]
    for cell, content in community.items():
        lines += [f"### {cell.replace('|', ' / ')}", ""]
        rows = [
            [
                str(r["community"]),
                f"{r['num_nodes']:.0f}",
                f"{r['mean_degree']:.2f}",
                f"{r['degree_cv']:.2f}",
                f"{r['transitivity']:.3f}",
                _fmt(r["log_loss_advantage"], 4),
                _fmt(r["sampled_auc_advantage"], 4),
                _fmt(r["ece_advantage"], 4),
            ]
            for r in content["rows"]
        ]
        lines += md_table(
            [
                "community",
                "nodes",
                "mean degree",
                "CV",
                "transitivity",
                "log-loss adv.",
                "AUROC adv.",
                "ECE adv.",
            ],
            rows,
        )
        if content["spearman"]:
            lines += [
                f"Spearman across communities: {json.dumps(content['spearman'])}",
                "",
            ]
    return lines


def _adjust_main(cells: dict[str, Any]) -> None:
    """Adds Holm (P1) and BH (P2) adjusted p-values and the win/loss labels, in place."""
    names = sorted(cells)
    primary = [cells[c]["comparisons"][PRIMARY_METRIC]["primary"] for c in names]
    if all("p" in p for p in primary):
        two_sided = holm([p["p"] for p in primary])
        one_sided = holm([p["p_win"] for p in primary] + [p["p_loss"] for p in primary])
        for index, name in enumerate(names):
            summary = cells[name]["comparisons"][PRIMARY_METRIC]["primary"]
            summary["p_holm"] = two_sided[index]
            summary["p_win_holm"] = one_sided[index]
            summary["p_loss_holm"] = one_sided[len(names) + index]
            if summary["p_win_holm"] < 0.05 and summary["mean"] > MARGIN:
                label = "confirmed win"
            elif summary["p_loss_holm"] < 0.05 and summary["mean"] < -MARGIN:
                label = "confirmed loss"
            elif summary["p_holm"] < 0.05:
                label = (
                    "significant win, below margin"
                    if summary["mean"] > 0
                    else "significant loss, below margin"
                )
            else:
                label = "not significant"
            summary["label"] = label
    tests = [(c, m) for c in names for m in SECONDARY_METRICS]
    pvalues = [cells[c]["comparisons"][m]["primary"].get("p", 1.0) for c, m in tests]
    for (cell, metric), q in zip(tests, benjamini_hochberg(pvalues), strict=True):
        cells[cell]["comparisons"][metric]["primary"]["q_bh"] = q


def _pair_losses(prob: Array, labels: Array) -> Array:
    clipped = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    return -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))


def _load_pairs(path: Path) -> dict[str, Array]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _bucket_summary(values: Array) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return {
            "mean": float(finite.mean()) if finite.size else None,
            "n": int(finite.size),
        }
    low, high = bootstrap_mean_ci(finite)
    return {
        "mean": float(finite.mean()),
        "ci": [low, high],
        "n": int(finite.size),
        "p_unadjusted": sign_flip_p(finite),
    }


def _edge_bootstrap(records: list[dict[str, Any]], bucket: int) -> list[float] | None:
    """Anti-conservative CI: resample pooled pairs of the bucket (shared endpoints ignored)."""
    positive, negative = [], []
    for rec in records:
        mask = rec["bucket"] == bucket
        positive.append(rec["advantage"][mask & (rec["labels"] > 0.5)])
        negative.append(rec["advantage"][mask & (rec["labels"] <= 0.5)])
    pos, neg = np.concatenate(positive), np.concatenate(negative)
    if pos.size < MIN_PAIRS or neg.size < MIN_PAIRS:
        return None
    rng = np.random.default_rng(20261104)
    draws = [
        0.5
        * (
            pos[rng.integers(pos.size, size=pos.size)].mean()
            + neg[rng.integers(neg.size, size=neg.size)].mean()
        )
        for _ in range(EDGE_BOOTSTRAP)
    ]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _slopes(matrix: dict[str, Array], covariate: Array, ok: Mask) -> dict[str, Any]:
    """OLS slope of advantage on log realised mean degree per metric, seed-bootstrap CI."""
    slopes: dict[str, Any] = {}
    log_degree = np.log(np.maximum(covariate[ok], 1e-3))
    for metric, values in matrix.items():
        advantage = values[ok]

        def slope(rows: npt.NDArray[np.intp], adv: Array = advantage) -> float:
            x = log_degree[rows].ravel()
            y = adv[rows].ravel()
            return float(np.polyfit(x, y, 1)[0])

        if advantage.shape[0] < 3:
            continue
        rng = np.random.default_rng(20261104)
        boots = [
            slope(rng.integers(advantage.shape[0], size=advantage.shape[0]))
            for _ in range(2000)
        ]
        slopes[metric] = {
            "slope_per_log_degree": slope(np.arange(advantage.shape[0])),
            "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "spearman": float(
                cast("Any", spearmanr(log_degree.ravel(), advantage.ravel()))[0]
            ),
        }
    return slopes


def _fmt(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _interval(summary: dict[str, Any], digits: int = 3) -> str:
    if "ci" not in summary:
        return "n/a"
    return (
        f"{_fmt(summary['mean'], digits)} "
        f"[{_fmt(summary['ci'][0], digits)}, {_fmt(summary['ci'][1], digits)}]"
    )


if __name__ == "__main__":
    main()
