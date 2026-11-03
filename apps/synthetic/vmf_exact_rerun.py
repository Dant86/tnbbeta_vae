"""Reruns the metric-first graph search's vMF arms with the library vs exact sampler.

Usage:
    uv run python -m apps.synthetic.vmf_exact_rerun manifest
    uv run python -m apps.synthetic.vmf_exact_rerun run --shard 0 --num-shards 96 --device cpu
    uv run python -m apps.synthetic.vmf_exact_rerun analyze

The subset is every fifth config of every family of the ``fresh`` stage (configs
``-00, -05, -10, -15``: 40 configs) with all 16 draws. Each ``vmf_init`` / ``vmf_pin``
item of the ``fresh`` manifest (same seed, same learning rate, same graph and edge split)
is rerun twice: with the library sampler (``lib``, a control that shows what a rerun
alone changes) and with ``VonMisesFisher(exact_envelope=True)`` (``exact``). TNBBeta and
PowerSpherical are not rerun; their ``fresh`` results are the comparison.
Everything lands under ``$TNBBETA_CHECKPOINT_DIR/graph_search/vmf_exact/``.
"""
# ruff: noqa: E501  (long markdown report strings)

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import wilcoxon

from apps.eval.graph_search_stats import holm
from apps.synthetic.graph_search import run_item, stage_dir
from tnbbeta_vae.distributions import VonMisesFisher
from tnbbeta_vae.models import heads
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import select_device

SUBSET_SUFFIXES = ("00", "05", "10", "15")
CONDITIONS = ("vmf_init", "vmf_pin")
SAMPLERS = ("lib", "exact")
# metric -> +1 if lower is better (advantage = comparator - TNBBeta), -1 if higher is better.
METRICS = {
    "truth_kl": 1,
    "sampled_log_loss": 1,
    "truth_kl_w": 1,
    "ece": 1,
    "cal_slope_dev": 1,
    "centre_auc": -1,
    "sampled_auc": -1,
    "rank_corr": -1,
}
ALPHA = 0.05


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``manifest``, ``run`` or ``analyze``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    run = sub.add_parser("run")
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--device", default=None)
    sub.add_parser("analyze")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        write_manifest()
    elif args.command == "run":
        run_shard(args.shard, args.num_shards, args.device)
    else:
        print(analyze())


def rerun_dir() -> Path:
    """Returns (without creating) the output directory."""
    return checkpoint_dir() / "graph_search" / "vmf_exact"


def build_items(fresh_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selects the subset of the ``fresh`` manifest and duplicates it per sampler."""
    items = []
    for item in fresh_items:
        if item["condition"] not in CONDITIONS:
            continue
        if item["config_id"].rsplit("-", 1)[1] not in SUBSET_SUFFIXES:
            continue
        for sampler in SAMPLERS:
            run_id = f"{item['run_id']}__{sampler}"
            items.append(
                {
                    **item,
                    "run_id": run_id,
                    "sampler": sampler,
                    "checkpoint": f"graph_search/vmf_exact/checkpoints/{run_id}.pt",
                }
            )
    items.sort(key=lambda item: (-item["num_nodes"], item["run_id"]))
    return items


def write_manifest() -> Path:
    """Writes the manifest and the subset's configs."""
    fresh = stage_dir("fresh")
    items = build_items(
        [
            json.loads(x)
            for x in (fresh / "manifest.jsonl").read_text().splitlines()
            if x
        ]
    )
    configs = json.loads((fresh / "configs.json").read_text())
    keep = {item["config_id"] for item in items}
    directory = rerun_dir()
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    (directory / "configs.json").write_text(
        json.dumps([c for c in configs if c["config_id"] in keep], indent=2)
    )
    path = directory / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    print(f"Wrote {len(items)} runs ({len(keep)} configs) to {path}")
    return path


class _ExactVonMisesFisher(VonMisesFisher):
    """The library vMF with ``exact_envelope`` switched on."""

    def __init__(self, loc: Any, scale: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loc, scale, *args, exact_envelope=True, **kwargs)


class _LibraryVonMisesFisher(VonMisesFisher):
    """The library vMF with the original Taylor-switch sampler (pre-fix default)."""

    def __init__(self, loc: Any, scale: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loc, scale, *args, exact_envelope=False, **kwargs)


def run_shard(shard: int, num_shards: int, device_name: str | None) -> None:
    """Runs this shard's share of the manifest, skipping finished runs."""
    directory = rerun_dir()
    configs = {
        c["config_id"]: c for c in json.loads((directory / "configs.json").read_text())
    }
    lines = (directory / "manifest.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines if line][shard::num_shards]
    device = select_device(device_name)
    failures = 0
    for item in items:
        output = directory / "runs" / f"{item['run_id']}.json"
        if output.exists():
            continue
        heads.VonMisesFisher = (  # pyright: ignore[reportAttributeAccessIssue]
            _ExactVonMisesFisher
            if item["sampler"] == "exact"
            else _LibraryVonMisesFisher
        )
        try:
            result = run_item(item, configs[item["config_id"]], device)
        except Exception as error:  # keep the shard going; report at the end
            failures += 1
            print(f"FAILED {item['run_id']}: {error!r}", flush=True)
            continue
        output.write_text(json.dumps(result))
        print(f"done {item['run_id']} in {result['seconds']:.0f}s", flush=True)
    heads.VonMisesFisher = VonMisesFisher  # pyright: ignore[reportAttributeAccessIssue]
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


def analyze() -> str:
    """Paired tests of TNBBeta against the library and exact-sampler vMF arms."""
    rerun = defaultdict(dict)  # (config, condition, sampler) -> draw -> metrics
    for path in (rerun_dir() / "runs").glob("*.json"):
        r = json.loads(path.read_text())
        rerun[(r["config_id"], r["condition"], r["sampler"])][r["draw"]] = r
    fresh_runs = stage_dir("fresh") / "runs"
    configs = sorted({key[0] for key in rerun})
    tnb = {}
    original = {}
    for config in configs:
        for draw in range(16):
            tnb[(config, draw)] = _load(fresh_runs, config, draw, "tnbbeta")
            for cond in CONDITIONS:
                original[(config, cond, draw)] = _load(fresh_runs, config, draw, cond)
    lines = [
        f"Configs: {len(configs)}; runs finished: {sum(len(v) for v in rerun.values())}.\n"
    ]

    # Reproducibility of the library arm against the original fresh run.
    diffs = []
    for (config, cond, sampler), draws in rerun.items():
        if sampler != "lib":
            continue
        for draw, r in draws.items():
            diffs.append(
                abs(
                    r["metrics"]["truth_kl"]
                    - original[(config, cond, draw)]["metrics"]["truth_kl"]
                )
            )
    lines.append(
        f"Library-sampler rerun vs the original fresh run (truth_kl, same run): median |diff| {np.median(diffs):.2e}, "
        f"max {np.max(diffs):.2e}, exactly equal in {np.mean(np.asarray(diffs) == 0):.0%} of {len(diffs)} runs.\n"
    )

    def config_means(arm: str, cond: str, metric: str) -> np.ndarray:
        values = []
        for config in configs:
            if arm == "orig":
                v = [original[(config, cond, d)]["metrics"][metric] for d in range(16)]
            else:
                v = [
                    rerun[(config, cond, arm)][d]["metrics"][metric]
                    for d in rerun[(config, cond, arm)]
                ]
            values.append(np.mean(v))
        return np.asarray(values)

    def tnb_means(metric: str) -> np.ndarray:
        return np.asarray(
            [
                np.mean([tnb[(c, d)]["metrics"][metric] for d in range(16)])
                for c in configs
            ]
        )

    rows = []
    for cond in CONDITIONS:
        for sampler in SAMPLERS:
            for metric, sign in METRICS.items():
                adv = sign * (config_means(sampler, cond, metric) - tnb_means(metric))
                rows.append((cond, sampler, metric, adv))
    p_values = [_wilcoxon(adv, "greater") for *_, adv in rows]
    holm_p = holm(p_values)
    lines.append(
        "### TNBBeta advantage over vMF (positive = TNBBeta better), config-level means over 16 draws\n"
    )
    lines.append(
        "One-sided Wilcoxon signed-rank over configs (H1: TNBBeta better), Holm over all rows. dz = mean/sd of the 40 per-config advantages; better/worse = configs with advantage > / < 0.\n"
    )
    lines.append(
        "| comparator | sampler | metric | median adv | mean adv | dz | better/worse | p | Holm p | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (cond, sampler, metric, adv), p, hp in zip(rows, p_values, holm_p, strict=True):
        sd = adv.std(ddof=1)
        lines.append(
            f"| {cond} | {sampler} | {metric} | {np.median(adv):+.4f} | {adv.mean():+.4f} | {adv.mean() / sd if sd > 0 else 0:+.2f} | "
            f"{int((adv > 0).sum())}/{int((adv < 0).sum())} | {p:.4f} | {hp:.4f} | {'TNBBeta better' if hp < ALPHA else 'null'} |"
        )
    # exact vs lib
    sub = []
    for cond in CONDITIONS:
        for metric, sign in METRICS.items():
            delta = sign * (
                config_means("lib", cond, metric) - config_means("exact", cond, metric)
            )
            sub.append((cond, metric, delta))
    two = [_wilcoxon(d, "two-sided") for *_, d in sub]
    two_holm = holm(two)
    lines.append(
        "\n### Effect of the sampler fix on vMF alone (positive = vMF better with the exact sampler)\n"
    )
    lines.append(
        "Two-sided Wilcoxon over configs, Holm over these rows. Per-config means over 16 draws, vMF(lib) vs vMF(exact), same seeds.\n"
    )
    lines.append(
        "| arm | metric | median change | mean change | dz | exact better/worse | p | Holm p |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for (cond, metric, d), p, hp in zip(sub, two, two_holm, strict=True):
        sd = d.std(ddof=1)
        lines.append(
            f"| {cond} | {metric} | {np.median(d):+.4f} | {d.mean():+.4f} | {d.mean() / sd if sd > 0 else 0:+.2f} | {int((d > 0).sum())}/{int((d < 0).sum())} | {p:.4f} | {hp:.4f} |"
        )
    # change in the advantage, per family, truth_kl
    lines.append(
        "\n### Per family, truth_kl advantage of TNBBeta over vMF (median over the family's 4 configs)\n"
    )
    lines.append(
        "| family | vmf_init lib | vmf_init exact | vmf_pin lib | vmf_pin exact |"
    )
    lines.append("|---|---|---|---|---|")
    fam = {c: c.rsplit("-", 1)[0] for c in configs}
    cols = {
        (cond, s): config_means(s, cond, "truth_kl") - tnb_means("truth_kl")
        for cond in CONDITIONS
        for s in SAMPLERS
    }
    for family in dict.fromkeys(fam.values()):
        idx = [i for i, c in enumerate(configs) if fam[c] == family]
        lines.append(
            f"| {family} | "
            + " | ".join(
                f"{np.median(cols[(cond, s)][idx]):+.4f}"
                for cond in CONDITIONS
                for s in SAMPLERS
            )
            + " |"
        )
    # kappa learned
    lines.append(
        "\n### Learned posterior concentration (mean log kappa, median over configs)\n"
    )
    for cond in CONDITIONS:
        for s in SAMPLERS:
            lines.append(
                f"- {cond} {s}: median exp(mean log kappa) = {np.exp(np.median(config_means(s, cond, 'kappa_log_mean'))):.2f}"
            )
    return "\n".join(lines)


def _load(directory: Path, config: str, draw: int, condition: str) -> dict[str, Any]:
    matches = list(directory.glob(f"{config}_d{draw:03d}_{condition}_lr*.json"))
    assert len(matches) == 1, (config, draw, condition, matches)
    return json.loads(matches[0].read_text())


def _wilcoxon(values: np.ndarray, alternative: str) -> float:
    if np.all(values == 0):
        return 1.0
    return float(cast("Any", wilcoxon(values, alternative=alternative)).pvalue)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    main()
