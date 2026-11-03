"""Tables for ``writeup/results/ring_condition_b_unpacked_2026-11-01.md``.

Two subcommands::

    uv run python -m apps.synthetic.ring_condition_b_summary controls \
        --dirs ring_recovery/l23_d3 ring_recovery/b_controls/n32 ...
    uv run python -m apps.synthetic.ring_condition_b_summary diag --json diag_l23.json

``controls`` aggregates stage-3 JSONs (one ``stage3`` directory per tag) per
(condition, variant): ELBO gap, validity flags and training-loss shape.
``diag`` aggregates ``ring_condition_b_diagnostics`` output per (condition, variant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_EPS_FLOOR = 0.3
_INVALID_NOTE = "eps_median < 0.3 (estimator-exploit regime, excluded)"


def main(argv: list[str] | None = None) -> None:
    """Prints markdown tables.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls")
    c.add_argument("--dirs", nargs="+", type=Path, required=True)
    d = sub.add_parser("diag")
    d.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "controls":
        for directory in args.dirs:
            _controls(directory)
    else:
        _diag(json.loads(args.json.read_text()))


def _load(directory: Path) -> list[dict[str, Any]]:
    runs = [
        json.loads(p.read_text()) for p in sorted((directory / "stage3").glob("*.json"))
    ]
    for r in runs:
        r["_valid"] = r.get("epsilon_median", 99.0) >= _EPS_FLOOR
    return runs


def _ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values)
    sd = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    return f"{arr.mean():.3f} +- {sd:.3f}"


def _controls(directory: Path) -> None:
    runs = _load(directory)
    print(f"\n### {directory}\n")
    print(
        "| cond | variant | n (valid) | gap or KL | metric | ELBO | sigma | "
        "final loss | loss drop ep(last 20%) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["condition"], r["variant"]) for r in runs})
    for condition, variant in keys:
        rows = [
            r for r in runs if (r["condition"], r["variant"]) == (condition, variant)
        ]
        ok = [r for r in rows if r["_valid"]]
        curve_drop = []
        for r in ok:
            curve = r.get("loss_curve_full") or r["loss_curve"]
            tail = max(2, len(curve) // 5)
            curve_drop.append(curve[-tail] - curve[-1])
        primary = [
            r["kl_mean"] if condition == "true" else r["elbo_gap_model"] for r in ok
        ]
        print(
            f"| {condition} | {variant} | {len(rows)} ({len(ok)}) | {_ms(primary)} | "
            f"{'KL(q||p)' if condition == 'true' else 'gap'} | "
            f"{_ms([r['elbo_model'] for r in ok])} | "
            f"{_ms([r['learned_sigma'] for r in ok])} | "
            f"{_ms([r['final_train_loss'] for r in ok])} | {_ms(curve_drop)} |"
        )
        for r in rows:
            if not r["_valid"]:
                print(f"|  | seed {r['seed']} | | {_INVALID_NOTE} | | | | | |")


def _diag(rows: list[dict[str, Any]]) -> None:
    keys = sorted({(r["condition"], r["variant"]) for r in rows})
    fields = [
        "recon",
        "kl_prior_mc",
        "elbo_gap",
        "decoder_misfit",
        "inference_gap",
        "learned_sigma",
        "dec_rms_dist_to_true_curve",
        "dec_wstar_tv_to_uniform",
        "dec_wstar_eta2_on_best_axis",
        "dec_abs_cos_best_axis_vs_true_axis",
    ]
    print("| cond | variant | n | " + " | ".join(fields) + " |")
    print("|---|---|---|" + "---|" * len(fields))
    for condition, variant in keys:
        group = [
            r for r in rows if (r["condition"], r["variant"]) == (condition, variant)
        ]
        cells = []
        for f in fields:
            vals = [
                r[f] for r in group if r.get(f) is not None and abs(r["elbo_gap"]) < 100
            ]
            cells.append(f"{np.mean(vals):.3f}" if vals else "")
        print(f"| {condition} | {variant} | {len(group)} | " + " | ".join(cells) + " |")


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean difference ``a - b`` and a one-sided Wilcoxon p (H1: a < b)."""
    diff = np.asarray(a) - np.asarray(b)
    result: Any = stats.wilcoxon(diff, alternative="less")
    return float(diff.mean()), float(result.pvalue)


if __name__ == "__main__":
    main()
