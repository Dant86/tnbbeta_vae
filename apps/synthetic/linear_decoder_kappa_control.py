"""Tuned-kappa control for the linear-decoder ring study (vMF / PowerSpherical caps).

Pre-registration:
``writeup/results/linear_decoder_kappa_control_preregistration_2026-11-03.md``
(arms, tuning rule, tests and interpretation rules are fixed there). Reuses the
parent study's model, exact-envelope vMF sampler, evaluation and test helpers from
:mod:`apps.synthetic.linear_decoder_ring`; decoder ``L`` only.

Usage (one run per call; the sbatch script loops over the grid)::

    uv run python -m apps.synthetic.linear_decoder_kappa_control train --dim 5 \
        --sigma 0.2 --arm vmf_pin30 --seed 0 --out-dir ring_recovery/kappa_control
    uv run python -m apps.synthetic.linear_decoder_kappa_control summarize \
        --out-dir ring_recovery/kappa_control --parent-dir ring_recovery/linear_decoder
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from apps.synthetic.linear_decoder_ring import (
    DIMS,
    HOLM_ALPHA,
    MARGIN_NATS,
    NUM_SEEDS,
    SIGMAS,
    TEST_SEED,
    ExactVonMisesFisher,
    LinearRingVAE,
    evaluate,
    wilcoxon_greater,
    wilcoxon_greater_enumerated,
)
from tnbbeta_vae.data.linear_ring import sample_observations
from tnbbeta_vae.models.heads import (
    fixed_kappa_posterior,
    power_spherical_kappa_inverse,
    vmf_kappa_inverse,
)
from tnbbeta_vae.training import select_device

FAMILIES = ("vmf", "power_spherical")
PIN_GRID = (3, 10, 30, 100, 300)
INIT_MULTIPLES = (5, 20)
FAST_HEAD_LR = 1e-2
FAST_EPOCHS = 600
BASE_EPOCHS = 300
BASE_LR = 1e-3
MIN_VALID_SEEDS = 8
TRAJECTORY_POINTS = 512
PARENT_ARMS = ("tnbbeta", "vmf", "power_spherical")
Runs = dict[tuple[int, float, str], dict[int, dict[str, Any]]]


def control_arms() -> list[str]:
    """Every new arm id, in queue priority order (pinned and high-init before fast)."""
    arms = []
    for family in FAMILIES:
        arms += [f"{family}_pin{k}" for k in PIN_GRID]
    for family in FAMILIES:
        arms += [f"{family}_k{m}d" for m in INIT_MULTIPLES]
    arms += ["vmf_H", "power_spherical_H", "tnbbeta_H"]
    return arms


def arm_settings(arm: str, dim: int) -> dict[str, Any]:
    """Parses an arm id.

    Returns:
        A dict with ``base`` (the parent arm whose head is built first), ``family``,
        ``kappa_pin`` (``None`` or float), ``kappa_init`` (``None`` or float),
        ``epochs``
        and ``head_lr``.
    """
    if arm not in control_arms():
        raise ValueError(f"unknown arm {arm!r}.")
    if arm == "tnbbeta_H":
        return _settings("tnbbeta", None, None, FAST_EPOCHS, FAST_HEAD_LR)
    family = next(f for f in FAMILIES if arm.startswith(f))
    suffix = arm[len(family) + 1 :]
    if suffix == "H":
        return _settings(family, None, float(dim), FAST_EPOCHS, FAST_HEAD_LR)
    if suffix.startswith("pin"):
        return _settings(family, float(suffix[3:]), None, BASE_EPOCHS, BASE_LR)
    multiple = int(suffix[1:-1])
    return _settings(family, None, float(multiple * dim), BASE_EPOCHS, BASE_LR)


def run_path(out_dir: Path, dim: int, sigma: float, arm: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / "L" / f"d{dim}_s{sigma:g}" / arm / f"s{seed}"


class ControlVAE(LinearRingVAE):
    """The parent's decoder-``L`` model with a kappa-controlled cap head.

    Attributes:
        kappa_pin: The constant concentration (pinned arms) or ``None``.
    """

    def __init__(self, dim: int, arm: str) -> None:
        """Builds the parent model (head last), then adjusts the head for the arm."""
        settings = arm_settings(arm, dim)
        super().__init__("L", dim, settings["base"])
        self.kappa_pin: float | None = settings["kappa_pin"]
        head = self.posterior_head
        if self.kappa_pin is not None:
            self.posterior_head = torch.nn.Linear(head.in_features, dim)
        elif settings["kappa_init"] is not None:
            inverse = (
                vmf_kappa_inverse
                if self.family == "vmf"
                else power_spherical_kappa_inverse
            )
            with torch.no_grad():
                head.bias[-1] = inverse(settings["kappa_init"])

    def posterior(self, y: Tensor):  # noqa: ANN201
        """``q(z | y)``; pinned arms use a constant kappa for every example."""
        if self.kappa_pin is None:
            return super().posterior(y)
        raw = self.posterior_head(self.encoder(y[:, None]))
        posterior = fixed_kappa_posterior(self.family, raw, self.kappa_pin)  # pyright: ignore[reportArgumentType]
        if self.family == "vmf":
            return ExactVonMisesFisher(posterior.loc, posterior.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return posterior


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train``, ``summarize`` or ``cells``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--dim", type=int, required=True)
    train.add_argument("--sigma", type=float, required=True)
    train.add_argument("--arm", choices=control_arms(), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=None, help="Default: the arm's.")
    train.add_argument("--num-train", type=int, default=20_000)
    train.add_argument("--num-test", type=int, default=2_000)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    summ.add_argument("--parent-dir", type=Path, required=True)
    sub.add_parser("cells")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        run_train(args)
    elif args.command == "cells":
        for cell in grid_cells():
            print(*cell)
    else:
        summarize(args.out_dir, args.parent_dir)


def grid_cells() -> list[tuple[int, float, str, int]]:
    """Every ``(dim, sigma, arm, seed)``: non-``H`` arms first, then ``H``."""
    arms = control_arms()
    cells: list[tuple[int, float, str, int]] = []
    for group in (
        [a for a in arms if not a.endswith("_H")],
        [a for a in arms if a.endswith("_H")],
    ):
        for seed, arm, dim, sigma in itertools.product(
            range(NUM_SEEDS), group, DIMS, SIGMAS
        ):
            cells.append((dim, sigma, arm, seed))
    return cells


def run_train(args: argparse.Namespace) -> None:
    """Trains one cell on the GPU (logging the trajectory), evaluates on the CPU."""
    settings = arm_settings(args.arm, args.dim)
    epochs = args.epochs or settings["epochs"]
    device = select_device(args.device)
    started = time.time()
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_y = sample_observations(args.dim, args.sigma, args.num_train, train_gen)
    torch.manual_seed(args.seed)
    model = ControlVAE(args.dim, args.arm)
    model.to(device)
    y_dev = train_y.to(device)
    head_ids = {id(p) for p in model.posterior_head.parameters()}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    optimiser = torch.optim.Adam(
        [
            {"params": rest, "lr": BASE_LR},
            {
                "params": list(model.posterior_head.parameters()),
                "lr": settings["head_lr"],
            },
        ]
    )
    shuffle = torch.Generator().manual_seed(200 + args.seed)
    warmup = max(1, epochs // 10)
    losses: list[float] = []
    trajectory: dict[str, list[float]] = {}
    diverged = False
    for epoch in range(epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(args.num_train, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for index in order.split(args.batch_size):
            optimiser.zero_grad()
            loss = model.negative_elbo(y_dev[index], kl_weight, args.num_elbo_samples)
            loss.backward()
            optimiser.step()
            running += loss.detach() * len(index)
        losses.append(float(running) / args.num_train)
        _log_trajectory(trajectory, model, y_dev[:TRAJECTORY_POINTS])
        if not math.isfinite(losses[-1]):
            diverged = True
            break
    model.cpu().double().eval()
    train_seconds = time.time() - started

    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    if not diverged:
        test_gen = torch.Generator().manual_seed(TEST_SEED)
        test_y = sample_observations(args.dim, args.sigma, args.num_test, test_gen)
        torch.manual_seed(31337)
        arrays, summary = evaluate(model, args.sigma, test_y.double())
        diverged = not all(
            np.isfinite(arrays[k]).all() for k in ("elbo", "iwae", "log_p_true")
        )
    summary.update(
        {
            "decoder": "L",
            "dim": args.dim,
            "sigma_true": args.sigma,
            "arm": args.arm,
            "seed": args.seed,
            "family": model.family,
            "epochs": epochs,
            "head_lr": settings["head_lr"],
            "kappa_pin": settings["kappa_pin"],
            "kappa_init": settings["kappa_init"],
            "diverged": bool(diverged),
            "epochs_run": len(losses),
            "final_train_loss": losses[-1] if losses else float("nan"),
            "loss_curve": losses,
            "trajectory": trajectory,
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(args.out_dir, args.dim, args.sigma, args.arm, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    print(
        f"L d={args.dim} sigma={args.sigma} {args.arm} s{args.seed}: gap "
        f"{summary.get('gap_elbo', float('nan')):.4f} sigma_hat "
        f"{summary.get('sigma_hat', float('nan')):.4f} diverged {diverged} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


# ---------------------------------------------------------------------------- analysis


def load_runs(out_dir: Path, arms: tuple[str, ...] | None = None) -> Runs:
    """Finished decoder-``L`` run summaries under ``out_dir``, by (d, sigma, arm)."""
    runs: Runs = {}
    for path in sorted(out_dir.glob("L/d*_s*/*/s*.json")):
        record = json.loads(path.read_text())
        if arms is not None and record["arm"] not in arms:
            continue
        key = (record["dim"], record["sigma_true"], record["arm"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def valid_gaps(
    group: dict[int, dict[str, Any]], key: str = "gap_elbo"
) -> dict[int, float]:
    """Seed -> finite value of ``key`` for non-diverged runs."""
    return {
        s: float(r[key])
        for s, r in group.items()
        if not r["diverged"] and math.isfinite(r.get(key, float("nan")))
    }


def cap_pool(runs: Runs, dim: int, sigma: float, family: str) -> list[str]:
    """Eligible cap arms of a family in a cell (parent initial-``d`` arm included)."""
    arms = [family] + [a for a in control_arms() if a.startswith(family + "_")]
    return [
        a
        for a in arms
        if len(valid_gaps(runs.get((dim, sigma, a), {}))) >= MIN_VALID_SEEDS
    ]


def best_arm(
    runs: Runs, dim: int, sigma: float, arms: list[str], exclude_seed: int | None = None
) -> str | None:
    """The arm with the smallest seed-mean gap (optionally leaving one seed out)."""
    best, best_value = None, math.inf
    for arm in arms:
        gaps = valid_gaps(runs[(dim, sigma, arm)])
        values = [v for s, v in gaps.items() if s != exclude_seed]
        if values and np.mean(values) < best_value:
            best, best_value = arm, float(np.mean(values))
    return best


def holm(tests: list[dict[str, Any]]) -> None:
    """Annotates tests in place: Holm threshold, rejection, margin, confirmation."""
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = HOLM_ALPHA / (len(tests) - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)
        test["confirmed"] = bool(test["holm_reject"] and test["margin"])


def tuned_tests(runs: Runs, leave_one_out: bool) -> list[dict[str, Any]]:
    """The 24 primary tests (best-tuned cap per family vs parent TNBBeta).

    Args:
        runs: Parent + control runs, keyed ``(d, sigma, arm)``.
        leave_one_out: If true, seed ``s``'s cap arm is chosen on the other seeds (S1).
    """
    tests: list[dict[str, Any]] = []
    for dim, sigma in itertools.product(DIMS, SIGMAS):
        tnb = valid_gaps(runs.get((dim, sigma, "tnbbeta"), {}))
        for family in FAMILIES:
            pool = cap_pool(runs, dim, sigma, family)
            if not pool:
                continue
            selected = best_arm(runs, dim, sigma, pool)
            cap_gaps: dict[int, float] = {}
            for seed in range(NUM_SEEDS):
                arm = (
                    best_arm(runs, dim, sigma, pool, exclude_seed=seed)
                    if leave_one_out
                    else selected
                )
                value = valid_gaps(runs[(dim, sigma, arm)]).get(seed) if arm else None
                if value is not None:
                    cap_gaps[seed] = value
            seeds = sorted(set(tnb) & set(cap_gaps))
            if len(seeds) < 5:
                continue
            diff = np.array([cap_gaps[s] - tnb[s] for s in seeds])
            tests.append(
                {
                    "dim": dim,
                    "sigma": sigma,
                    "family": family,
                    "arm": selected,
                    "n": len(seeds),
                    "diff": diff,
                    "p": wilcoxon_greater(diff),
                    "p_enum": wilcoxon_greater_enumerated(diff),
                }
            )
    holm(tests)
    return tests


def summarize(out_dir: Path, parent_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown."""
    runs = load_runs(out_dir)
    runs.update({k: v for k, v in load_parent(parent_dir).items() if k not in runs})
    if not runs:
        print("no runs found")
        return
    print("\n## Per-arm test ELBO gap (nats)\n")
    _print_arm_table(runs)
    for title, loo in (
        ("Primary: best-tuned cap per family (in-sample) vs TNBBeta", False),
        ("Sensitivity S1: leave-one-seed-out tuned cap vs TNBBeta", True),
    ):
        print(f"\n## {title} (one-sided paired Wilcoxon, Holm 0.05 over 24)\n")
        _print_tests(tuned_tests(runs, loo))
    print("\n## S2 / S3: per-cell best caps, equal budget and pooled families\n")
    _print_pooled(runs)
    print("\n## S4: best pinned kappa, grid-edge flags, fitted kappa and sigma_hat\n")
    _print_pins(runs)
    print("\n## S4: kappa trajectories of trainable arms (median kappa at epochs)\n")
    _print_trajectories(runs)


def load_parent(parent_dir: Path) -> Runs:
    """The parent study's reused arms (decoder ``L``)."""
    return load_runs(parent_dir, PARENT_ARMS)


def _settings(
    base: str, pin: float | None, init: float | None, epochs: int, head_lr: float
) -> dict[str, Any]:
    family = "tnbbeta" if base == "tnbbeta" else base
    return {
        "base": base,
        "family": family,
        "kappa_pin": pin,
        "kappa_init": init,
        "epochs": epochs,
        "head_lr": head_lr,
    }


def _log_trajectory(
    trajectory: dict[str, list[float]], model: ControlVAE, y: Tensor
) -> None:
    """Appends the end-of-epoch summary statistics (no RNG consumed)."""
    with torch.no_grad():
        posterior = model.posterior(y)
        trajectory.setdefault("sigma_hat", []).append(float(model.learned_scale()))
        if hasattr(posterior, "epsilon"):
            for key in ("p", "q", "epsilon"):
                value = getattr(posterior, key).median()
                trajectory.setdefault(key, []).append(float(value))
        else:
            kappa = getattr(posterior, "kappa", None)
            if kappa is None:
                kappa = posterior.scale  # pyright: ignore[reportAttributeAccessIssue]
            trajectory.setdefault("kappa", []).append(float(kappa.median()))


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _arm_order(arm: str) -> tuple[int, int]:
    full = list(PARENT_ARMS) + control_arms()
    return (0, full.index(arm))


def _print_arm_table(runs: Runs) -> None:
    print(
        "| d | sigma | arm | valid | gap mean +- sd | min / max | IWAE gap | "
        "sigma_hat / sigma | kappa median | KL to model posterior |"
    )
    print("|" + "---|" * 10)
    keys = sorted(runs, key=lambda k: (k[1], k[0], _arm_order(k[2])))
    for dim, sigma, arm in keys:
        group = runs[(dim, sigma, arm)]
        gaps = np.array(list(valid_gaps(group).values()))
        if len(gaps) == 0:
            print(
                f"| {dim} | {sigma} | {arm} | 0/{len(group)} | - | - | - | - | - | - |"
            )
            continue
        iwae = np.mean(list(valid_gaps(group, "gap_iwae").values()))
        ratio = np.mean(list(valid_gaps(group, "sigma_ratio").values()))
        kappa = list(valid_gaps(group, "kappa_median").values())
        kl = np.mean(list(valid_gaps(group, "kl_to_model_posterior").values()))
        print(
            f"| {dim} | {sigma} | {arm} | {len(gaps)}/{len(group)} | "
            f"{gaps.mean():.4f} +- {_sd(gaps):.4f} | {gaps.min():.4f} / "
            f"{gaps.max():.4f} | {iwae:.4f} | {ratio:.3f} | "
            f"{np.median(kappa) if kappa else float('nan'):.3g} | {kl:.4f} |"
        )


def _print_tests(tests: list[dict[str, Any]]) -> None:
    print(
        "| d | sigma | family | best-tuned arm (in-sample) | n | "
        "mean diff (cap - tnbbeta) | "
        "seeds tnbbeta better | p (scipy) | p (enumerated) | Holm thr | Holm reject | "
        "margin | confirmed |"
    )
    print("|" + "---|" * 13)
    for t in sorted(tests, key=lambda t: (t["sigma"], t["dim"], t["family"])):
        wins = int((t["diff"] > 0).sum())
        print(
            f"| {t['dim']} | {t['sigma']} | {t['family']} | {t['arm']} | {t['n']} | "
            f"{t['diff'].mean():.4f} | {wins}/{t['n']} | {t['p']:.5f} | "
            f"{t['p_enum']:.5f} | {t['threshold']:.5f} | {t['holm_reject']} | "
            f"{t['margin']} | {t['confirmed']} |"
        )
        if abs(t["p"] - t["p_enum"]) > 1e-9:
            print(
                f"WARNING: scipy and enumerated p differ (d={t['dim']}, {t['sigma']})"
            )
    confirmed = sum(t["confirmed"] for t in tests)
    print(f"\nConfirmed (Holm and margin): {confirmed} of {len(tests)} tests.\n")
    print("| d | sigma | category (both families) |")
    print("|---|---|---|")
    for dim, sigma in itertools.product(DIMS, SIGMAS):
        mine = [t for t in tests if t["dim"] == dim and t["sigma"] == sigma]
        if len(mine) != 2:
            continue
        if all(t["confirmed"] for t in mine):
            label = "advantage survives"
        elif any(t["diff"].mean() <= MARGIN_NATS for t in mine):
            label = "within margin / artefact (diff <= 0.05 for at least one family)"
        else:
            label = "inconclusive (above margin, not Holm-significant)"
        print(f"| {dim} | {sigma} | {label} |")


def _print_pooled(runs: Runs) -> None:
    print(
        "| d | sigma | tnbbeta | tnbbeta_H | best TNBBeta (in-sample) | "
        "best cap overall "
        "(arm) | mean diff best cap - best TNBBeta | seeds TNBBeta better |"
    )
    print("|" + "---|" * 8)
    for dim, sigma in itertools.product(DIMS, SIGMAS):
        tnb_arms = [a for a in ("tnbbeta", "tnbbeta_H") if runs.get((dim, sigma, a))]
        caps = [a for f in FAMILIES for a in cap_pool(runs, dim, sigma, f)]
        if not tnb_arms or not caps:
            continue
        bt = best_arm(runs, dim, sigma, tnb_arms)
        bc = best_arm(runs, dim, sigma, caps)
        if bt is None or bc is None:
            continue
        g_t = valid_gaps(runs[(dim, sigma, bt)])
        g_c = valid_gaps(runs[(dim, sigma, bc)])
        seeds = sorted(set(g_t) & set(g_c))
        diff = np.array([g_c[s] - g_t[s] for s in seeds])
        means = {
            a: np.mean(list(valid_gaps(runs[(dim, sigma, a)]).values()))
            if runs.get((dim, sigma, a))
            else float("nan")
            for a in ("tnbbeta", "tnbbeta_H")
        }
        print(
            f"| {dim} | {sigma} | {means['tnbbeta']:.4f} | {means['tnbbeta_H']:.4f} | "
            f"{bt} | {bc} | {diff.mean():.4f} | {int((diff > 0).sum())}/{len(seeds)} |"
        )


def _print_pins(runs: Runs) -> None:
    print(
        "| d | sigma | family | best pinned kappa | edge? | gap at best pin | "
        "best pool arm |"
    )
    print("|" + "---|" * 7)
    for dim, sigma, family in itertools.product(DIMS, SIGMAS, FAMILIES):
        pins = [
            (
                k,
                float(
                    np.mean(
                        list(
                            valid_gaps(runs[(dim, sigma, f"{family}_pin{k}")]).values()
                        )
                    )
                ),
            )
            for k in PIN_GRID
            if runs.get((dim, sigma, f"{family}_pin{k}"))
            and valid_gaps(runs[(dim, sigma, f"{family}_pin{k}")])
        ]
        if not pins:
            continue
        kappa, gap = min(pins, key=lambda kv: kv[1])
        edge = "EDGE" if kappa in (PIN_GRID[0], PIN_GRID[-1]) else ""
        pool = cap_pool(runs, dim, sigma, family)
        print(
            f"| {dim} | {sigma} | {family} | {kappa} | {edge} | {gap:.4f} | "
            f"{best_arm(runs, dim, sigma, pool)} |"
        )


def _print_trajectories(runs: Runs) -> None:
    epochs = (1, 10, 30, 100, 300)
    print("| d | sigma | arm | " + " | ".join(f"ep {e}" for e in epochs) + " | final |")
    print("|" + "---|" * (4 + len(epochs)))
    trainable = [a for a in control_arms() if "pin" not in a]
    for dim, sigma, arm in itertools.product(DIMS, SIGMAS, trainable):
        group = runs.get((dim, sigma, arm), {})
        curves = []
        for record in group.values():
            traj = record.get("trajectory", {})
            curve = traj.get("kappa") or traj.get("epsilon")
            if curve:
                curves.append(curve)
        if not curves:
            continue
        cells = []
        for e in epochs:
            vals = [c[e - 1] for c in curves if len(c) >= e]
            cells.append(f"{np.median(vals):.3g}" if vals else "-")
        final = np.median([c[-1] for c in curves])
        name = arm + (" (eps)" if arm.startswith("tnbbeta") else " (kappa)")
        print(
            f"| {dim} | {sigma} | {name} | " + " | ".join(cells) + f" | {final:.3g} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
