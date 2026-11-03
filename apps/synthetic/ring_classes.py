"""Four-class ring designs on S^2: does TNBBeta use its ring capacity, reliably?

Pre-registration: ``writeup/results/ring_class_designs_preregistration_2026-11-01.md``
(generative models, arms, training, metrics, tests and decision rules are fixed there;
deviations are listed at its end). Task: :mod:`tnbbeta_vae.data.ring_classes`.

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.ring_classes train --design 1 --arm tnbbeta \
        --seed 0 --out-dir ring_recovery/classes
    uv run python -m apps.synthetic.ring_classes summarize \
        --out-dir ring_recovery/classes

Each run writes ``<out-dir>/design<d>/<arm>/s<seed>.{json,npz,pt}``: the summary, the
per-x arrays and the checkpoint.
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
from scipy import stats
import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from tnbbeta_vae.data.ring_classes import (
    NUM_CLASSES,
    RingClassTask,
    exact_class_kl,
    ring_class_task,
)
from tnbbeta_vae.models.architectures.mlp import mlp_stack
from tnbbeta_vae.models.heads import (
    P_RAW_INIT,
    LatentFamily,
    head_size,
    init_tnbbeta_p_bias,
    posterior_from_raw,
    standard_prior,
    tnbbeta_posterior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo
from tnbbeta_vae.training import select_device

SIGMA = 0.2
ARMS = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0", "tnbbeta_fixed_mu")
PRIMARY_COMPARATORS = ("vmf", "power_spherical")
DESIGN_ARMS = {
    1: ARMS,
    2: ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0"),
}
NUM_SEEDS = 10
X3_NATS = 0.25
TV_THRESHOLD = 0.25
RING_FRACTION = 0.5
RING_SEEDS = 8
ALIGNED_COS = 0.99
HOLM_ALPHA = 0.05
TEST_SEED = 424242
TV_DRAWS = 8000
_NUM_BINS = 90
_POLE_DEG = 10.0
_EPSILON_FLOOR = 1e-4
_HIDDEN = [128, 128]


class RingClassVAE(nn.Module):
    """A thin study wrapper: MLP encoder + posterior head, frozen true decoder.

    Attributes:
        task: The ring task (selects the axis by class in the decoder).
        family: The latent family of the head.
        fixed_mean_direction: TNBBeta only: pin the mean direction to the pole ``e_1``.
    """

    def __init__(
        self,
        task: RingClassTask,
        family: LatentFamily,
        p_init_offset: float = P_RAW_INIT,
        fixed_mean_direction: bool = False,
    ) -> None:
        """Builds the encoder and head.

        Args:
            task: The task (its ``input_dim`` is the encoder input size).
            family: ``"vmf"``, ``"power_spherical"`` or ``"tnbbeta"``.
            p_init_offset: TNBBeta only: initial raw-p bias of the head (library default
                ``ln 3``; ``0.0`` keeps the plain random ``nn.Linear`` bias).
            fixed_mean_direction: TNBBeta only: head outputs no direction.
        """
        super().__init__()
        self.task = task
        self.family: LatentFamily = family
        self.fixed_mean_direction = fixed_mean_direction
        self.encoder = mlp_stack([task.input_dim, *_HIDDEN])
        size = head_size(family, 3) - (3 if fixed_mean_direction else 0)
        self.posterior_head = nn.Linear(_HIDDEN[-1], size)
        if family == "tnbbeta":
            init_tnbbeta_p_bias(
                self.posterior_head,
                3,
                fixed_mean_direction=fixed_mean_direction,
                offset=p_init_offset,
            )

    def posterior(self, x: Tensor, cls: Tensor) -> Distribution:
        """``q(z | x, c)`` for a batch (the class is an input only in design 2)."""
        raw = self.posterior_head(self.encoder(self.task.encoder_input(x, cls)))
        if self.family == "tnbbeta":
            return tnbbeta_posterior(
                raw,
                3,
                fixed_mean_direction=self.fixed_mean_direction,
                epsilon_floor=_EPSILON_FLOOR,
            )
        return posterior_from_raw(self.family, raw, 3)

    def negative_elbo(
        self,
        x: Tensor,
        cls: Tensor,
        kl_weight: float,
        num_samples: int,
    ) -> Tensor:
        """The training loss: mean over the batch of ``-(log p(x|z) - w * KL)``.

        The decoder is the frozen true decoder with sigma 0.2; the KL to the uniform
        sphere is analytic for vMF / Power Spherical and Monte Carlo (exact ``log q``)
        for TNBBeta.
        """
        posterior = self.posterior(x, cls)
        prior = standard_prior(self.family, 3, x.device)
        terms = monte_carlo_elbo(
            x,
            posterior,
            prior,
            lambda z: self.task.decode(z, cls),
            SIGMA,
            num_samples,
            analytic_kl=self.family != "tnbbeta",
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()


def arm_spec(arm: str) -> dict[str, Any]:
    """Model settings of an arm: family, p-init offset and fixed-direction flag."""
    if arm in ("vmf", "power_spherical"):
        return {
            "family": arm,
            "p_init_offset": P_RAW_INIT,
            "fixed_mean_direction": False,
        }
    if arm == "tnbbeta":
        return {
            "family": "tnbbeta",
            "p_init_offset": P_RAW_INIT,
            "fixed_mean_direction": False,
        }
    if arm == "tnbbeta_init0":
        return {
            "family": "tnbbeta",
            "p_init_offset": 0.0,
            "fixed_mean_direction": False,
        }
    if arm == "tnbbeta_fixed_mu":
        return {
            "family": "tnbbeta",
            "p_init_offset": P_RAW_INIT,
            "fixed_mean_direction": True,
        }
    raise ValueError(f"unknown arm {arm!r}.")


def run_path(out_dir: Path, design: int, arm: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / f"design{design}" / arm / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--design", type=int, choices=[1, 2], required=True)
    train.add_argument("--arm", choices=list(ARMS), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--num-train", type=int, default=20_000)
    train.add_argument("--num-test", type=int, default=2_000)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--kl-draws", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        run_train(args)
    else:
        summarize(args.out_dir)


def run_train(args: argparse.Namespace) -> None:
    """Trains one (design, arm, seed) on the GPU and evaluates it on the CPU."""
    if args.design == 2 and args.arm == "tnbbeta_fixed_mu":
        raise SystemExit("tnbbeta_fixed_mu is a design-1 arm only.")
    device = select_device(args.device)
    started = time.time()
    task = ring_class_task(args.design)
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_x, _, train_cls, _ = task.sample(args.num_train, train_gen)
    spec = arm_spec(args.arm)
    torch.manual_seed(args.seed)
    model = RingClassVAE(task, **spec)
    model.to(device)
    train_x, train_cls = train_x.to(device), train_cls.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(200 + args.seed)
    warmup = max(1, args.epochs // 10)
    losses = []
    for epoch in range(args.epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(args.num_train, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for index in order.split(args.batch_size):
            optimiser.zero_grad()
            loss = model.negative_elbo(
                train_x[index], train_cls[index], kl_weight, args.num_elbo_samples
            )
            loss.backward()
            optimiser.step()
            running += loss.detach() * len(index)
        losses.append(float(running) / args.num_train)
    model.cpu().double().eval()
    train_seconds = time.time() - started

    test_gen = torch.Generator().manual_seed(TEST_SEED)
    test_x, _, test_cls, test_w = task.sample(
        args.num_test, test_gen, balanced=args.design == 2
    )
    torch.manual_seed(31337)
    arrays = evaluate(model, task, test_x.double(), test_cls, test_w, args.kl_draws)
    summary = summarize_run(arrays, task)
    summary.update(
        {
            "design": args.design,
            "arm": args.arm,
            "seed": args.seed,
            "family": spec["family"],
            "p_init_offset": spec["p_init_offset"],
            "fixed_mean_direction": spec["fixed_mean_direction"],
            "epochs": args.epochs,
            "num_train": args.num_train,
            "num_test": args.num_test,
            "num_elbo_samples": args.num_elbo_samples,
            "kl_draws": args.kl_draws,
            "tv_draws": TV_DRAWS,
            "final_train_loss": losses[-1],
            "loss_curve": losses[:: max(1, args.epochs // 30)],
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(args.out_dir, args.design, args.arm, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    print(
        f"design{args.design} {args.arm} s{args.seed}: KL {summary['kl_mean']:.4f} "
        f"frac TV<=0.25 {summary['frac_tv_within']:.3f} "
        f"median |cos| {summary['median_abs_cos']:.4f} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: RingClassVAE,
    task: RingClassTask,
    x: Tensor,
    cls: Tensor,
    w: Tensor,
    kl_draws: int,
    chunk: int = 250,
) -> dict[str, np.ndarray]:
    """Per-x exact KL, latitude-marginal TV, band/pole mass and fitted parameters.

    Args:
        model: The (float64, CPU) model.
        task: The task.
        x: Test observations ``(n, 3)``, float64.
        cls: Test classes ``(n,)``.
        w: True latitudes ``(n,)``.
        kl_draws: ``z ~ q`` draws per x for the Monte Carlo exact KL.
        chunk: Observations per pass.

    Returns:
        A dict of numpy arrays of length ``n`` (``mu`` is ``(n, 3)``).
    """
    columns: dict[str, list[Tensor]] = {}
    edges = torch.linspace(0, math.pi, _NUM_BINS + 1, dtype=torch.float64)
    pole_rad = math.radians(_POLE_DEG)
    theta_nodes = (torch.arange(task.grid_size, dtype=torch.float64) + 0.5) * (
        math.pi / task.grid_size
    )
    for start in range(0, x.shape[0], chunk):
        sl = slice(start, start + chunk)
        xs, cs = x[sl], cls[sl]
        posterior = model.posterior(xs, cs)
        kl, se = exact_class_kl(posterior, task, xs, cs, kl_draws)
        z = posterior.rsample(torch.Size([TV_DRAWS]))
        angle = torch.arccos((z * task.axis_of(cs)).sum(-1).clamp(-1, 1))  # (draws, b)
        bins = (angle / math.pi * _NUM_BINS).long().clamp(max=_NUM_BINS - 1)
        counts = torch.zeros(xs.shape[0], _NUM_BINS, dtype=torch.float64)
        counts.scatter_add_(
            1, bins.T.contiguous(), torch.ones_like(bins.T, dtype=torch.float64)
        )
        true_mass = torch.diff(task.latitude_cdf(xs, edges), dim=-1)
        tv = 0.5 * (counts / TV_DRAWS - true_mass).abs().sum(-1)
        _, density = task.latitude_density(xs)
        cdf = torch.cumsum(density * (math.pi / task.grid_size), dim=-1)
        lo = theta_nodes[(cdf < 0.025).sum(-1).clamp(max=task.grid_size - 1)]
        hi = theta_nodes[(cdf < 0.975).sum(-1).clamp(max=task.grid_size - 1)]
        band = ((angle >= lo) & (angle <= hi)).double().mean(0)
        pole = ((angle < pole_rad) | (angle > math.pi - pole_rad)).double().mean(0)
        pieces = {
            "kl": kl,
            "kl_se": se,
            "tv": tv,
            "band_mass": band,
            "pole_mass": pole,
            "kl_uniform": task.kl_uniform_to_posterior(xs),
            "log_evidence": task.log_evidence(xs),
            **_fitted_parameters(posterior, task.axis_of(cs)),
        }
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["x"] = x.numpy()
    arrays["cls"] = cls.numpy()
    arrays["w"] = w.double().numpy()
    return arrays


def summarize_run(arrays: dict[str, np.ndarray], task: RingClassTask) -> dict[str, Any]:
    """Scalar summaries of one run's per-x arrays (overall and by class)."""
    cls = arrays["cls"]
    within = arrays["tv"] <= TV_THRESHOLD
    summary: dict[str, Any] = {
        "kl_mean": float(arrays["kl"].mean()),
        "kl_median": float(np.median(arrays["kl"])),
        "kl_sd": float(arrays["kl"].std(ddof=1)),
        "kl_se_mean": float(arrays["kl_se"].mean()),
        "kl_uniform_mean": float(arrays["kl_uniform"].mean()),
        "tv_mean": float(arrays["tv"].mean()),
        "tv_median": float(np.median(arrays["tv"])),
        "frac_tv_within": float(within.mean()),
        "band_mass_mean": float(arrays["band_mass"].mean()),
        "pole_mass_mean": float(arrays["pole_mass"].mean()),
        "median_abs_cos": float(np.median(arrays["abs_cos"])),
        "aligned": bool(np.median(arrays["abs_cos"]) >= ALIGNED_COS),
        "n_by_class": [int((cls == c).sum()) for c in range(NUM_CLASSES)],
        "kl_by_class": _by_class(arrays["kl"], cls, np.mean),
        "median_abs_cos_by_class": _by_class(arrays["abs_cos"], cls, np.median),
        "frac_tv_within_by_class": _by_class(within.astype(float), cls, np.mean),
    }
    for key in ("p", "q", "epsilon", "kappa"):
        values = arrays[key]
        if np.isfinite(values).any():
            summary[f"{key}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        summary["m_median"] = float(np.nanmedian(arrays["epsilon"])) - 1.0
    del task
    return summary


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = _load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    for design in (1, 2):
        arms = {a: runs.get((design, a), {}) for a in DESIGN_ARMS[design]}
        if not any(arms.values()):
            continue
        print(f"\n## Design {design}\n")
        _print_arm_table(design, arms)
        _print_per_seed(design, arms)
        _print_class_tables(design, arms)
    print("\n## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)\n")
    tests = _confirmatory_tests(runs)
    _print_confirmatory(tests, runs)
    print("\n## Descriptive comparisons (no p-values)\n")
    _print_descriptive(runs)


# ----------------------------------------------------------------------------- helpers


def _fitted_parameters(posterior: Distribution, axis: Tensor) -> dict[str, Tensor]:
    """Per-x fitted parameters (NaN if absent), ``mu`` and ``|cos(mu, axis)|``."""
    nan = torch.full(axis.shape[:-1], float("nan"), dtype=axis.dtype)
    out = {"p": nan, "q": nan, "epsilon": nan, "kappa": nan}
    if hasattr(posterior, "epsilon"):
        direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
        out["p"] = posterior.p  # pyright: ignore[reportAttributeAccessIssue]
        out["q"] = posterior.q  # pyright: ignore[reportAttributeAccessIssue]
        out["epsilon"] = posterior.epsilon  # pyright: ignore[reportAttributeAccessIssue]
    elif hasattr(posterior, "kappa"):
        direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
        out["kappa"] = posterior.kappa  # pyright: ignore[reportAttributeAccessIssue]
    else:
        direction = posterior.loc  # pyright: ignore[reportAttributeAccessIssue]
        out["kappa"] = posterior.scale[..., 0]  # pyright: ignore[reportAttributeAccessIssue]
    out["mu"] = direction
    out["abs_cos"] = (direction * axis).sum(-1).abs()
    return out


def _by_class(values: np.ndarray, cls: np.ndarray, reducer: Any) -> list[float]:
    return [float(reducer(values[cls == c])) for c in range(NUM_CLASSES)]


def _load_runs(out_dir: Path) -> dict[tuple[int, str], dict[int, dict[str, Any]]]:
    runs: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("design*/*/s*.json")):
        record = json.loads(path.read_text())
        key = (record["design"], record["arm"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _print_arm_table(design: int, arms: dict[str, dict[int, dict[str, Any]]]) -> None:
    print(
        "| arm | seeds | mean test KL +- sd over seeds | median seed KL | "
        "seeds with >=half x within TV 0.25 | aligned seeds (median abs cos >= 0.99) | "
        "median abs cos (seed range) | KL(Uniform) ref |"
    )
    print("|" + "---|" * 8)
    for arm, seeds in arms.items():
        if not seeds:
            continue
        kl = np.array([s["kl_mean"] for s in seeds.values()])
        ring = sum(s["frac_tv_within"] >= RING_FRACTION for s in seeds.values())
        aligned = sum(s["aligned"] for s in seeds.values())
        cos = np.array([s["median_abs_cos"] for s in seeds.values()])
        ref = np.mean([s["kl_uniform_mean"] for s in seeds.values()])
        print(
            f"| {arm} | {len(seeds)} | {kl.mean():.3f} +- {_sd(kl):.3f} | "
            f"{np.median(kl):.3f} | {ring}/{len(seeds)} | {aligned}/{len(seeds)} | "
            f"{np.median(cos):.3f} ({cos.min():.3f}-{cos.max():.3f}) | {ref:.2f} |"
        )
    del design


def _print_per_seed(design: int, arms: dict[str, dict[int, dict[str, Any]]]) -> None:
    print(f"\nPer-seed (design {design}): KL / frac TV<=0.25 / median abs cos\n")
    seeds = sorted({s for runs in arms.values() for s in runs})
    print("| seed | " + " | ".join(a for a, r in arms.items() if r) + " |")
    print("|" + "---|" * (1 + sum(bool(r) for r in arms.values())))
    for seed in seeds:
        cells = []
        for runs in arms.values():
            if not runs:
                continue
            r = runs.get(seed)
            cells.append(
                "-"
                if r is None
                else f"{r['kl_mean']:.3f} / {r['frac_tv_within']:.2f} / "
                f"{r['median_abs_cos']:.3f}"
            )
        print(f"| {seed} | " + " | ".join(cells) + " |")


def _print_class_tables(
    design: int, arms: dict[str, dict[int, dict[str, Any]]]
) -> None:
    label = "band" if design == 1 else "class"
    print(f"\nMean exact KL by {label} (mean over seeds; test x of that {label})\n")
    print(
        "| arm | " + " | ".join(f"{label} {c + 1}" for c in range(NUM_CLASSES)) + " |"
    )
    print("|" + "---|" * (1 + NUM_CLASSES))
    rows = []
    for arm, seeds in arms.items():
        if seeds:
            rows.append((arm, list(seeds.values())))
    for arm, group in rows:
        by_class = np.mean([s["kl_by_class"] for s in group], axis=0)
        print(f"| {arm} | " + " | ".join(f"{v:.3f}" for v in by_class) + " |")
    print(f"\nMean exact KL by {label}, split by alignment of the seed\n")
    print(
        "| arm | seeds | "
        + " | ".join(f"{label} {c + 1}" for c in range(NUM_CLASSES))
        + " |"
    )
    print("|" + "---|" * (2 + NUM_CLASSES))
    for arm, group in rows:
        for aligned in (True, False):
            subset = [s for s in group if s["aligned"] == aligned]
            if not subset:
                continue
            by_class = np.mean([s["kl_by_class"] for s in subset], axis=0)
            tag = "aligned" if aligned else "not aligned"
            print(
                f"| {arm} ({tag}) | {len(subset)} | "
                + " | ".join(f"{v:.3f}" for v in by_class)
                + " |"
            )
    print(f"\nSeeds with per-{label} median abs cos >= 0.99 (out of seeds run)\n")
    print(
        "| arm | " + " | ".join(f"{label} {c + 1}" for c in range(NUM_CLASSES)) + " |"
    )
    print("|" + "---|" * (1 + NUM_CLASSES))
    for arm, group in rows:
        counts = [
            sum(s["median_abs_cos_by_class"][c] >= ALIGNED_COS for s in group)
            for c in range(NUM_CLASSES)
        ]
        print(f"| {arm} | " + " | ".join(f"{n}/{len(group)}" for n in counts) + " |")
    print(f"\nFraction of test x within TV 0.25, by {label} (mean over seeds)\n")
    print(
        "| arm | " + " | ".join(f"{label} {c + 1}" for c in range(NUM_CLASSES)) + " |"
    )
    print("|" + "---|" * (1 + NUM_CLASSES))
    for arm, group in rows:
        by_class = np.mean([s["frac_tv_within_by_class"] for s in group], axis=0)
        print(f"| {arm} | " + " | ".join(f"{v:.3f}" for v in by_class) + " |")


def _wilcoxon_greater(diff: np.ndarray) -> float:
    """One-sided (greater) Wilcoxon signed-rank p-value (scipy) of paired diffs."""
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def _wilcoxon_greater_enumerated(diff: np.ndarray) -> float:
    """The same exact p-value by enumerating all ``2^n`` sign patterns."""
    diff = diff[diff != 0]
    ranks = stats.rankdata(np.abs(diff))
    observed = ranks[diff > 0].sum()
    count = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=len(diff)):
        total += 1
        if (ranks * np.array(signs)).sum() >= observed - 1e-9:
            count += 1
    return count / total


def _confirmatory_tests(
    runs: dict[tuple[int, str], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    tests = []
    for design in (1, 2):
        tnb = runs.get((design, "tnbbeta"), {})
        for other in PRIMARY_COMPARATORS:
            cap = runs.get((design, other), {})
            seeds = sorted(set(tnb) & set(cap))
            if len(seeds) < 5:
                continue
            diff = np.array([cap[s]["kl_mean"] - tnb[s]["kl_mean"] for s in seeds])
            tests.append(
                {
                    "design": design,
                    "other": other,
                    "n": len(seeds),
                    "diff": diff,
                    "p": _wilcoxon_greater(diff),
                    "p_enum": _wilcoxon_greater_enumerated(diff),
                }
            )
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = HOLM_ALPHA / (len(tests) - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
        test["margin"] = bool(test["diff"].mean() >= X3_NATS)
    return tests


def _print_confirmatory(
    tests: list[dict[str, Any]],
    runs: dict[tuple[int, str], dict[int, dict[str, Any]]],
) -> None:
    print(
        "| design | comparator | n | mean paired diff (comparator - tnbbeta) | "
        "seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | "
        "Holm reject | margin >= 0.25 |"
    )
    print("|" + "---|" * 10)
    for test in sorted(tests, key=lambda t: (t["design"], t["other"])):
        wins = int((test["diff"] > 0).sum())
        print(
            f"| {test['design']} | {test['other']} | {test['n']} | "
            f"{test['diff'].mean():.3f} | {wins}/{test['n']} | {test['p']:.5f} | "
            f"{test['p_enum']:.5f} | {test['threshold']:.5f} | "
            f"{test['holm_reject']} | {test['margin']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for design {test['design']}")
    print("\nDecision rule 'TNBBeta uses its ring capacity' (both Holm tests, margin, ")
    print("and >= 8 of 10 seeds with >= half of test x within TV 0.25):\n")
    print(
        "| design | Holm (both) | margin (both) | ring-shape seeds | "
        "aligned seeds | rule met |"
    )
    print("|" + "---|" * 6)
    for design in (1, 2):
        mine = [t for t in tests if t["design"] == design]
        group = runs.get((design, "tnbbeta"), {})
        if len(mine) < 2 or not group:
            continue
        holm = all(t["holm_reject"] for t in mine)
        margin = all(t["margin"] for t in mine)
        ring = sum(s["frac_tv_within"] >= RING_FRACTION for s in group.values())
        aligned = sum(s["aligned"] for s in group.values())
        met = holm and margin and ring >= RING_SEEDS
        print(
            f"| {design} | {holm} | {margin} | {ring}/{len(group)} | "
            f"{aligned}/{len(group)} | {met} |"
        )


def _print_descriptive(runs: dict[tuple[int, str], dict[int, dict[str, Any]]]) -> None:
    print(
        "| design | comparison | n | mean KL (A) | mean KL (B) | mean paired diff "
        "(A - B) | seeds A better |"
    )
    print("|" + "---|" * 7)
    pairs = [
        (1, "tnbbeta", "tnbbeta_init0"),
        (2, "tnbbeta", "tnbbeta_init0"),
        (1, "tnbbeta_fixed_mu", "tnbbeta"),
        (1, "tnbbeta_fixed_mu", "vmf"),
        (1, "tnbbeta_fixed_mu", "power_spherical"),
        (1, "tnbbeta_init0", "vmf"),
        (2, "tnbbeta_init0", "vmf"),
        (1, "tnbbeta_init0", "power_spherical"),
        (2, "tnbbeta_init0", "power_spherical"),
        (1, "vmf", "power_spherical"),
        (2, "vmf", "power_spherical"),
    ]
    for design, a, b in pairs:
        ra, rb = runs.get((design, a), {}), runs.get((design, b), {})
        seeds = sorted(set(ra) & set(rb))
        if not seeds:
            continue
        ka = np.array([ra[s]["kl_mean"] for s in seeds])
        kb = np.array([rb[s]["kl_mean"] for s in seeds])
        print(
            f"| {design} | {a} vs {b} | {len(seeds)} | {ka.mean():.3f} | "
            f"{kb.mean():.3f} | "
            f"{(ka - kb).mean():.3f} | {int((ka < kb).sum())}/{len(seeds)} |"
        )
    print(
        "\nFitted parameters (median over seeds of the per-seed median over test x):\n"
    )
    print("| design | arm | p | q | epsilon | m = eps - 1 | kappa |")
    print("|" + "---|" * 7)
    for (design, arm), seeds in sorted(runs.items()):

        def med(key: str, seeds: dict[int, dict[str, Any]] = seeds) -> str:
            values = [s[key] for s in seeds.values() if key in s]
            return f"{np.median(values):.3g}" if values else "-"

        print(
            f"| {design} | {arm} | {med('p_median')} | {med('q_median')} | "
            f"{med('epsilon_median')} | {med('m_median')} | {med('kappa_median')} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
