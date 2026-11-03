"""ITD-only HRTF task: does TNBBeta use its ring capacity on measured data?

Pre-registration: ``writeup/results/hrtf_itd_ring_preregistration_2026-11-01.md``
(data, splits, arms, training, metrics, tests and decision rules are fixed there;
deviations are listed at its end). Data: ``apps/data/prepare_sadie2_itd.py`` ->
:mod:`tnbbeta_vae.data.hrtf`.

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.hrtf_itd train --arm V --family tnbbeta \
        --seed 0 --estimator xcorr --out-dir hrtf_runs
    uv run python -m apps.synthetic.hrtf_itd summarize --out-dir hrtf_runs

Each run writes ``<out-dir>/<estimator>/arm<A>/<family>/s<seed>.{json,npz,pt}``.
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

from apps.eval.hrtf_itd_gate import _assign, _cap_nll, _fit_bins, _ring_nll
from tnbbeta_vae.data.hrtf import (
    AXIS,
    AreaWeightedSampler,
    HrtfSplit,
    load_hrtf,
    make_split,
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
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.training import select_device

FAMILIES = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0")
ARMS = ("V", "L")
ESTIMATORS = ("xcorr", "onset")
PRIMARY_COMPARATORS = ("vmf", "power_spherical")
NUM_SPLITS = 10
MARGIN_NATS = 0.25
ALIGNED_COS = 0.99
HOLM_ALPHA = 0.05
KL_DRAWS = 1000
LAT_DRAWS = 4000
GRID_NODES = 4001
INITIAL_SIGMA = 0.5
EVAL_SEED = 31337
_HIDDEN = [128, 128]
_G_HIDDEN = 32
_EPSILON_FLOOR = 1e-4
_AXIS = torch.tensor(AXIS, dtype=torch.float64)


class HrtfModel(nn.Module):
    """Study wrapper: MLP encoder + posterior head, and in arm V a single-index decoder.

    Attributes:
        arm: ``"V"`` (VAE with decoder ``g(u . z)``) or ``"L"`` (encoder + head only).
        family: The latent family of the head.
    """

    def __init__(
        self,
        arm: str,
        family: LatentFamily,
        p_init_offset: float = P_RAW_INIT,
        initial_sigma: float = INITIAL_SIGMA,
    ) -> None:
        """Builds the encoder, the head and (arm V) the decoder.

        Args:
            arm: ``"V"`` or ``"L"``.
            family: ``"vmf"``, ``"power_spherical"`` or ``"tnbbeta"``.
            p_init_offset: TNBBeta only: initial raw-p bias of the head.
            initial_sigma: Arm V: starting value of the learned likelihood sigma.
        """
        super().__init__()
        self.arm = arm
        self.family: LatentFamily = family
        self.encoder = mlp_stack([1, *_HIDDEN])
        self.posterior_head = nn.Linear(_HIDDEN[-1], head_size(family, 3))
        if family == "tnbbeta":
            init_tnbbeta_p_bias(self.posterior_head, 3, offset=p_init_offset)
        if arm == "V":
            self.raw_u = nn.Parameter(torch.randn(3))
            self.g = nn.Sequential(
                nn.Linear(1, _G_HIDDEN),
                nn.Tanh(),
                nn.Linear(_G_HIDDEN, _G_HIDDEN),
                nn.Tanh(),
                nn.Linear(_G_HIDDEN, 1),
            )
            self.learned_scale = LearnedLikelihoodScale(initial_sigma)

    @property
    def u(self) -> Tensor:
        """The decoder's unit vector (arm V)."""
        return self.raw_u / self.raw_u.norm()

    def posterior(self, x: Tensor) -> Distribution:
        """``q(z | x)`` for standardised ITDs ``x`` of shape ``(batch, 1)``."""
        raw = self.posterior_head(self.encoder(x))
        if self.family == "tnbbeta":
            return tnbbeta_posterior(raw, 3, epsilon_floor=_EPSILON_FLOOR)
        return posterior_from_raw(self.family, raw, 3)

    def g_of_latitude(self, w: Tensor) -> Tensor:
        """The decoder's scalar mean ``g(w')`` at latitudes ``w'`` (shape of ``w``)."""
        return self.g(w[..., None])[..., 0]

    def decode(self, z: Tensor) -> Tensor:
        """The decoder mean ``g(u . z)``, shape ``(..., batch, 1)``."""
        return self.g_of_latitude((z * self.u).sum(-1))[..., None]

    def loss(self, x: Tensor, z_true: Tensor, kl_weight: float, num_samples: int):
        """The training loss for a batch (labels ``z_true`` are used in arm L only)."""
        posterior = self.posterior(x)
        if self.arm == "L":
            return -posterior.log_prob(z_true).mean()
        prior = standard_prior(self.family, 3, x.device)
        terms = monte_carlo_elbo(
            x,
            posterior,
            prior,
            self.decode,
            self.learned_scale(),
            num_samples,
            analytic_kl=self.family != "tnbbeta",
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()


def family_spec(name: str) -> dict[str, Any]:
    """Model settings of a family arm name: family and p-init offset."""
    if name in ("vmf", "power_spherical"):
        return {"family": name, "p_init_offset": P_RAW_INIT}
    if name == "tnbbeta":
        return {"family": "tnbbeta", "p_init_offset": P_RAW_INIT}
    if name == "tnbbeta_init0":
        return {"family": "tnbbeta", "p_init_offset": 0.0}
    raise ValueError(f"unknown family {name!r}.")


def run_path(out_dir: Path, estimator: str, arm: str, family: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / estimator / f"arm{arm}" / family / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=list(ARMS), required=True)
    train.add_argument("--family", choices=list(FAMILIES), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--estimator", choices=list(ESTIMATORS), default="xcorr")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--kl-draws", type=int, default=KL_DRAWS)
    train.add_argument("--lat-draws", type=int, default=LAT_DRAWS)
    train.add_argument("--grid-nodes", type=int, default=GRID_NODES)
    train.add_argument("--data-path", type=Path, default=None)
    train.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="Smoke tests only: keep the first N subject ids.",
    )
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    if args.command == "train":
        run_train(args)
    else:
        summarize(args.out_dir)


def load_split(
    estimator: str, seed: int, path: Path | None = None, subjects: int | None = None
) -> HrtfSplit:
    """The seed's split of the table (optionally the first ``subjects`` subjects)."""
    table = load_hrtf(path, estimator)  # pyright: ignore[reportArgumentType]
    if subjects is not None:
        keep = table.subject <= np.sort(table.subjects)[subjects - 1]
        table = type(table)(
            table.subject[keep],
            table.direction[keep],
            table.itd[keep],
            table.weight[keep],
        )
    return make_split(table, seed)


def run_train(args: argparse.Namespace) -> None:
    """Trains one (arm, family, seed, estimator) on the GPU and evaluates on the CPU."""
    device = select_device(args.device)
    started = time.time()
    split = load_split(args.estimator, args.seed, args.data_path, args.subjects)
    spec = family_spec(args.family)
    torch.manual_seed(args.seed)
    model = HrtfModel(args.arm, **spec)
    model.to(device)
    x_all = split.train_x.to(device)
    z_all = split.train_z.to(device)
    sampler = AreaWeightedSampler(split.train_weight, args.batch_size, 200 + args.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup = max(1, args.epochs // 10)
    losses = []
    for epoch in range(args.epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        running = torch.zeros((), device=device)
        batches = sampler.epoch()
        for index in batches:
            index = index.to(device)
            optimiser.zero_grad()
            loss = model.loss(
                x_all[index], z_all[index], kl_weight, args.num_elbo_samples
            )
            loss.backward()
            optimiser.step()
            running += loss.detach()
        losses.append(float(running) / len(batches))
    model.cpu().double().eval()
    train_seconds = time.time() - started

    torch.manual_seed(EVAL_SEED)
    arrays, extra = evaluate(
        model, split, args.kl_draws, args.lat_draws, args.grid_nodes
    )
    summary = summarize_run(arrays, extra, split)
    summary.update(
        {
            "arm": args.arm,
            "family": args.family,
            "seed": args.seed,
            "estimator": args.estimator,
            "p_init_offset": spec["p_init_offset"],
            "epochs": args.epochs,
            "n_train": len(split.train_weight),
            "n_test": len(split.test_weight),
            "train_subjects": split.train_subjects.tolist(),
            "test_subjects": split.test_subjects.tolist(),
            "itd_mean_us": split.mean,
            "itd_sd_us": split.sd,
            "num_elbo_samples": args.num_elbo_samples,
            "kl_draws": args.kl_draws,
            "lat_draws": args.lat_draws,
            "grid_nodes": args.grid_nodes,
            "final_train_loss": losses[-1],
            "loss_curve": losses[:: max(1, args.epochs // 30)],
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(args.out_dir, args.estimator, args.arm, args.family, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    print(
        f"{args.estimator} arm{args.arm} {args.family} s{args.seed}: "
        f"{summary['metric_name']} {summary['metric']:.4f} "
        f"median |cos| {summary['median_abs_cos']:.4f} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: HrtfModel,
    split: HrtfSplit,
    kl_draws: int = KL_DRAWS,
    lat_draws: int = LAT_DRAWS,
    grid_nodes: int = GRID_NODES,
    chunk: int = 250,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-x metric, alignment and latitude check on the held-out subjects.

    Arm V: exact ``KL(q || p_model(. | x))`` (``kl``). Arm L: ``-log q(z_true | x)``
    (``nll``). Both: ``|cos(mu, axis)|`` with the axis ``u`` (V) or ``+y`` (L), and the
    label-based latitude check about the signed axis.

    Args:
        model: The float64 CPU model.
        split: The split (its test rows are evaluated).
        kl_draws: ``z ~ q`` draws per x for the Monte Carlo exact KL (arm V).
        lat_draws: ``z ~ q`` draws per x for the latitude marginal.
        grid_nodes: Midpoint nodes in ``w'`` for the model posterior's normaliser.
        chunk: Observations per pass.

    Returns:
        ``(arrays, extra)``: per-x numpy arrays and scalar facts (axis, sign, sigma).
    """
    x, z_true = split.test_x, split.test_z
    if model.arm == "V":
        sign = _axis_sign(model, split)
        axis = sign * model.u.detach()
        sigma = float(model.learned_scale())
        nodes = -1 + (torch.arange(grid_nodes, dtype=torch.float64) + 0.5) * (
            2.0 / grid_nodes
        )
        g_nodes = model.g_of_latitude(nodes)
    else:
        sign, axis, sigma = 1.0, _AXIS, float("nan")
        nodes = g_nodes = torch.zeros(0)
    columns: dict[str, list[Tensor]] = {}
    for start in range(0, x.shape[0], chunk):
        sl = slice(start, start + chunk)
        xs, zs = x[sl], z_true[sl]
        posterior = model.posterior(xs)
        pieces: dict[str, Tensor] = {}
        if model.arm == "V":
            pieces["kl"], pieces["kl_se"] = _exact_kl(
                model, posterior, xs, g_nodes, sigma, grid_nodes, kl_draws
            )
        else:
            pieces["nll"] = -posterior.log_prob(zs)
        pieces.update(_fitted_parameters(posterior, axis))
        pieces.update(_latitude_check(posterior, axis, zs, lat_draws))
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["itd_us"] = split.test_itd
    arrays["z_true"] = z_true.numpy()
    arrays["weight"] = split.test_weight
    arrays["subject"] = split.test_subject
    extra: dict[str, Any] = {
        "axis": axis.tolist(),
        "axis_sign": float(sign),
        "sigma": sigma,
    }
    if model.arm == "V":
        extra["u_abs_cos_y"] = float(abs(model.u.detach()[1]))
    return arrays, extra


def oracle_baselines(split: HrtfSplit) -> dict[str, float]:
    """The gate note's uniform / oracle ring / oracle cap NLLs on this split (nats).

    ITD quantile bins (50) and the ring / cap fits come from the training subjects, the
    NLLs are the area-weighted means over the held-out subjects, with
    :mod:`apps.eval.hrtf_itd_gate`'s own functions.
    """
    z_tr = split.train_z.double().numpy()
    z_te = split.test_z.numpy()
    edges = _fit_bins(split.train_itd)
    bin_tr = _assign(split.train_itd, edges)
    bin_te = _assign(split.test_itd, edges)
    ring = np.zeros(len(z_te))
    cap = np.zeros(len(z_te))
    for b in range(len(edges) - 1):
        tr, te = bin_tr == b, bin_te == b
        if te.any():
            ring[te] = _ring_nll(z_tr[tr], split.train_weight[tr], z_te[te])
            cap[te] = _cap_nll(z_tr[tr], split.train_weight[tr], z_te[te])
    weight = split.test_weight
    return {
        "uniform": math.log(4 * math.pi),
        "oracle_ring": float((ring * weight).sum()),
        "oracle_cap": float((cap * weight).sum()),
    }


def summarize_run(
    arrays: dict[str, np.ndarray], extra: dict[str, Any], split: HrtfSplit
) -> dict[str, Any]:
    """Scalar summaries (area-weighted unless stated) of one run's per-x arrays."""
    weight = arrays["weight"]
    arm_v = "kl" in arrays
    key = "kl" if arm_v else "nll"
    value = arrays[key]
    summary: dict[str, Any] = {
        "metric_name": "kl" if arm_v else "nll",
        "metric": float((value * weight).sum()),
        "metric_unweighted": float(value.mean()),
        "metric_median": float(np.median(value)),
        "median_abs_cos": float(np.median(arrays["abs_cos"])),
        "aligned": bool(np.median(arrays["abs_cos"]) >= ALIGNED_COS),
        "cov50": float((arrays["in50"] * weight).sum()),
        "cov95": float((arrays["in95"] * weight).sum()),
        "median_abs_error": float((arrays["abs_err"] * weight).sum()),
        "median_abs_error_median": float(np.median(arrays["abs_err"])),
        "baselines": oracle_baselines(split),
        **extra,
    }
    if arm_v:
        summary["metric_mc_se"] = float(
            np.sqrt(((arrays["kl_se"] * weight) ** 2).sum())
        )
    for name in ("p", "q", "epsilon", "kappa"):
        values = arrays[name]
        if np.isfinite(values).any():
            summary[f"{name}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        summary["m_median"] = float(np.nanmedian(arrays["epsilon"])) - 1.0
    return summary


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = _load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    print(
        "Note: the 10 split replicates overlap in subjects (each of 18 subjects is "
        "held out in about 1/6 of the splits and trains in the rest), so the Wilcoxon "
        "p-values are optimistic; the range over splits accompanies every effect.\n"
    )
    for estimator in ESTIMATORS:
        mine: Cell = {(k[1], k[2]): v for k, v in runs.items() if k[0] == estimator}
        if not mine:
            continue
        label = "primary" if estimator == "xcorr" else "sensitivity"
        print(f"\n# Estimator: {estimator} ({label})\n")
        _print_main_tables(mine)
        print("\n## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)\n")
        tests = _confirmatory_tests(mine)
        _print_confirmatory(tests)
        print("\n## Descriptive comparisons (no p-values)\n")
        _print_descriptive(mine)


# ----------------------------------------------------------------------------- helpers

Runs = dict[tuple[str, str, str], dict[int, dict[str, Any]]]
Cell = dict[tuple[str, str], dict[int, dict[str, Any]]]


def _axis_sign(model: HrtfModel, split: HrtfSplit) -> float:
    """+1 or -1: the sign of ``u`` making ``u . z_true`` positively correlated with x.

    Chosen on the *training* subjects (area-weighted correlation); labels never enter
    training.
    """
    u = model.u.detach()
    w = (split.train_z.double() * u).sum(-1).numpy()
    x = split.train_x[:, 0].double().numpy()
    weight = split.train_weight
    cov = np.average(
        (w - np.average(w, weights=weight)) * (x - np.average(x, weights=weight)),
        weights=weight,
    )
    return 1.0 if cov >= 0 else -1.0


def _exact_kl(
    model: HrtfModel,
    posterior: Distribution,
    x: Tensor,
    g_nodes: Tensor,
    sigma: float,
    grid_nodes: int,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    """Monte Carlo ``KL(q || p_model(. | x))`` with the exact ``log q``.

    ``p_model(z | x) = exp(-(x - g(u . z))^2 / 2 sigma^2) / Z'`` with ``Z' = 2 pi *
    integral over w' in [-1, 1] of exp(-(x - g(w'))^2 / 2 sigma^2)`` by midpoint
    quadrature (a uniform point on S^2 has uniform ``w'``).
    """
    energy_nodes = (x - g_nodes[None]).pow(2) / (2 * sigma**2)  # (b, nodes)
    log_z = math.log(2 * math.pi) + torch.logsumexp(
        -energy_nodes + math.log(2.0 / grid_nodes), dim=-1
    )
    z, log_q = sample_with_log_prob(posterior, torch.Size([num_samples]))
    assert log_q is not None
    mean = model.decode(z)[..., 0]  # (draws, b)
    log_p = -((x[:, 0] - mean).pow(2)) / (2 * sigma**2) - log_z
    integrand = log_q - log_p
    kl = integrand.mean(0)
    se = (integrand.var(0, unbiased=False).clamp_min(0) / num_samples).sqrt()
    return kl, se


def _fitted_parameters(posterior: Distribution, axis: Tensor) -> dict[str, Tensor]:
    """Per-x fitted parameters (NaN if absent), ``mu`` and ``|cos(mu, axis)|``."""
    nan = torch.full(posterior.batch_shape, float("nan"), dtype=axis.dtype)
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


def _latitude_check(
    posterior: Distribution, axis: Tensor, z_true: Tensor, draws: int
) -> dict[str, Tensor]:
    """The true ``axis . z`` against q's latitude marginal (``draws`` draws per x)."""
    z = posterior.rsample(torch.Size([draws]))
    lat = (z * axis).sum(-1)  # (draws, b)
    levels = torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975], dtype=lat.dtype)
    q = torch.quantile(lat.T.contiguous(), levels, dim=-1)  # (5, b)
    w_true = (z_true * axis).sum(-1)
    return {
        "w_true": w_true,
        "lat_q025": q[0],
        "lat_q25": q[1],
        "lat_q50": q[2],
        "lat_q75": q[3],
        "lat_q975": q[4],
        "in50": ((w_true >= q[1]) & (w_true <= q[3])).double(),
        "in95": ((w_true >= q[0]) & (w_true <= q[4])).double(),
        "abs_err": (q[2] - w_true).abs(),
    }


def _load_runs(out_dir: Path) -> Runs:
    runs: Runs = {}
    for path in sorted(out_dir.glob("*/arm*/*/s*.json")):
        record = json.loads(path.read_text())
        key = (record["estimator"], record["arm"], record["family"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _print_main_tables(runs: Cell) -> None:
    for arm in ARMS:
        families = {f: runs.get((arm, f), {}) for f in FAMILIES}
        if not any(families.values()):
            continue
        name = (
            "Arm V (VAE; primary metric: exact KL(q || p_model(.|x)), nats)"
            if arm == "V"
            else "Arm L (supervised; primary metric: held-out -log q(z_true|x), nats)"
        )
        print(f"\n## {name}\n")
        print(
            "| family | splits | mean +- sd over splits | median | range | "
            "aligned splits (median abs cos >= 0.99) | median abs cos (range) | "
            + ("|u_y| median (range) | " if arm == "V" else "")
            + "p / q / eps / m medians |"
        )
        print("|" + "---|" * (9 if arm == "V" else 8))
        for family, seeds in families.items():
            if not seeds:
                continue
            m = np.array([s["metric"] for s in seeds.values()])
            cos = np.array([s["median_abs_cos"] for s in seeds.values()])
            aligned = sum(s["aligned"] for s in seeds.values())
            params = " / ".join(
                f"{np.median([s[k] for s in seeds.values() if k in s]):.3g}"
                if any(k in s for s in seeds.values())
                else "-"
                for k in ("p_median", "q_median", "epsilon_median", "m_median")
            )
            row = (
                f"| {family} | {len(seeds)} | {m.mean():.3f} +- {_sd(m):.3f} | "
                f"{np.median(m):.3f} | {m.min():.3f} to {m.max():.3f} | "
                f"{aligned}/{len(seeds)} | {np.median(cos):.3f} "
                f"({cos.min():.3f}-{cos.max():.3f}) | "
            )
            if arm == "V":
                uy = np.array([s["u_abs_cos_y"] for s in seeds.values()])
                row += f"{np.median(uy):.3f} ({uy.min():.3f}-{uy.max():.3f}) | "
            print(row + params + " |")
        _print_per_split(arm, families)
        _print_latitude(arm, families)
    _print_baselines(runs)


def _print_per_split(arm: str, families: dict[str, dict[int, dict[str, Any]]]) -> None:
    present = [f for f, r in families.items() if r]
    print(f"\nPer split (arm {arm}): metric / median abs cos\n")
    print("| split | test subjects | " + " | ".join(present) + " |")
    print("|" + "---|" * (2 + len(present)))
    seeds = sorted({s for runs in families.values() for s in runs})
    for seed in seeds:
        cells = []
        subjects = ""
        for family in present:
            r = families[family].get(seed)
            if r is not None:
                subjects = ",".join(f"H{s}" for s in r["test_subjects"])
            cells.append(
                "-" if r is None else f"{r['metric']:.3f} / {r['median_abs_cos']:.3f}"
            )
        print(f"| {seed} | {subjects} | " + " | ".join(cells) + " |")
    if arm == "V":
        print("\nPer split (arm V): |u_y| = |cos(u, +/- y)|\n")
        print("| split | " + " | ".join(present) + " |")
        print("|" + "---|" * (1 + len(present)))
        for seed in seeds:
            cells = [
                "-"
                if families[f].get(seed) is None
                else f"{families[f][seed]['u_abs_cos_y']:.3f}"
                for f in present
            ]
            print(f"| {seed} | " + " | ".join(cells) + " |")


def _print_latitude(arm: str, families: dict[str, dict[int, dict[str, Any]]]) -> None:
    frame = "about the signed decoder axis u" if arm == "V" else "about +y"
    print(
        f"\nLabel-based latitude check, arm {arm} ({frame}; mean over splits; nominal "
        "coverage 0.50 / 0.95)\n"
    )
    print(
        "| family | central 50% coverage | central 95% coverage | "
        "mean abs error of the marginal's median (w units) | median abs error |"
    )
    print("|" + "---|" * 5)
    for family, seeds in families.items():
        if not seeds:
            continue
        mean = lambda k, s=seeds: np.mean([r[k] for r in s.values()])  # noqa: E731
        print(
            f"| {family} | {mean('cov50'):.3f} | {mean('cov95'):.3f} | "
            f"{mean('median_abs_error'):.4f} | {mean('median_abs_error_median'):.4f} |"
        )


def _print_baselines(runs: Cell) -> None:
    seeds: dict[int, dict[str, Any]] = {}
    for group in runs.values():
        for seed, record in group.items():
            seeds.setdefault(seed, record)
    if not seeds:
        return
    print(
        "\n## Held-out NLL of the true direction under the gate baselines "
        "(nats, same splits; for the scale of arm L)\n"
    )
    print("| baseline | mean over splits | range |")
    print("|---|---|---|")
    for key in ("uniform", "oracle_ring", "oracle_cap"):
        values = np.array([s["baselines"][key] for s in seeds.values()])
        low, high = values.min(), values.max()
        print(f"| {key} | {values.mean():.3f} | {low:.3f} to {high:.3f} |")
    for arm in ARMS:
        for family in FAMILIES:
            group = runs.get((arm, family), {})
            if arm == "L" and group:
                values = np.array([s["metric"] for s in group.values()])
                print(
                    f"| arm L {family} | {values.mean():.3f} | "
                    f"{values.min():.3f} to {values.max():.3f} |"
                )


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
    runs: Cell,
) -> list[dict[str, Any]]:
    tests = []
    for arm in ARMS:
        tnb = runs.get((arm, "tnbbeta"), {})
        for other in PRIMARY_COMPARATORS:
            cap = runs.get((arm, other), {})
            seeds = sorted(set(tnb) & set(cap))
            if len(seeds) < 5:
                continue
            diff = np.array([cap[s]["metric"] - tnb[s]["metric"] for s in seeds])
            tests.append(
                {
                    "arm": arm,
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
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)
    return tests


def _print_confirmatory(tests: list[dict[str, Any]]) -> None:
    print(
        "| arm | comparator | n | mean paired diff (comparator - tnbbeta) | "
        "range over splits | splits tnbbeta better | p (scipy) | p (enumerated) | "
        "Holm threshold | Holm reject | margin >= 0.25 |"
    )
    print("|" + "---|" * 11)
    for test in sorted(tests, key=lambda t: (t["arm"], t["other"])):
        diff = test["diff"]
        wins = int((diff > 0).sum())
        print(
            f"| {test['arm']} | {test['other']} | {test['n']} | {diff.mean():.3f} | "
            f"{diff.min():.3f} to {diff.max():.3f} | {wins}/{test['n']} | "
            f"{test['p']:.5f} | {test['p_enum']:.5f} | {test['threshold']:.5f} | "
            f"{test['holm_reject']} | {test['margin']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for arm {test['arm']}")
    complete = len(tests) == len(ARMS) * len(PRIMARY_COMPARATORS)
    holm = complete and all(t["holm_reject"] for t in tests)
    margin = complete and all(t["margin"] for t in tests)
    print(
        "\nDecision rule 'TNBBeta uses its ring capacity on this task' (all four tests "
        f"survive Holm and the 0.25-nat margin holds): Holm all = {holm}, margin all = "
        f"{margin}, rule met = {holm and margin}"
        + ("" if complete else " (fewer than four tests available)")
        + "\n"
    )


def _print_descriptive(runs: Cell) -> None:
    print(
        "| arm | comparison | n | mean metric (A) | mean metric (B) | mean paired diff "
        "(A - B) | range of diff | splits A lower |"
    )
    print("|" + "---|" * 8)
    pairs = [
        ("tnbbeta", "tnbbeta_init0"),
        ("tnbbeta_init0", "vmf"),
        ("tnbbeta_init0", "power_spherical"),
        ("vmf", "power_spherical"),
    ]
    for arm in ARMS:
        for a, b in pairs:
            ra, rb = runs.get((arm, a), {}), runs.get((arm, b), {})
            seeds = sorted(set(ra) & set(rb))
            if not seeds:
                continue
            ka = np.array([ra[s]["metric"] for s in seeds])
            kb = np.array([rb[s]["metric"] for s in seeds])
            d = ka - kb
            print(
                f"| {arm} | {a} vs {b} | {len(seeds)} | {ka.mean():.3f} | "
                f"{kb.mean():.3f} | {d.mean():.3f} | {d.min():.3f} to {d.max():.3f} | "
                f"{int((ka < kb).sum())}/{len(seeds)} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
