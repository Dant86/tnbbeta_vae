"""Single-beacon ranging: can TNBBeta represent / optimise / amortise a latitude ring?

The task (``tnbbeta_vae.data.ring_posterior``) has an exact posterior that is a latitude
ring about a fixed axis ``a``, so vMF and Power Spherical (unimodal caps) cannot
match it.
Design and decision rules: ``writeup/results/ring_posterior_design_2026-11-01.md``.

Usage (one cell / run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.ring_recovery stage1 --dim 8 --latitude 60 \
        --sigma 0.2 --out-dir ring_recovery/stage1
    uv run python -m apps.synthetic.ring_recovery stage2 --dim 8 --steps 2000 \
        --out-dir ring_recovery/stage2
    uv run python -m apps.synthetic.ring_recovery stage3 --dim 8 --condition true \
        --family tnbbeta --seed 0 --out-dir ring_recovery/stage3
    uv run python -m apps.synthetic.ring_recovery summarize --out-dir ring_recovery

Stage 1 (representation) minimises ``KL(q || p_true)`` over each family; stage 2 runs
per-datum gradient ascent on the exact ELBO with the true decoder; stage 3 trains an MLP
encoder (true or learned decoder). ``summarize`` prints markdown tables and the paired
tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
from scipy import optimize, stats
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence

from tnbbeta_vae.data.ring_posterior import (
    RingPosteriorTask,
    exact_kl,
    kl_uniform_to_posterior,
    ring_task,
    sphere_log_area,
)
from tnbbeta_vae.distributions import (
    HypersphericalUniform,
    PowerSpherical,
    TNBBetaSpherical,
    VonMisesFisher,
)
from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.heads import (
    P_RAW_INIT,
    epsilon_from_m,
    head_size,
    posterior_from_raw,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.training import select_device

PRIMARY_SIGMA = 0.2
FAMILIES = ("vmf", "power_spherical", "tnbbeta")
STAGE3_VARIANTS = (
    "vmf",
    "power_spherical",
    "tnbbeta",
    "tnbbeta_fixed_m_neg",
    "tnbbeta_fixed_m_ring",
    "tnbbeta_p_init",
)
X1_NATS = 0.5
X2_NATS = 0.25
X3_NATS = 0.25
_POLE_DEG = 10.0
_NUM_BINS = 90
_P_CLAMP = 1e-6
_MAX_KAPPA_SEARCH = 3e4


def main(argv: list[str] | None = None) -> None:
    """Dispatches to a stage.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("stage1")
    s1.add_argument("--dim", type=int, required=True)
    s1.add_argument("--latitude", type=float, required=True, help="degrees")
    s1.add_argument("--sigma", type=float, required=True)
    s1.add_argument("--seed", type=int, default=0)
    s1.add_argument("--out-dir", type=Path, required=True)
    s2 = sub.add_parser("stage2")
    s2.add_argument("--dim", type=int, required=True)
    s2.add_argument("--steps", type=int, default=2000)
    s2.add_argument("--num-x", type=int, default=300)
    s2.add_argument("--restarts", type=int, default=8)
    s2.add_argument("--out-dir", type=Path, required=True)
    s3 = sub.add_parser("stage3")
    s3.add_argument("--dim", type=int, required=True)
    s3.add_argument("--condition", choices=["true", "learned"], required=True)
    s3.add_argument("--family", choices=list(STAGE3_VARIANTS), required=True)
    s3.add_argument("--seed", type=int, required=True)
    s3.add_argument("--epochs", type=int, default=100)
    s3.add_argument("--num-train", type=int, default=20_000)
    s3.add_argument("--num-test", type=int, default=2_000)
    s3.add_argument(
        "--num-elbo-samples",
        type=int,
        default=4,
        help="z draws per x in the training ELBO (reconstruction and MC KL).",
    )
    s3.add_argument(
        "--mc-kl",
        action="store_true",
        help="Train vMF/PS with a Monte Carlo KL to the uniform prior (as TNBBeta "
        "does) instead of the analytic one; a no-op for TNBBeta variants.",
    )
    s3.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear KL warm-up length; default epochs // 10 (the original runs).",
    )
    s3.add_argument(
        "--epsilon-floor",
        type=float,
        default=1e-4,
        help="TNBBeta head's smallest epsilon (MlpVAEConfig.epsilon_floor); the "
        "default 1e-4 is the original head. Use a separate --out-dir per value.",
    )
    s3.add_argument(
        "--device",
        default=None,
        help="Training device; default picks CUDA and exits 75 on a broken GPU node.",
    )
    s3.add_argument("--out-dir", type=Path, required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.stage == "stage1":
        run_stage1(args)
    elif args.stage == "stage2":
        run_stage2(args)
    elif args.stage == "stage3":
        run_stage3(args)
    else:
        summarize(args.out_dir)


# --------------------------------------------------------------------------- stage 1


def run_stage1(args: argparse.Namespace) -> None:
    """Representation: best exact KL per family for one (dim, latitude, sigma) cell."""
    torch.set_default_dtype(torch.float64)
    task = RingPosteriorTask(args.dim, args.sigma, _axis_at(args.dim, 0.0))
    x = task.features(
        torch.tensor([math.cos(math.radians(args.latitude))], dtype=torch.float64)
    )
    torch.manual_seed(args.seed)
    truth = _truth_summary(task, x)
    result: dict[str, Any] = {
        "dim": args.dim,
        "latitude": args.latitude,
        "sigma": args.sigma,
        "truth": truth,
        "uniform_kl": float(kl_uniform_to_posterior(task, x)[0]),
    }
    axis_fit = _fit_tnbbeta_on_axis(task, x, args.latitude, truth)
    result["tnbbeta_mu_fixed"] = axis_fit
    result["tnbbeta_mu_free"] = _fit_tnbbeta_free(task, x, axis_fit, truth)
    for family in ("vmf", "power_spherical"):
        result[family] = _fit_cap(task, x, family, truth)
    best_tnb = min(
        result["tnbbeta_mu_fixed"]["best"]["kl"],
        result["tnbbeta_mu_free"]["best"]["kl"],
    )
    best_cap = min(result["vmf"]["best"]["kl"], result["power_spherical"]["best"]["kl"])
    tnb_member = (
        result["tnbbeta_mu_fixed"]["best"]
        if result["tnbbeta_mu_fixed"]["best"]["kl"]
        <= result["tnbbeta_mu_free"]["best"]["kl"]
        else result["tnbbeta_mu_free"]["best"]
    )
    ring_shaped = tnb_member["band_mass"] >= 0.8 and tnb_member["tv"] < 0.25
    result["summary"] = {
        "best_tnbbeta_kl": best_tnb,
        "best_cap_kl": best_cap,
        "gap": best_cap - best_tnb,
        "ring_shaped": bool(ring_shaped),
        "ring_only": bool(tnb_member["m"] > 0 and tnb_member["pole_mass"] < 0.01),
        "pass": bool(best_cap - best_tnb >= X1_NATS and ring_shaped),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"d{args.dim}_t{int(args.latitude)}_s{args.sigma}.json"
    (args.out_dir / name).write_text(json.dumps(result, indent=1))
    print(name, json.dumps(result["summary"]))


def _axis_at(dim: int, alpha: float) -> Tensor:
    """Unit vector at angle ``alpha`` from e_1 in the (e_1, e_2) plane, float64."""
    axis = torch.zeros(dim, dtype=torch.float64)
    axis[0] = math.cos(alpha)
    axis[1] = math.sin(alpha)
    return axis


def _truth_summary(task: RingPosteriorTask, x: Tensor) -> dict[str, float]:
    """Latitude summary of the exact posterior for a single ``x`` (batch of one)."""
    lo, hi, mean, sd = _truth_stats(task, x)
    return {
        "band_lo_deg": math.degrees(float(lo[0])),
        "band_hi_deg": math.degrees(float(hi[0])),
        "mean_theta_deg": math.degrees(float(mean[0])),
        "sd_theta_deg": math.degrees(float(sd[0])),
    }


def _truth_stats(
    task: RingPosteriorTask, x: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-x central-95% latitude interval, mean and sd of the truth (radians)."""
    theta, density = task.latitude_density(x)
    delta = math.pi / task.grid_size
    cdf = torch.cumsum(density * delta, dim=-1)
    lo = _quantile(theta, cdf, 0.025)
    hi = _quantile(theta, cdf, 0.975)
    mean = (density * theta * delta).sum(-1)
    var = (density * (theta - mean[:, None]).pow(2) * delta).sum(-1)
    return lo, hi, mean, var.sqrt()


def _quantile(theta: Tensor, cdf: Tensor, prob: float) -> Tensor:
    index = (cdf < prob).sum(-1).clamp(max=theta.shape[0] - 1)
    return theta[index]


def _fit_tnbbeta_on_axis(
    task: RingPosteriorTask, x: Tensor, latitude: float, truth: dict[str, float]
) -> dict[str, Any]:
    """Multi-start Adam on the quadrature KL for TNBBeta with mu = the axis."""
    dim = task.dim
    theta, delta = _grid(task)
    _, density = task.latitude_density(x)
    log_p = density[0].clamp_min(1e-300).log()
    starts = _tnbbeta_starts(dim, latitude, num_random=85)
    raw = starts.clone().requires_grad_(True)
    steps = 3000
    optimiser = torch.optim.Adam([raw], lr=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=(0.04) ** (1 / steps)
    )
    for _ in range(steps):
        optimiser.zero_grad()
        kl = _tnbbeta_quadrature_kl(raw, theta, delta, log_p)
        torch.nan_to_num(kl, nan=1e6, posinf=1e6).sum().backward()
        optimiser.step()
        scheduler.step()
    with torch.no_grad():
        final = torch.nan_to_num(
            _tnbbeta_quadrature_kl(raw, theta, delta, log_p), nan=1e6, posinf=1e6
        )
    order = final.argsort()
    candidates = []
    for index in order[:5].tolist():
        p, q, eps = _unpack_tnbbeta(raw[index].detach())
        member = _evaluate_tnbbeta(task, x, 0.0, p, q, eps, truth)
        member["kl_quadrature"] = float(final[index])
        candidates.append(member)
    best = min(candidates, key=lambda c: c["kl"])
    return {
        "best": best,
        "top5_quadrature_kl": [float(final[i]) for i in order[:5]],
        "starts_within_0.05_of_best": int((final <= final.min() + 0.05).sum()),
        "num_starts": len(final),
        "candidates": candidates,
        "best_raw": [r.detach().tolist() for r in raw[order[:3]]],
    }


def _grid(task: RingPosteriorTask) -> tuple[Tensor, float]:
    delta = math.pi / task.grid_size
    return (torch.arange(task.grid_size, dtype=torch.float64) + 0.5) * delta, delta


def _unpack_tnbbeta(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(logit p, logit q, log eps) -> (p, q, eps), clamped like the model head."""
    p = raw[..., 0].sigmoid().clamp(_P_CLAMP, 1 - _P_CLAMP)
    q = raw[..., 1].sigmoid().clamp(_P_CLAMP, 1 - _P_CLAMP)
    eps = raw[..., 2].exp().clamp(1e-3, 1e5)
    return p, q, eps


def _tnbbeta_quadrature_kl(
    raw: Tensor, theta: Tensor, delta: float, log_p: Tensor
) -> Tensor:
    """KL(q || p_true) by quadrature over the latitude for TNBBeta(mu = axis)."""
    p, q, eps = _unpack_tnbbeta(raw)
    log_q = _tnbbeta_log_latitude_density(p, q, eps, theta)
    density = log_q.exp()
    kl = (density * (log_q - log_p) * delta).sum(-1)
    # A spike narrower than the grid spacing is invisible to the rule (mass << 1) and
    # would otherwise show a spuriously tiny KL; penalise any quadrature mass deficit.
    mass_error = (density.sum(-1) * delta - 1).abs()
    return kl + 1e3 * (mass_error - 0.005).clamp_min(0)


def _tnbbeta_log_latitude_density(
    p: Tensor, q: Tensor, eps: Tensor, theta: Tensor
) -> Tensor:
    """``log`` density of the polar angle ``t`` for TNBBetaSpherical(mu=axis).

    ``h(t) = f_w(cos t) sin t`` with ``f_w`` the rescaled univariate density (the
    sphere's ``sin^(d-2)`` weight cancels the Jacobian, so this does not depend on d).
    """
    w = torch.cos(theta)
    y = ((w + 1) / 2).expand(p.shape[0], -1)
    log_f = TNBBetaUnivariate(p[:, None], q[:, None], eps[:, None]).log_prob(y)
    return log_f - math.log(2) + torch.log(torch.sin(theta))


def _tnbbeta_starts(dim: int, latitude: float, num_random: int) -> Tensor:
    """Raw (logit p, logit q, log eps) starts: random, ring-like and trimodal-scaled."""
    generator = torch.Generator().manual_seed(1234)
    rows: list[list[float]] = []
    for _ in range(num_random):
        rows.append(
            [
                float(torch.empty(1).uniform_(-4, 4, generator=generator)),
                float(torch.empty(1).uniform_(-8, 8, generator=generator)),
                float(
                    torch.empty(1).uniform_(
                        math.log(0.05), math.log(3000.0), generator=generator
                    )
                ),
            ]
        )
    p_ring = (1 + math.cos(math.radians(latitude))) / 2
    logit = math.log(p_ring / (1 - p_ring))
    for q_value in (1e-6, 0.9):
        for eps in (1.0, 5.0, 20.0, 100.0, 500.0):
            rows.append([logit, math.log(q_value / (1 - q_value)), math.log(eps)])
    # The certified trimodal member (0.85, 0.95, 0.3) of the expressivity note, d = 3,
    # with eps rescaled so that m / ((d - 1) / 2) matches.
    rows.append(
        [math.log(0.85 / 0.15), math.log(0.95 / 0.05), math.log(0.3 * (dim - 1) / 2)]
    )
    return torch.tensor(rows, dtype=torch.float64)


def _evaluate_tnbbeta(
    task: RingPosteriorTask,
    x: Tensor,
    alpha: float,
    p: Tensor,
    q: Tensor,
    eps: Tensor,
    truth: dict[str, float],
    num_samples: int = 400_000,
) -> dict[str, Any]:
    """Monte Carlo evaluation of TNBBeta(e_1, p, q, eps) against a truth at alpha."""
    mean_direction = torch.zeros(1, task.dim, dtype=torch.float64)
    mean_direction[0, 0] = 1.0
    distribution = TNBBetaSpherical(
        mean_direction, p.reshape(1), q.reshape(1), eps.reshape(1)
    )
    rotated = RingPosteriorTask(task.dim, task.sigma, _axis_at(task.dim, alpha))
    member = _evaluate_q(distribution, rotated, x, num_samples)
    member.update(
        {
            "p": float(p),
            "q": float(q),
            "epsilon": float(eps),
            "m": float(eps) - (task.dim - 1) / 2,
            "alpha_deg": math.degrees(alpha),
        }
    )
    return member


@torch.no_grad()
def _evaluate_q(
    distribution: Distribution,
    task: RingPosteriorTask,
    x: Tensor,
    num_samples: int,
    chunk: int = 100_000,
) -> dict[str, Any]:
    """KL, TV, band/pole mass and latitude moments of one ``q`` versus the truth."""
    lo, hi, _, _ = _truth_stats(task, x)
    counts = torch.zeros(_NUM_BINS, dtype=torch.float64)
    total = total_sq = 0.0
    band = pole = theta_sum = theta_sq = 0.0
    done = 0
    pole_rad = math.radians(_POLE_DEG)
    while done < num_samples:
        count = min(chunk, num_samples - done)
        z, log_q = sample_with_log_prob(distribution, torch.Size([count]))
        assert log_q is not None
        integrand = log_q - task.log_posterior(x, z)
        total += float(integrand.sum())
        total_sq += float(integrand.pow(2).sum())
        angle = torch.arccos((z @ task.axis).clamp(-1, 1)).reshape(-1)
        counts += torch.histc(angle, _NUM_BINS, 0.0, math.pi).to(torch.float64)
        band += float(((angle >= lo) & (angle <= hi)).sum())
        pole += float(((angle < pole_rad) | (angle > math.pi - pole_rad)).sum())
        theta_sum += float(angle.sum())
        theta_sq += float(angle.pow(2).sum())
        done += count
    edges = torch.linspace(0, math.pi, _NUM_BINS + 1, dtype=torch.float64)
    true_mass = torch.diff(task.latitude_cdf(x, edges)[0])
    tv = 0.5 * float((counts / counts.sum() - true_mass).abs().sum())
    mean = total / done
    se = math.sqrt(max(total_sq / done - mean**2, 0.0) / done)
    theta_mean = theta_sum / done
    return {
        "kl": mean,
        "kl_se": se,
        "tv": tv,
        "band_mass": band / done,
        "pole_mass": pole / done,
        "mean_theta_deg": math.degrees(theta_mean),
        "sd_theta_deg": math.degrees(
            math.sqrt(max(theta_sq / done - theta_mean**2, 0))
        ),
    }


def _fit_tnbbeta_free(
    task: RingPosteriorTask,
    x: Tensor,
    axis_fit: dict[str, Any],
    truth: dict[str, float],
) -> dict[str, Any]:
    """Monte Carlo Adam over (alpha, p, q, eps): mu need not equal the axis."""
    dim = task.dim
    seeds = torch.tensor(axis_fit["best_raw"], dtype=torch.float64)
    alphas = [0.1, 0.5, 1.0, 2.0]
    rows = []
    for seed_row in seeds:
        for alpha in alphas:
            rows.append([alpha, *seed_row.tolist()])
    generator = torch.Generator().manual_seed(99)
    for _ in range(20):
        rows.append(
            [
                float(torch.empty(1).uniform_(0, math.pi, generator=generator)),
                float(torch.empty(1).uniform_(-4, 4, generator=generator)),
                float(torch.empty(1).uniform_(-8, 8, generator=generator)),
                float(
                    torch.empty(1).uniform_(
                        math.log(0.05), math.log(3000.0), generator=generator
                    )
                ),
            ]
        )
    params = torch.tensor(rows, dtype=torch.float64).requires_grad_(True)
    num = params.shape[0]
    mean_direction = torch.zeros(num, dim, dtype=torch.float64)
    mean_direction[:, 0] = 1.0
    steps = 1200
    optimiser = torch.optim.Adam([params], lr=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=(0.05) ** (1 / steps)
    )
    for _ in range(steps):
        optimiser.zero_grad()
        p, q, eps = _unpack_tnbbeta(params[:, 1:])
        distribution = TNBBetaSpherical(mean_direction, p, q, eps)
        z, log_q = sample_with_log_prob(distribution, torch.Size([2048]))
        assert log_q is not None
        w = z[..., 0] * torch.cos(params[:, 0]) + z[..., 1] * torch.sin(params[:, 0])
        loss = (log_q - task.log_posterior_at_latitude(x, w)).mean(0)
        torch.nan_to_num(loss, nan=1e6, posinf=1e6).sum().backward()
        optimiser.step()
        scheduler.step()
    with torch.no_grad():
        p, q, eps = _unpack_tnbbeta(params[:, 1:])
        coarse = []
        for index in range(num):
            member = _evaluate_tnbbeta(
                task,
                x,
                float(params[index, 0]),
                p[index],
                q[index],
                eps[index],
                truth,
                num_samples=20_000,
            )
            coarse.append(member["kl"])
        order = np.argsort(np.nan_to_num(coarse, nan=1e6))
        finals = [
            _evaluate_tnbbeta(task, x, float(params[i, 0]), p[i], q[i], eps[i], truth)
            for i in order[:3]
        ]
    return {"best": min(finals, key=lambda c: c["kl"]), "candidates": finals}


def _fit_cap(
    task: RingPosteriorTask, x: Tensor, family: str, truth: dict[str, float]
) -> dict[str, Any]:
    """Grid + Nelder-Mead over (alpha, log kappa) for vMF or Power Spherical."""
    dim = task.dim

    def make(kappa: float) -> Distribution:
        mu = torch.zeros(1, dim, dtype=torch.float64)
        mu[0, 0] = 1.0
        if family == "vmf":
            return VonMisesFisher(mu, torch.full((1, 1), kappa, dtype=torch.float64))
        return PowerSpherical(mu, torch.full((1,), kappa, dtype=torch.float64))

    def objective(params: np.ndarray, num: int, seed: int) -> float:
        alpha = float(np.clip(params[0], 0.0, math.pi))
        log_kappa = float(
            np.clip(params[1], math.log(0.05), math.log(_MAX_KAPPA_SEARCH))
        )
        rotated = RingPosteriorTask(dim, task.sigma, _axis_at(dim, alpha))
        torch.manual_seed(seed)
        kl, _ = exact_kl(make(math.exp(log_kappa)), rotated, x, num_samples=num)
        value = float(kl[0])
        return value if math.isfinite(value) else 1e6

    alphas = np.linspace(0, math.pi, 19)
    log_kappas = np.linspace(math.log(0.1), math.log(_MAX_KAPPA_SEARCH), 25)
    grid = np.array(
        [
            [objective(np.array([a, k]), 20_000, 5), a, k]
            for a in alphas
            for k in log_kappas
        ]
    )
    top = grid[np.argsort(grid[:, 0])[:3]]
    refined = []
    for row in top:
        res = optimize.minimize(
            objective,
            x0=row[1:],
            args=(40_000, 6),
            method="Nelder-Mead",
            options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-3},
        )
        refined.append(res.x)
    finals = []
    for params in refined:
        alpha = float(np.clip(params[0], 0.0, math.pi))
        kappa = math.exp(
            float(np.clip(params[1], math.log(0.05), math.log(_MAX_KAPPA_SEARCH)))
        )
        rotated = RingPosteriorTask(dim, task.sigma, _axis_at(dim, alpha))
        torch.manual_seed(7)
        member = _evaluate_q(make(kappa), rotated, x, 400_000)
        member.update({"kappa": kappa, "alpha_deg": math.degrees(alpha)})
        finals.append(member)
    uniform = grid[:, 0].min()
    return {
        "best": min(finals, key=lambda c: c["kl"]),
        "candidates": finals,
        "grid_best_kl": float(uniform),
    }


# --------------------------------------------------------------------------- stage 2


def run_stage2(args: argparse.Namespace) -> None:
    """Per-datum optimisation of the exact ELBO with the true decoder."""
    task = ring_task(args.dim, PRIMARY_SIGMA)
    generator = torch.Generator().manual_seed(777)
    x, _ = task.sample(args.num_x, generator)
    log_evidence = task.log_evidence(x.double()).float()
    result: dict[str, Any] = {
        "dim": args.dim,
        "steps": args.steps,
        "num_x": args.num_x,
        "uniform_kl_mean": float(kl_uniform_to_posterior(task, x).mean()),
        "families": {},
    }
    for family in FAMILIES:
        for init, restarts, scale in (
            ("amortised_like", 1, 0.1),
            ("random_restarts", args.restarts, 2.0),
        ):
            torch.manual_seed(0)
            raw, curve = _optimise_per_datum(
                family, task, x, restarts, scale, args.steps
            )
            summary = _evaluate_posteriors(
                lambda sl, raw=raw, family=family: posterior_from_raw(
                    cast("Any", family), raw[sl], task.dim
                ),
                task,
                x,
                log_evidence,
                family,
            )
            summary["loss_curve"] = curve
            result["families"][f"{family}/{init}"] = summary
            print(args.dim, args.steps, family, init, _short(summary), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"d{args.dim}_steps{args.steps}.json").write_text(
        json.dumps(result, indent=1)
    )


def _short(summary: dict[str, Any]) -> str:
    return (
        f"KL mean {summary['kl_mean']:.3f} median {summary['kl_median']:.3f} "
        f"band {summary['band_mass_mean']:.2f} pole {summary['pole_mass_mean']:.3f}"
    )


def _optimise_per_datum(
    family: str,
    task: RingPosteriorTask,
    x: Tensor,
    restarts: int,
    scale: float,
    steps: int,
    lr: float = 0.05,
    num_samples: int = 8,
) -> tuple[Tensor, list[float]]:
    """Adam on each datum's raw head vector; returns the best restart's raw."""
    num = x.shape[0]
    x_rep = x.repeat_interleave(restarts, dim=0)
    raw = scale * torch.randn(num * restarts, head_size(cast("Any", family), task.dim))
    raw.requires_grad_(True)
    optimiser = torch.optim.Adam([raw], lr=lr)
    curve: list[float] = []
    for step in range(steps):
        optimiser.zero_grad()
        negative_elbo = -_elbo_estimate(family, task, x_rep, raw, num_samples)
        negative_elbo.sum().backward()
        optimiser.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append(float(negative_elbo.detach().mean()))
    with torch.no_grad():
        elbo = _elbo_estimate(family, task, x_rep, raw, 64).reshape(num, restarts)
        best = elbo.argmax(dim=1)
    chosen = raw.detach().reshape(num, restarts, -1)[torch.arange(num), best]
    return chosen, curve


def _elbo_estimate(
    family: str, task: RingPosteriorTask, x: Tensor, raw: Tensor, num_samples: int
) -> Tensor:
    """Per-row ELBO ``E_q[log p(x|z)] - KL(q || Uniform)``, reparameterised."""
    posterior = posterior_from_raw(cast("Any", family), raw, task.dim)
    if family == "tnbbeta":
        z, log_q = sample_with_log_prob(posterior, torch.Size([num_samples]))
        assert log_q is not None
        reconstruction = task.log_likelihood(x, z).mean(0)
        kl = (log_q + sphere_log_area(task.dim)).mean(0)
    else:
        z = posterior.rsample(torch.Size([num_samples]))
        reconstruction = task.log_likelihood(x, z).mean(0)
        kl = kl_divergence(posterior, HypersphericalUniform(task.dim - 1))
    return reconstruction - kl


@torch.no_grad()
def _evaluate_posteriors(
    make_posterior: Any,
    task: RingPosteriorTask,
    x: Tensor,
    log_evidence: Tensor,
    family: str,
    num_samples: int = 2000,
    chunk: int = 250,
) -> dict[str, Any]:
    """Exact KL, ELBO gap, ring-shape statistics and TNBBeta parameters over many x."""
    lo, hi, truth_mean, truth_sd = _truth_stats(task, x)
    pole_rad = math.radians(_POLE_DEG)
    columns: dict[str, list[Tensor]] = {
        k: [] for k in ("kl", "kl_se", "elbo", "band", "pole", "mean", "sd", "param")
    }
    extras: dict[str, list[Tensor]] = {}
    for start in range(0, x.shape[0], chunk):
        sl = slice(start, start + chunk)
        posterior = make_posterior(sl)
        xs = x[sl]
        kl, se = exact_kl(posterior, task, xs, num_samples)
        with torch.no_grad():
            z, log_q = sample_with_log_prob(posterior, torch.Size([500]))
            assert log_q is not None
            elbo = (
                task.log_likelihood(xs, z) - log_q - sphere_log_area(task.dim)
            ).mean(0)
            angle = torch.arccos((z @ task.axis).clamp(-1, 1))
        columns["kl"].append(kl)
        columns["kl_se"].append(se)
        columns["elbo"].append(elbo)
        columns["band"].append(((angle >= lo[sl]) & (angle <= hi[sl])).float().mean(0))
        columns["pole"].append(
            ((angle < pole_rad) | (angle > math.pi - pole_rad)).float().mean(0)
        )
        columns["mean"].append(angle.mean(0))
        columns["sd"].append(angle.std(0))
        for key, value in _distribution_params(posterior, task).items():
            extras.setdefault(key, []).append(value)
    cat = {k: torch.cat(v) for k, v in columns.items() if v}
    gap = log_evidence - cat["elbo"]
    summary: dict[str, Any] = {
        "family": family,
        "kl_mean": float(cat["kl"].mean()),
        "kl_median": float(cat["kl"].median()),
        "kl_sd": float(cat["kl"].std()),
        "kl_se_mean": float(cat["kl_se"].mean()),
        "elbo_gap_mean_mc": float(gap.mean()),
        "band_mass_mean": float(cat["band"].mean()),
        "frac_band_ge_0.8": float((cat["band"] >= 0.8).float().mean()),
        "pole_mass_mean": float(cat["pole"].mean()),
        "mean_theta_error_deg": math.degrees(
            float((cat["mean"] - truth_mean).abs().mean())
        ),
        "sd_theta_ratio": float((cat["sd"] / truth_sd).median()),
        "kl_per_x": [float(v) for v in cat["kl"]],
    }
    for key, values in extras.items():
        merged = torch.cat(values)
        summary[f"{key}_mean"] = float(merged.mean())
        summary[f"{key}_median"] = float(merged.median())
        if key == "m":
            summary["frac_m_negative"] = float((merged < 0).float().mean())
    return summary


@torch.no_grad()
def _distribution_params(
    posterior: Distribution, task: RingPosteriorTask
) -> dict[str, Tensor]:
    """Per-x scalar parameters of a posterior batch (for reporting)."""
    if hasattr(posterior, "epsilon"):
        direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
        return {
            "p": posterior.p,  # pyright: ignore[reportAttributeAccessIssue]
            "q": posterior.q,  # pyright: ignore[reportAttributeAccessIssue]
            "epsilon": posterior.epsilon,  # pyright: ignore[reportAttributeAccessIssue]
            "m": posterior.epsilon - (task.dim - 1) / 2,  # pyright: ignore[reportAttributeAccessIssue]
            "abs_cos_mu_axis": (direction @ task.axis).abs(),
        }
    if hasattr(posterior, "kappa"):
        return {
            "kappa": posterior.kappa,  # pyright: ignore[reportAttributeAccessIssue]
            "abs_cos_mu_axis": (posterior.mean_direction @ task.axis).abs(),  # pyright: ignore[reportAttributeAccessIssue]
        }
    return {
        "kappa": posterior.scale[..., 0],  # pyright: ignore[reportAttributeAccessIssue]
        "abs_cos_mu_axis": (posterior.loc @ task.axis).abs(),  # pyright: ignore[reportAttributeAccessIssue]
    }


# --------------------------------------------------------------------------- stage 3


class _TrueDecoder(nn.Module):
    """The task's true decoder ``z -> g(a . z)`` (no parameters)."""

    def __init__(self, task: RingPosteriorTask) -> None:
        super().__init__()
        self.task = task

    def forward(self, z: Tensor) -> Tensor:
        return self.task.decode(z)


def run_stage3(args: argparse.Namespace) -> None:
    """Amortised training of one (dim, condition, family, seed)."""
    device = select_device(args.device)
    task = ring_task(args.dim, PRIMARY_SIGMA)
    data_gen = torch.Generator().manual_seed(100 + args.seed)
    train_x, _ = task.sample(args.num_train, data_gen)
    test_gen = torch.Generator().manual_seed(424242)
    test_x, _ = task.sample(args.num_test, test_gen)
    fixed_epsilon = _stage3_fixed_epsilon(args.family, task)
    family = "tnbbeta" if args.family.startswith("tnbbeta") else args.family
    torch.manual_seed(args.seed)
    model = MlpVAE(
        MlpVAEConfig(
            family=cast("Any", family),
            input_dim=3,
            hidden_dims=[128, 128],
            latent_dim=args.dim,
            likelihood_scale=PRIMARY_SIGMA,
            num_elbo_samples=args.num_elbo_samples,
            fixed_epsilon=fixed_epsilon,
            # 'tnbbeta' and the fixed-m variants keep the original init (offset 0.0)
            # so the pre-registered L3 runs stay reproducible; only 'tnbbeta_p_init'
            # (exploratory, added after the L3 results) uses the new raw-p offset.
            p_init_offset=P_RAW_INIT if args.family == "tnbbeta_p_init" else 0.0,
            epsilon_floor=args.epsilon_floor,
        )
    )
    if args.condition == "true":
        model.decoder = cast("Any", _TrueDecoder(task))
        model.learned_scale.log_variance.requires_grad_(False)
    model.to(device)
    train_x = train_x.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=1e-3)
    shuffle = torch.Generator().manual_seed(200 + args.seed)
    warmup = max(
        1, args.epochs // 10 if args.warmup_epochs is None else args.warmup_epochs
    )
    losses = []
    for epoch in range(args.epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(args.num_train, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for batch in train_x[order].split(256):
            optimiser.zero_grad()
            if args.mc_kl and family != "tnbbeta":
                loss = _mc_kl_loss(model, batch, kl_weight)
            else:
                loss = model.training_step(batch, kl_weight=kl_weight)["loss"]
            loss.backward()
            optimiser.step()
            running += loss.detach() * len(batch)
        losses.append(float(running) / args.num_train)
    model.cpu().eval()
    log_evidence = task.log_evidence(test_x.double()).float()
    torch.manual_seed(31337)
    summary = _evaluate_posteriors(
        lambda sl: model.posterior_and_prior(test_x[sl])[0],
        task,
        test_x,
        log_evidence,
        family,
        num_samples=1000,
    )
    summary.update(_model_elbo(model, task, test_x, log_evidence, args.condition))
    summary["representative"] = _representative(model, task)
    summary.update(
        {
            "dim": args.dim,
            "condition": args.condition,
            "variant": args.family,
            "seed": args.seed,
            "fixed_epsilon": fixed_epsilon,
            "epsilon_floor": args.epsilon_floor,
            "num_elbo_samples": args.num_elbo_samples,
            "mc_kl": bool(args.mc_kl),
            "epochs": args.epochs,
            "loss_curve_full": losses,
            "final_train_loss": losses[-1],
            "learned_sigma": float(model.learned_scale()),
            "loss_curve": losses[:: max(1, args.epochs // 20)],
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"d{args.dim}_{args.condition}_{args.family}_s{args.seed}.json"
    (args.out_dir / name).write_text(json.dumps(summary))
    (args.out_dir / "ckpt").mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "ckpt" / name.replace(".json", ".pt"))
    print(
        name, _short(summary), f"ELBO gap {summary['elbo_gap_model']:.3f}", flush=True
    )


def _mc_kl_loss(model: MlpVAE, batch: Tensor, kl_weight: float) -> Tensor:
    """``MlpVAE.training_step``'s loss with a Monte Carlo (not analytic) KL."""
    posterior, prior = model.posterior_and_prior(batch)
    terms = monte_carlo_elbo(
        batch,
        posterior,
        prior,
        model.decoder,
        model.learned_scale(),
        model.config.num_elbo_samples,
        analytic_kl=False,
    )
    return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()


def _stage3_fixed_epsilon(variant: str, task: RingPosteriorTask) -> float | None:
    if variant == "tnbbeta_fixed_m_neg":
        return epsilon_from_m(-0.5, task.dim)
    if variant == "tnbbeta_fixed_m_ring":
        generator = torch.Generator().manual_seed(555)
        x, _ = task.sample(5000, generator)
        theta, density = task.latitude_density(x.double())
        delta = math.pi / task.grid_size
        w = torch.cos(theta)
        mean = (density * w * delta).sum(-1)
        var = (density * (w - mean[:, None]).pow(2) * delta).sum(-1)
        return float((1 / var.median() - 1) / 2)
    return None


@torch.no_grad()
def _model_elbo(
    model: MlpVAE,
    task: RingPosteriorTask,
    x: Tensor,
    log_evidence: Tensor,
    condition: str,
) -> dict[str, float]:
    """The model's own ELBO on the test set and its gap to the true ``log p(x)``."""
    values, recon, kl_prior = [], [], []
    for start in range(0, x.shape[0], 250):
        xs = x[start : start + 250]
        posterior, _ = model.posterior_and_prior(xs)
        z, log_q = sample_with_log_prob(posterior, torch.Size([200]))
        assert log_q is not None
        reconstruction = model.log_likelihood(xs, z).mean(0)
        kl = (log_q + sphere_log_area(task.dim)).mean(0)
        values.append(reconstruction - kl)
        recon.append(reconstruction)
        kl_prior.append(kl)
    elbo_all = torch.cat(values)
    return {
        "elbo_model": float(elbo_all.mean()),
        "elbo_gap_model": float((log_evidence - elbo_all).mean()),
        "recon_model": float(torch.cat(recon).mean()),
        "kl_prior_model_mc": float(torch.cat(kl_prior).mean()),
    }


@torch.no_grad()
def _representative(model: MlpVAE, task: RingPosteriorTask) -> list[dict[str, float]]:
    """The fitted latitude marginal at x* = g(cos t*) for t* = 30, 60, 80 degrees."""
    rows = []
    for degrees in (30.0, 60.0, 80.0):
        x = task.features(torch.tensor([math.cos(math.radians(degrees))]))[None]
        x = x.reshape(1, 3)
        posterior, _ = model.posterior_and_prior(x)
        stats_ = _evaluate_q_single(posterior, task, x)
        stats_["latitude"] = degrees
        rows.append(stats_)
    return rows


def _evaluate_q_single(
    posterior: Distribution, task: RingPosteriorTask, x: Tensor
) -> dict[str, float]:
    torch.manual_seed(0)
    axis = task.axis.double()
    task64 = RingPosteriorTask(task.dim, task.sigma, axis)
    z = posterior.rsample(torch.Size([20_000]))[:, 0].double()
    angle = torch.arccos((z @ axis).clamp(-1, 1))
    lo, hi, mean, sd = _truth_stats(task64, x.double())
    result = {
        "mean_theta_deg": math.degrees(float(angle.mean())),
        "sd_theta_deg": math.degrees(float(angle.std())),
        "true_mean_theta_deg": math.degrees(float(mean[0])),
        "true_sd_theta_deg": math.degrees(float(sd[0])),
        "band_mass": float(((angle >= lo) & (angle <= hi)).double().mean()),
        "pole_mass": float(
            (
                (angle < math.radians(_POLE_DEG))
                | (angle > math.pi - math.radians(_POLE_DEG))
            )
            .double()
            .mean()
        ),
    }
    if hasattr(posterior, "epsilon"):
        result["epsilon"] = float(posterior.epsilon[0])  # pyright: ignore[reportAttributeAccessIssue]
        result["p"] = float(posterior.p[0])  # pyright: ignore[reportAttributeAccessIssue]
        result["q"] = float(posterior.q[0])  # pyright: ignore[reportAttributeAccessIssue]
    return result


# --------------------------------------------------------------------------- summary


def summarize(out_dir: Path) -> None:
    """Prints markdown tables for every stage found under ``out_dir``."""
    stage1 = sorted((out_dir / "stage1").glob("*.json"))
    if stage1:
        _summarize_stage1([json.loads(p.read_text()) for p in stage1])
    stage2 = sorted((out_dir / "stage2").glob("*.json"))
    if stage2:
        _summarize_stage2([json.loads(p.read_text()) for p in stage2])
    stage3 = sorted((out_dir / "stage3").glob("*.json"))
    if stage3:
        _summarize_stage3([json.loads(p.read_text()) for p in stage3])


def _summarize_stage1(cells: list[dict[str, Any]]) -> None:
    print("\n## Stage 1\n")
    print(
        "| d | lat | sigma | truth mean/sd (deg) | KL uniform | TNB mu=a | "
        "TNB mu free | "
        "vMF | PS | gap | ring-shaped | band | TV | pole | m | ring-only | pass |"
    )
    print("|" + "---|" * 17)
    cells.sort(key=lambda c: (c["sigma"], c["dim"], c["latitude"]))
    for c in cells:
        s = c["summary"]
        fixed, free = c["tnbbeta_mu_fixed"]["best"], c["tnbbeta_mu_free"]["best"]
        tnb = fixed if fixed["kl"] <= free["kl"] else free
        print(
            f"| {c['dim']} | {c['latitude']:.0f} | {c['sigma']} | "
            f"{c['truth']['mean_theta_deg']:.1f}/{c['truth']['sd_theta_deg']:.1f} | "
            f"{c['uniform_kl']:.2f} | {fixed['kl']:.3f} | {free['kl']:.3f} | "
            f"{c['vmf']['best']['kl']:.2f} | "
            f"{c['power_spherical']['best']['kl']:.2f} | "
            f"{s['gap']:.2f} | {s['ring_shaped']} | {tnb['band_mass']:.2f} | "
            f"{tnb['tv']:.3f} | {tnb['pole_mass']:.3f} | {tnb['m']:.2f} | "
            f"{s['ring_only']} | {s['pass']} |"
        )
    passed = sum(c["summary"]["pass"] for c in cells)
    print(f"\n{passed} of {len(cells)} cells pass (X1 = {X1_NATS} nats).")


def _summarize_stage2(runs: list[dict[str, Any]]) -> None:
    print("\n## Stage 2\n")
    print(
        "| d | steps | family | init | KL mean | KL median | gap_mc | band>=0.8 | "
        "pole | eps med | m med | frac m<0 |"
    )
    print("|" + "---|" * 12)
    for run in sorted(runs, key=lambda r: (r["dim"], r["steps"])):
        for key, s in run["families"].items():
            family, init = key.split("/")
            print(
                f"| {run['dim']} | {run['steps']} | {family} | {init} | "
                f"{s['kl_mean']:.3f} | {s['kl_median']:.3f} | "
                f"{s['elbo_gap_mean_mc']:.3f} | "
                f"{s['frac_band_ge_0.8']:.2f} | {s['pole_mass_mean']:.3f} | "
                f"{s.get('epsilon_median', float('nan')):.2f} | "
                f"{s.get('m_median', float('nan')):.2f} | "
                f"{s.get('frac_m_negative', float('nan')):.2f} |"
            )
    print("\nLevel-2 decision (amortised-like init, per d and step budget):\n")
    for run in sorted(runs, key=lambda r: (r["dim"], r["steps"])):
        fam = run["families"]
        tnb = fam["tnbbeta/amortised_like"]["kl_median"]
        cap = min(
            fam["vmf/amortised_like"]["kl_median"],
            fam["power_spherical/amortised_like"]["kl_median"],
        )
        verdict = tnb <= cap - X2_NATS and tnb < 1.0
        print(
            f"* d={run['dim']} steps={run['steps']}: TNBBeta median {tnb:.3f}, "
            f"best cap "
            f"median {cap:.3f}, difference {cap - tnb:.3f} -> "
            f"{'PASS' if verdict else 'FAIL'}"
        )
        a = np.array(fam["tnbbeta/amortised_like"]["kl_per_x"])
        for other in ("vmf", "power_spherical"):
            b = np.array(fam[f"{other}/amortised_like"]["kl_per_x"])
            print(
                f"    paired over x vs {other}: TNBBeta better on "
                f"{np.mean(a < b):.2f} of x; "
                f"Wilcoxon one-sided p = "
                f"{_wilcoxon_greater(b - a):.2g}"
            )


def _summarize_stage3(runs: list[dict[str, Any]]) -> None:
    print("\n## Stage 3\n")
    keys = sorted({(r["dim"], r["condition"], r["variant"]) for r in runs})
    table: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = {}
    for r in runs:
        table.setdefault((r["dim"], r["condition"], r["variant"]), {})[r["seed"]] = r
    print(
        "| d | cond | variant | n | KL mean +- sd | gap +- sd | ELBO +- sd | "
        "band>=0.8 | "
        "pole | eps med | m med | frac m<0 | |cos(mu,a)| |"
    )
    print("|" + "---|" * 13)
    for key in keys:
        seeds = table[key]
        kls = np.array([s["kl_mean"] for s in seeds.values()])
        gaps = np.array([s["elbo_gap_model"] for s in seeds.values()])
        elbos = np.array([s["elbo_model"] for s in seeds.values()])

        def avg(name: str, seeds: dict[int, dict[str, Any]] = seeds) -> float:
            return float(np.mean([s.get(name, float("nan")) for s in seeds.values()]))

        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(seeds)} | "
            f"{kls.mean():.3f} +- {kls.std(ddof=1) if len(kls) > 1 else 0:.3f} | "
            f"{gaps.mean():.3f} +- {gaps.std(ddof=1) if len(gaps) > 1 else 0:.3f} | "
            f"{elbos.mean():.3f} +- {elbos.std(ddof=1) if len(elbos) > 1 else 0:.3f} | "
            f"{avg('frac_band_ge_0.8'):.2f} | {avg('pole_mass_mean'):.3f} | "
            f"{avg('epsilon_median'):.2f} | {avg('m_median'):.2f} | "
            f"{avg('frac_m_negative'):.2f} | {avg('abs_cos_mu_axis_mean'):.3f} |"
        )
    print("\n### Paired one-sided Wilcoxon, TNBBeta free vs caps (Holm over all)\n")
    tests = []
    for dim in sorted({k[0] for k in keys}):
        for condition in ("true", "learned"):
            tnb = table.get((dim, condition, "tnbbeta"), {})
            for other in ("vmf", "power_spherical"):
                cap = table.get((dim, condition, other), {})
                seeds = sorted(set(tnb) & set(cap))
                if len(seeds) < 5:
                    continue
                metric = "kl_mean" if condition == "true" else "elbo_gap_model"
                diff = np.array([cap[s][metric] - tnb[s][metric] for s in seeds])
                p = _wilcoxon_greater(diff)
                tests.append((dim, condition, other, metric, len(seeds), diff, p))
    order = np.argsort([t[6] for t in tests])
    holm_rejected = True
    print(
        "| d | cond | vs | metric | n | mean paired diff (cap - TNB) | p | Holm thr | "
        "margin >= 0.25 | reject |"
    )
    print("|" + "---|" * 10)
    for rank, index in enumerate(order):
        dim, condition, other, metric, n, diff, p = tests[index]
        threshold = 0.05 / (len(tests) - rank)
        holm_rejected = holm_rejected and p <= threshold
        print(
            f"| {dim} | {condition} | {other} | {metric} | {n} | {diff.mean():.3f} | "
            f"{p:.4f} | {threshold:.4f} | {diff.mean() >= X3_NATS} | {holm_rejected} |"
        )
    print("\n### Representative x (seed-averaged), true-decoder condition\n")
    print(
        "| d | variant | t* | truth mean/sd | q mean/sd | band | pole | eps | p | q |"
    )
    print("|" + "---|" * 10)
    for key in keys:
        if key[1] != "true":
            continue
        seeds = table[key]
        for i, degrees in enumerate((30.0, 60.0, 80.0)):
            rows = [s["representative"][i] for s in seeds.values()]

            def mean(name: str, rows: list[dict[str, float]] = rows) -> float:
                return float(np.mean([r.get(name, float("nan")) for r in rows]))

            print(
                f"| {key[0]} | {key[2]} | {degrees:.0f} | "
                f"{mean('true_mean_theta_deg'):.1f}/{mean('true_sd_theta_deg'):.1f} | "
                f"{mean('mean_theta_deg'):.1f}/{mean('sd_theta_deg'):.1f} | "
                f"{mean('band_mass'):.2f} | {mean('pole_mass'):.3f} | "
                f"{mean('epsilon'):.2f} | {mean('p'):.2f} | {mean('q'):.2f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:])


def _wilcoxon_greater(diff: np.ndarray) -> float:
    """One-sided (greater) Wilcoxon signed-rank p-value of paired differences."""
    return float(cast("Any", stats.wilcoxon(diff, alternative="greater")).pvalue)
