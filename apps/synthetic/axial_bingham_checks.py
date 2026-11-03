"""Pre-training checks of the axial Bingham study (design constants, no networks).

Pre-registration: ``writeup/results/axial_bingham_preregistration_2026-11-02.md``.

Subcommands (each writes ``<out-dir>/<name>.json`` and prints markdown):

* ``threshold``: the detectability transition of ``X = z z^T + sigma E`` (``E`` iid
  N(0,1), ``d x d``, only ``X_sym`` matters), by simulation of the top eigenvalue and
  eigenvector overlap at several ``d``, and the exact posterior's axial concentration
  ``E_post[(v1 . z)^2]`` at ``d = 2, 3, 4``.
* ``normaliser``: accuracy of the Bingham normaliser (quadrature refinement, Monte
  Carlo) and of the sphere quadrature used for the learned decoders' exact evidence.
* ``representability``: does TNBBetaSpherical contain / approximate the rank-one Bingham
  density ``exp(kappa (mu . z)^2)``? Best-fit ``KL(q || Bingham)`` over (p, q, epsilon),
  and the same for vMF / Power Spherical and their sign-symmetrised versions.
* ``anisotropy``: the exact best axially symmetric ``KL(q || Bingham(A))`` (any axis,
  any radial profile) at the study's sigmas, and the one-sided (hemisphere) value.

Usage::

    uv run python -m apps.synthetic.axial_bingham_checks threshold --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import optimize, special
import torch

from tnbbeta_vae.data.axial_bingham import (
    AxialBinghamTask,
    bingham_log_normaliser,
    sphere_quadrature,
)
from tnbbeta_vae.distributions import PowerSpherical, TNBBetaSpherical, VonMisesFisher

SIGMAS_D = (0.1, 0.2, 0.4, 0.8)
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
_LATITUDE_NODES = 20_000


def main(argv: list[str] | None = None) -> None:
    """Runs one check.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "threshold",
            "normaliser",
            "representability",
            "representability_general",
            "anisotropy",
        ],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS_D))
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result: Any
    if args.command == "threshold":
        result = threshold_check()
    elif args.command == "normaliser":
        result = normaliser_check()
    elif args.command == "representability":
        result = representability_check()
    elif args.command == "representability_general":
        result = representability_general_check()
    else:
        result = anisotropy_check(args.sigmas)
    (args.out_dir / f"{args.command}.json").write_text(json.dumps(result))


# --------------------------------------------------------------------------- threshold


def threshold_check() -> dict[str, Any]:
    """Top eigenvalue / overlap simulations and the exact posterior's axis
    concentration."""
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"eigen": [], "posterior": []}
    print("\n### Spike recovery by X_sym's top eigenvector (simulation, 200 reps)\n")
    print(
        "| d | sigma sqrt(d) | mean top eig of X_sym | BBP edge-based prediction | "
        "mean (v1.z)^2 | prediction 1 - sigma^2 d / 2 |"
    )
    print("|" + "---|" * 6)
    for dim in (2, 3, 4, 10, 50, 200):
        reps = 200 if dim <= 50 else 40
        for ratio in (0.25, 0.5, 0.75, 1.0, 1.25, 1.414, 1.75, 2.0, 3.0):
            sigma = ratio / math.sqrt(dim)
            tops, overlaps = [], []
            for _ in range(reps):
                z = rng.normal(size=dim)
                z /= np.linalg.norm(z)
                x = np.outer(z, z) + sigma * rng.normal(size=(dim, dim))
                values, vectors = np.linalg.eigh((x + x.T) / 2)
                tops.append(values[-1])
                overlaps.append(float((vectors[:, -1] @ z) ** 2))
            theta, edge = 1.0, sigma * math.sqrt(2 * dim)
            above = theta > edge / 2
            pred_top = theta + (edge / 2) ** 2 / theta if above else edge
            pred_overlap = max(0.0, 1 - (edge / 2) ** 2 / theta**2)
            row = {
                "dim": dim,
                "ratio": ratio,
                "sigma": sigma,
                "top_eig": float(np.mean(tops)),
                "pred_top": pred_top,
                "overlap": float(np.mean(overlaps)),
                "pred_overlap": pred_overlap,
            }
            out["eigen"].append(row)
            print(
                f"| {dim} | {ratio:.3f} | {row['top_eig']:.3f} | {pred_top:.3f} | "
                f"{row['overlap']:.3f} | {pred_overlap:.3f} |"
            )
    print(
        "\n### Exact posterior axis concentration E_post[(v1 . z)^2] (median over X)\n"
    )
    print(
        "| d | sigma | sigma sqrt(d) | median E_post[(v1.z)^2] | 10th pct | "
        "90th pct | 1/d |"
    )
    print("|" + "---|" * 7)
    for dim in (2, 3, 4):
        for sigma in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5):
            gen = torch.Generator().manual_seed(7)
            task = AxialBinghamTask(dim, sigma)
            x, _ = task.sample(400, gen)
            conc = axis_concentration(task, x.double())
            q10, q50, q90 = np.percentile(conc, [10, 50, 90])
            out["posterior"].append(
                {
                    "dim": dim,
                    "sigma": sigma,
                    "median": float(q50),
                    "p10": float(q10),
                    "p90": float(q90),
                }
            )
            print(
                f"| {dim} | {sigma} | {sigma * math.sqrt(dim):.3f} | {q50:.3f} | "
                f"{q10:.3f} | {q90:.3f} | {1 / dim:.3f} |"
            )
    return out


def axis_concentration(task: AxialBinghamTask, x: torch.Tensor) -> np.ndarray:
    """``E_post[(v1 . z)^2] = d log Z / d lam_1`` (central difference), per ``x``."""
    eigenvalues = torch.linalg.eigvalsh(task.bingham_matrix(x))
    step = 1e-3
    up, down = eigenvalues.clone(), eigenvalues.clone()
    up[:, -1] += step
    down[:, -1] -= step
    return (
        (bingham_log_normaliser(up) - bingham_log_normaliser(down)) / (2 * step)
    ).numpy()


# -------------------------------------------------------------------------- normaliser


def normaliser_check() -> dict[str, Any]:
    """Quadrature accuracy of the Bingham normaliser and the sphere grids."""
    torch.manual_seed(0)
    out: dict[str, Any] = {"rows": []}
    print("\n### Bingham normaliser: quadrature refinement and Monte Carlo (float64)\n")
    print(
        "| d | sigma | max abs diff 2000 vs 8000 nodes | "
        "max abs diff sphere grid x1 vs x2 | "
        "max abs diff vs Monte Carlo (2M draws, 20 X) |"
    )
    print("|" + "---|" * 5)
    for dim in (2, 3, 4):
        for sigma in SIGMAS_D:
            task = AxialBinghamTask(dim, sigma)
            gen = torch.Generator().manual_seed(11)
            x, _ = task.sample(200, gen)
            eig = torch.linalg.eigvalsh(task.bingham_matrix(x.double()))
            coarse = bingham_log_normaliser(eig, 2000)
            fine = bingham_log_normaliser(eig, 8000)
            gl = float((coarse - fine).abs().max())
            grid = float("nan")
            if dim < 4:
                grid = float(
                    (_grid_normaliser(eig[:20], 1) - _grid_normaliser(eig[:20], 2))
                    .abs()
                    .max()
                )
            mc_diff = float("nan")
            if sigma >= 0.4:  # Monte Carlo is useless on peaked integrands
                z = torch.randn(2_000_000, dim, dtype=torch.float64)
                z /= z.norm(dim=-1, keepdim=True)
                mc = torch.stack(
                    [
                        torch.logsumexp((z**2 * lam).sum(-1), 0) - math.log(len(z))
                        for lam in eig[:20]
                    ]
                )
                mc_diff = float((mc - coarse[:20]).abs().max())
            if dim == 4:  # independent brute-force 3-D quadrature in Hopf coordinates
                grid = float((_hopf_normaliser(eig[:3]) - coarse[:3]).abs().max())
            out["rows"].append(
                {"dim": dim, "sigma": sigma, "gl": gl, "grid": grid, "mc": mc_diff}
            )
            print(f"| {dim} | {sigma} | {gl:.2e} | {grid:.2e} | {mc_diff:.2e} |")
    return out


def _hopf_normaliser(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Brute-force ``d = 4`` normaliser: Gauss-Legendre in ``s`` times trapezoid in
    (alpha, beta)."""
    nodes, weights = np.polynomial.legendre.leggauss(300)
    s = torch.tensor((nodes + 1) / 2, dtype=torch.float64)
    log_w = torch.tensor(np.log(weights / 2), dtype=torch.float64)
    count = 256
    angle = 2 * math.pi * (torch.arange(count, dtype=torch.float64) + 0.5) / count
    ca, sa = angle.cos() ** 2, angle.sin() ** 2
    out = []
    for lam in eigenvalues:
        first = lam[3] * ca + lam[2] * sa  # largest two eigenvalues (ascending input)
        second = lam[1] * ca + lam[0] * sa
        expo = (
            s[:, None, None] * first[None, :, None]
            + (1 - s)[:, None, None] * second[None, None, :]
        )
        total = expo + log_w[:, None, None] - 2 * math.log(count)
        out.append(torch.logsumexp(total.reshape(-1), 0))
    return torch.stack(out)


def _grid_normaliser(eigenvalues: torch.Tensor, size: int) -> torch.Tensor:
    """The sphere-quadrature version of the Bingham normaliser in the eigenbasis."""
    z, log_w = sphere_quadrature(eigenvalues.shape[-1], size)
    return torch.logsumexp(
        (z[None] ** 2 * eigenvalues[:, None]).sum(-1) + log_w, dim=-1
    )


# ----------------------------------------------------------------------
# representability


def representability_check() -> dict[str, Any]:
    """Best-fit ``KL(q || Bingham)`` for TNBBeta, vMF, PS and symmetrised versions."""
    out: dict[str, Any] = {"rows": []}
    print("\n### Best-fit KL(q || rank-one Bingham exp(kappa (mu.z)^2)) in nats\n")
    print(
        "| d | kappa | TNBBeta free (p,q,eps) | fit p, q, eps, m | TNBBeta p=0.5 | "
        "vMF | PS | sym vMF | sym PS | ln 2 |"
    )
    print("|" + "---|" * 10)
    for dim in (2, 3, 4):
        grid = _Latitude(dim)
        for kappa in KAPPAS:
            log_target = grid.log_bingham(kappa)
            free = _fit_tnbbeta(grid, log_target, fixed_half=False)
            half = _fit_tnbbeta(grid, log_target, fixed_half=True)
            vmf = _fit_cap(grid, log_target, "vmf", False)
            ps = _fit_cap(grid, log_target, "ps", False)
            svmf = _fit_cap(grid, log_target, "vmf", True)
            sps = _fit_cap(grid, log_target, "ps", True)
            p, q, eps = free["params"]
            m = eps - (dim - 1) / 2
            out["rows"].append(
                {
                    "dim": dim,
                    "kappa": kappa,
                    "tnb": free["kl"],
                    "tnb_params": free["params"],
                    "m": m,
                    "tnb_half": half["kl"],
                    "tnb_half_params": half["params"],
                    "vmf": vmf["kl"],
                    "ps": ps["kl"],
                    "svmf": svmf["kl"],
                    "sps": sps["kl"],
                }
            )
            print(
                f"| {dim} | {kappa:g} | {free['kl']:.2e} | "
                f"{p:.3f}, {q:.3f}, {eps:.3f}, {m:.3f} | {half['kl']:.2e} | "
                f"{vmf['kl']:.3f} | {ps['kl']:.3f} | {svmf['kl']:.2e} | "
                f"{sps['kl']:.2e} | {math.log(2):.3f} |"
            )
    return out


class _Latitude:
    """Latitude quadrature on ``S^(d-1)`` about a pole (nodes clustered at both
    poles)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        count = _LATITUDE_NODES
        j = (torch.arange(count, dtype=torch.float64) + 0.5) / count
        self.theta = math.pi * (1 - torch.cos(math.pi * j)) / 2
        self.dtheta = (math.pi**2 / 2) * torch.sin(math.pi * j) / count
        pole = torch.zeros(dim, dtype=torch.float64)
        pole[0] = 1.0
        self.pole = pole
        self.points = torch.zeros(count, dim, dtype=torch.float64)
        self.points[:, 0] = self.theta.cos()
        self.points[:, 1] = self.theta.sin()
        area = 2 * math.pi ** ((dim - 1) / 2) / math.gamma((dim - 1) / 2)
        self.log_measure = (
            math.log(area) + (dim - 2) * self.theta.sin().log() + self.dtheta.log()
        )

    def log_bingham(self, kappa: float) -> torch.Tensor:
        """Surface-measure log density of ``exp(kappa w^2) / Z`` at the nodes."""
        eigenvalues = torch.zeros(1, self.dim, dtype=torch.float64)
        eigenvalues[0, -1] = kappa
        log_z = bingham_log_normaliser(eigenvalues)[0]
        log_area = (
            math.log(2) + (self.dim / 2) * math.log(math.pi) - math.lgamma(self.dim / 2)
        )
        return kappa * self.theta.cos() ** 2 - log_z - log_area

    def kl(self, log_q: torch.Tensor, log_target: torch.Tensor) -> float:
        """``KL(q || target)`` from surface-measure log densities at the nodes."""
        weights = (log_q + self.log_measure).exp()
        total = float(weights.sum())
        if not 0.999 < total < 1.001 or not torch.isfinite(log_q).all():
            return float("inf")
        return float((weights * (log_q - log_target)).sum() / total)


def _tnb_log_q(grid: _Latitude, p: float, q: float, eps: float) -> torch.Tensor:
    dist = TNBBetaSpherical(
        grid.pole, torch.tensor(p), torch.tensor(q), torch.tensor(eps)
    )
    return dist.log_prob(grid.points)


def _fit_tnbbeta(
    grid: _Latitude, log_target: torch.Tensor, fixed_half: bool
) -> dict[str, Any]:
    """Minimises ``KL(q || target)`` over (p, q, epsilon) from several starts."""

    def unpack(v: np.ndarray) -> tuple[float, float, float]:
        p = 0.5 if fixed_half else float(special.expit(v[0]))
        return p, float(special.expit(v[1])), float(math.exp(v[2]))

    def objective(v: np.ndarray) -> float:
        p, q, eps = unpack(v)
        if not (1e-6 < q < 1 - 1e-6 and 1e-3 < eps < 50):
            return 1e3
        p = min(max(p, 1e-6), 1 - 1e-6)
        value = grid.kl(_tnb_log_q(grid, p, q, eps), log_target)
        return value if math.isfinite(value) else 1e3

    best: dict[str, Any] = {"kl": float("inf"), "params": [0.5, 0.0, 1.0]}
    rng = np.random.default_rng(0)
    starts = [
        [special.logit(pp), special.logit(qq), math.log(ee)]
        for pp in (0.5, 0.9, 0.99)
        for qq in (1e-3, 0.3, 0.9, 0.999)
        for ee in (0.1, 0.4, 1.0, 3.0)
    ]
    starts += [
        [rng.normal(), rng.normal() * 2 - 1, rng.normal() * 1.2] for _ in range(12)
    ]
    for start in starts:
        result = optimize.minimize(
            objective, np.array(start), method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 400},
        )  # fmt: skip
        if result.fun < best["kl"]:
            best = {"kl": float(result.fun), "params": list(unpack(result.x))}
    return best


def _fit_cap(
    grid: _Latitude, log_target: torch.Tensor, family: str, symmetrise: bool
) -> dict[str, Any]:
    """Best ``KL`` of a vMF / PS cap (optionally sign-symmetrised) to the target."""

    def log_q(kappa: float) -> torch.Tensor:
        loc = grid.pole[None]
        if family == "vmf":
            dist: Any = VonMisesFisher(
                loc, torch.tensor([[kappa]], dtype=torch.float64)
            )
            value = dist.log_prob(grid.points)
            flipped = dist.log_prob(-grid.points)
        else:
            dist = PowerSpherical(loc, torch.tensor([kappa], dtype=torch.float64))
            value = dist.log_prob(grid.points)
            flipped = dist.log_prob(-grid.points)
        if symmetrise:
            return torch.logaddexp(value, flipped) - math.log(2)
        return value

    def objective(log_kappa: float) -> float:
        value = grid.kl(log_q(math.exp(log_kappa)), log_target)
        return value if math.isfinite(value) else 1e3

    result = optimize.minimize_scalar(
        objective, bounds=(-2.0, 7.0), method="bounded", options={"xatol": 1e-6}
    )
    return {"kl": float(result.fun), "params": [math.exp(float(result.x))]}


def representability_general_check() -> dict[str, Any]:
    """Best-fit ``KL(q || Bingham)`` with the TNBBeta axis free of the spike.

    The aligned fit fixes ``mu`` on the Bingham axis ``a``. A TNBBeta about a ``mu`` at
    angle ``psi`` from ``a`` (notably ``psi = pi / 2``, where on ``S^1`` its latitude
    ``mu . z = 0`` is exactly the two points ``+-a``) is not rotationally symmetric
    about ``a``, so the KL is estimated by Monte Carlo with common random numbers and
    exact ``log q`` and minimised by Adam over ``(psi, p, q, epsilon)`` from several
    starts; the reported value re-estimates the winner with 1M fresh draws.
    """

    out: dict[str, Any] = {"rows": []}
    print(
        "\n### TNBBeta best-fit KL(q || Bingham) with free axis angle psi (MC, CRN)\n"
    )
    print("| d | kappa | KL | psi (deg) | p | q | epsilon | m |")
    print("|" + "---|" * 8)
    for dim in (2, 3, 4):
        for kappa in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            best = _fit_tnbbeta_free_axis(dim, kappa)
            row = {"dim": dim, "kappa": kappa, **best}
            out["rows"].append(row)
            print(
                f"| {dim} | {kappa:g} | {best['kl']:.3f} | "
                f"{math.degrees(best['psi']):.1f} | {best['p']:.3f} | "
                f"{best['q']:.3f} | {best['epsilon']:.3f} | "
                f"{best['epsilon'] - (dim - 1) / 2:.3f} |"
            )
    return out


def _fit_tnbbeta_free_axis(dim: int, kappa: float) -> dict[str, float]:
    eigenvalues = torch.zeros(1, dim, dtype=torch.float64)
    eigenvalues[0, -1] = kappa
    log_z = float(bingham_log_normaliser(eigenvalues)[0])
    log_area = math.log(2) + (dim / 2) * math.log(math.pi) - math.lgamma(dim / 2)

    def kl_estimate(params: torch.Tensor, draws: int, seed: int) -> torch.Tensor:
        psi, raw_p, raw_q, log_eps = params
        mu = torch.zeros(dim, dtype=torch.float64)
        mu = torch.cat([psi.cos()[None], psi.sin()[None], mu[2:]])
        dist = TNBBetaSpherical(
            mu,
            raw_p.sigmoid().clamp(1e-6, 1 - 1e-6),
            raw_q.sigmoid().clamp(1e-6, 1 - 1e-6),
            log_eps.exp() + 1e-4,
        )
        torch.manual_seed(seed)
        z, log_q = dist.rsample_with_log_prob(torch.Size([draws]))
        log_p = kappa * z[..., 0] ** 2 - log_z - log_area
        return (log_q - log_p).mean()

    best: dict[str, float] = {"kl": float("inf")}
    for psi0 in (0.0, math.pi / 4, math.pi / 2):
        for p0 in (0.0, 2.0):
            for eps0 in (math.log(0.2), math.log(1.0)):
                params = torch.tensor(
                    [psi0, p0, -4.0, eps0], dtype=torch.float64, requires_grad=True
                )
                optimiser = torch.optim.Adam([params], lr=0.05)
                for _ in range(250):
                    optimiser.zero_grad()
                    loss = kl_estimate(params, 20_000, 0)
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    optimiser.step()
                with torch.no_grad():
                    final = float(kl_estimate(params, 1_000_000, 1))
                if math.isfinite(final) and final < best["kl"]:
                    psi, raw_p, raw_q, log_eps = (float(v) for v in params)
                    best = {
                        "kl": final,
                        "psi": psi % math.pi,
                        "p": float(torch.tensor(raw_p).sigmoid()),
                        "q": float(torch.tensor(raw_q).sigmoid()),
                        "epsilon": math.exp(log_eps) + 1e-4,
                    }
    return best


# ------------------------------------------------------------------------- anisotropy


def anisotropy_check(sigmas: list[float]) -> dict[str, Any]:
    """Exact best axially symmetric KL to the Bingham posterior at the study's
    sigmas."""
    out: dict[str, Any] = {"rows": []}
    print("\n### Best axially symmetric KL(q || Bingham(A)), any axis and profile\n")
    print(
        "| d | sigma | mean KL(axial, any profile) | mean KL(axis = top eigvec) | "
        "mean one-sided (hemisphere) KL = axial + ln 2 | mean KL(uniform q = prior) | "
        "mean lam1 - lam2 | mean (lam1 - lam2) / d |"
    )
    print("|" + "---|" * 8)
    for dim in (2, 3, 4):
        for sigma in sigmas:
            task = AxialBinghamTask(dim, sigma)
            gen = torch.Generator().manual_seed(23)
            x, _ = task.sample(300, gen)
            matrix = task.bingham_matrix(x.double())
            eig = torch.linalg.eigvalsh(matrix)
            log_z = bingham_log_normaliser(eig).numpy()
            lam = eig.numpy()
            best, at_top, uniform = [], [], []
            for index in range(lam.shape[0]):
                trace = lam[index].sum()
                lo, hi = lam[index, 0], lam[index, -1]

                def fun(
                    a: float,
                    t: float = trace,
                    d: int = dim,
                    lz: float = float(log_z[index]),
                ) -> float:
                    return _axial_kl(a, t, d, lz)

                grid_a = np.linspace(lo, hi, 41)
                values = [fun(a) for a in grid_a]
                k = int(np.argmin(values))
                a0, a1 = grid_a[max(k - 1, 0)], grid_a[min(k + 1, 40)]
                res: Any = optimize.minimize_scalar(
                    fun, bounds=(a0, a1), method="bounded"
                )
                best.append(min(float(res.fun), min(values)))
                at_top.append(fun(hi))
                uniform.append(float(log_z[index] - trace / dim))
            gap = float((lam[:, -1] - lam[:, -2]).mean())
            row = {
                "dim": dim,
                "sigma": sigma,
                "axial": float(np.mean(best)),
                "axial_top": float(np.mean(at_top)),
                "one_sided": float(np.mean(best) + math.log(2)),
                "uniform_q": float(np.mean(uniform)),
                "gap": gap,
            }
            out["rows"].append(row)
            print(
                f"| {dim} | {sigma} | {row['axial']:.4f} | {row['axial_top']:.4f} | "
                f"{row['one_sided']:.4f} | {row['uniform_q']:.4f} | {gap:.2f} | "
                f"{gap / dim:.2f} |"
            )
    return out


def _axial_kl(alpha: float, trace: float, dim: int, log_z: float) -> float:
    """``KL`` of the best axially symmetric ``q`` about an axis with ``a^T A a =
    alpha``.

    With ``w`` the latitude and orbit-averaged log density ``Lbar(w) = alpha w^2 + (1 -
    w^2) (trace - alpha) / (d - 1) - log Z`` the optimal profile is ``r proportional to
    exp(Lbar)`` against the uniform measure and ``KL = log Z - log E_w exp(alpha w^2 +
    (1 - w^2) (trace - alpha) / (d - 1))``.
    """
    count = 6000
    j = (np.arange(count) + 0.5) / count
    theta = math.pi * (1 - np.cos(math.pi * j)) / 2
    dtheta = (math.pi**2 / 2) * np.sin(math.pi * j) / count
    w2 = np.cos(theta) ** 2
    log_measure = (dim - 2) * np.log(np.sin(theta)) + np.log(dtheta)
    log_measure -= special.logsumexp(log_measure)
    expo = alpha * w2 + (1 - w2) * (trace - alpha) / (dim - 1)
    total: Any = special.logsumexp(expo + log_measure)
    return float(log_z - total)


if __name__ == "__main__":
    main(sys.argv[1:])
