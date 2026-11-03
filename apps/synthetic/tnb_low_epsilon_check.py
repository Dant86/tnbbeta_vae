"""Why the Monte Carlo KL of TNBBetaSpherical is invalid at small epsilon (d = 3).

Two independent defects, both dtype-independent (float64 does not fix either):

1. ``TNBBetaUnivariate.rsample`` draws ``Beta(eps, eps)`` through torch's Gamma
   sampler; below ``eps`` of about ``3e-3`` both Gamma draws underflow, the draw
   becomes exactly 0.5 and the sample lands at ``y = p`` (the median), not at the
   boundary spikes where the density puts nearly all its mass.
2. ``log_prob`` clamps ``y`` to ``[1e-6, 1 - 1e-6]``. Below ``eps`` of about 0.3 a
   non-negligible share of the mass is closer to the boundary than that, where the
   true density is ``~ eps / y`` (unbounded); the clamped function integrates to
   far less than 1 (0.0015 at ``eps = 1e-4``), so ``E_q[log q]`` is under-estimated
   by about ``1 / eps`` nats and ``KL(q || p)`` can come out negative.

This module is the unclamped, underflow-free reference used to check both: an exact
sampler (log-Gamma representation), the exact ``E_q[log q]`` by quadrature in
``t = logit(y)``, and ``honest_kl``. ``python -m apps.synthetic.tnb_low_epsilon_check``
is the minimal reproducer. See
``writeup/results/tnb_negative_kl_clamp_and_beta_underflow_2026-11-01.md``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Gamma
from torch.nn.functional import softplus

from tnbbeta_vae.data.ring_posterior import RingPosteriorTask
from tnbbeta_vae.distributions import TNBBetaSpherical, TNBBetaUnivariate
from tnbbeta_vae.distributions.tnbbeta_spherical import _householder_reflect

__all__ = ["exact_elq", "honest_kl", "log_density_y", "sample_t", "sample_z"]

_DTYPE = torch.float64


def log_density_y(
    t: Tensor, p: float, q: float, eps: float, clamp: float | None = None
) -> Tensor:
    """Unclamped ``log f_Y(y)`` of ``TNBBeta(p, q, eps)`` at ``t = logit(y)`` (float64).

    Args:
        t: Logits of ``y``.
        p: Median.
        q: Concentration.
        eps: Boundary parameter.
        clamp: If set, ``y`` is clamped to ``[clamp, 1 - clamp]`` first, mimicking
            ``TNBBetaUnivariate.log_prob`` (``clamp = 1e-6``).

    Returns:
        The log-density of ``y`` at the (possibly clamped) points.
    """
    if clamp is not None:
        low = math.log(clamp) - math.log1p(-clamp)
        t = t.clamp(low, -low)
    ly, l1y = -softplus(-t), -softplus(t)
    lp, l1p = math.log(p), math.log1p(-p)
    log_denom = torch.logaddexp(lp + l1y, l1p + ly)
    log_gamma = ly + l1y + lp + l1p - 2 * log_denom
    # 1 - 4 q gamma = (1 - q) + q (1 - 4 gamma), computed without cancellation.
    one_minus = (1 - q) + q * (
        torch.exp(lp + l1y - log_denom) - torch.exp(l1p + ly - log_denom)
    ) ** 2
    log_beta = 2 * math.lgamma(eps) - math.lgamma(2 * eps)
    return (
        -ly
        - l1y
        - log_beta
        + eps * math.log1p(-q)
        + eps * log_gamma
        - (eps + 0.5) * torch.log(one_minus)
    )


def exact_elq(p: Tensor, q: Tensor, eps: Tensor, points: int = 40001) -> Tensor:
    """Exact ``E_q[log q]`` (surface density, ``d = 3``) by quadrature in ``logit(y)``.

    Args:
        p: Medians, shape ``(batch,)``.
        q: Concentrations, shape ``(batch,)``.
        eps: Boundary parameters, shape ``(batch,)``.
        points: Quadrature nodes (on a ``sinh``-spaced grid reaching ``60 / eps``).

    Returns:
        Float64 tensor of shape ``(batch,)``. About ``1 / eps`` for small ``eps``.
    """
    out = []
    for pi, qi, ei in zip(p.tolist(), q.tolist(), eps.tolist(), strict=True):
        reach = 60 / ei + 800.0
        u = torch.linspace(-math.asinh(reach), math.asinh(reach), points, dtype=_DTYPE)
        t = torch.sinh(u)
        weight = torch.cosh(u) * (u[1] - u[0])
        lf = log_density_y(t, pi, qi, ei)
        density_t = torch.exp(lf - softplus(-t) - softplus(t))
        shift = -math.log(2) - math.log(2 * math.pi)
        out.append(float((density_t * (lf + shift) * weight).sum()))
    return torch.tensor(out, dtype=_DTYPE)


def _log_gamma_sample(eps: Tensor, shape: tuple[int, ...]) -> Tensor:
    """``log Gamma(eps)`` draws without underflow: ``Gamma(1 + eps) * U^(1 / eps)``."""
    e = eps.expand(shape)
    g = Gamma(1 + e, torch.ones_like(e)).sample()
    return torch.log(g) + torch.log(torch.rand(shape, dtype=_DTYPE)) / e


def sample_t(p: Tensor, q: Tensor, eps: Tensor, num: int) -> Tensor:
    """Exact draws of ``t = logit(y)``, ``y ~ TNBBeta``; shape ``(num, batch)``."""
    shape = (num, *p.shape)
    big_t = _log_gamma_sample(eps, shape) - _log_gamma_sample(eps, shape)  # logit z
    s = torch.tanh(big_t / 2)
    a = s * torch.sqrt((1 - q) / (1 - q * s**2))
    log_1ms2 = -2 * (
        big_t.abs() / 2 + torch.log1p(torch.exp(-big_t.abs())) - math.log(2)
    )
    log_1ma2 = log_1ms2 - torch.log1p(-q * s**2)
    logit_u = torch.sign(big_t) * (2 * torch.log1p(a.abs()) - log_1ma2)
    return torch.log(p / (1 - p)) + logit_u


def sample_z(mu: Tensor, p: Tensor, q: Tensor, eps: Tensor, num: int) -> Tensor:
    """Exact samples of ``TNBBetaSpherical`` (``d = 3``), shape ``(num, batch, 3)``."""
    t = sample_t(p, q, eps, num)
    w, radius = torch.tanh(t / 2), 1 / torch.cosh(t / 2)
    g = torch.randn(*t.shape, 2, dtype=_DTYPE)
    v = g / g.norm(dim=-1, keepdim=True)
    z = torch.cat([w.unsqueeze(-1), radius.unsqueeze(-1) * v], dim=-1)
    return _householder_reflect(z, mu.expand_as(z))


def honest_kl(
    posterior: TNBBetaSpherical, task: RingPosteriorTask, x: Tensor, num: int = 2000
) -> Tensor:
    """``KL(q || p_true)`` as exact ``E[log q]`` minus MC ``E[log p_true]``.

    Args:
        posterior: Batch of ``TNBBetaSpherical`` over ``S^2``, batch shape ``(B,)``.
        task: The ring task with a float64 axis.
        x: Observations, shape ``(B, 3)``.
        num: Samples for the cross term.

    Returns:
        Float64 tensor of shape ``(B,)``.
    """
    p, q, eps = (v.double() for v in (posterior.p, posterior.q, posterior.epsilon))
    z = sample_z(posterior.mean_direction.double(), p, q, eps, num)
    xd = x.double()
    energy = ((xd[None] - task.features(z @ task.axis)) / task.sigma).pow(2).sum(-1) / 2
    cross = (-energy - task._log_normaliser(xd)).mean(0)  # noqa: SLF001
    return exact_elq(p, q, eps) - cross


def main() -> None:
    """Minimal reproducer of both defects (no checkpoints needed)."""
    torch.manual_seed(0)
    print("eps     lib P(y==p)  lib E[log f]  exact E[log f]  clamped-density mass")
    for eps in (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0):
        d = TNBBetaUnivariate(
            torch.tensor(0.9, dtype=_DTYPE),
            torch.tensor(0.5, dtype=_DTYPE),
            torch.tensor(eps, dtype=_DTYPE),
        )
        y = d.rsample((400000,))
        t = torch.linspace(-1, 1, 400001, dtype=_DTYPE)
        reach = 60 / eps + 800.0
        u = torch.linspace(-math.asinh(reach), math.asinh(reach), 400001, dtype=_DTYPE)
        t = torch.sinh(u)
        weight = torch.cosh(u) * (u[1] - u[0])
        lf = log_density_y(t, 0.9, 0.5, eps)
        mass_t = torch.exp(lf - softplus(-t) - softplus(t))
        clamped = torch.exp(
            log_density_y(t, 0.9, 0.5, eps, 1e-6) - softplus(-t) - softplus(t)
        )
        print(
            f"{eps:6.0e}  {float((y == 0.9).double().mean()):10.3f}  "
            f"{float(d.log_prob(y).mean()):12.3f}  "
            f"{float((mass_t * lf * weight).sum()):14.3f}  "
            f"{float((clamped * weight).sum()):10.4f}"
        )


if __name__ == "__main__":
    main()
