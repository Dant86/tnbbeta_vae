"""Correctness of TNBBeta at small epsilon, near q = 1 and near the poles.

Regression tests for two defects fixed in ``tnbbeta_univariate`` /
``tnbbeta_spherical`` (see ``writeup/results/tnb_small_epsilon_fix_2026-11-01.md``):

* ``rsample`` drew ``Beta(eps, eps)`` through Gamma draws that underflow below
  ``eps ~ 3e-3`` (the sample landed at ``y = p``, not at the boundary spikes);
* ``log_prob`` clamped ``y`` to ``[1e-6, 1 - 1e-6]``, so the evaluated density
  was sub-normalised and ``E_q[log q]`` was biased downward (negative MC KL).

Everything is checked against float64 quadrature in ``t = logit(y)`` on a grid
that is geometrically refined around ``logit(p)`` (the density can be a spike
of width ``1e-4`` there) and out to ``60 / eps`` in the tails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
import torch
from torch import Tensor
from torch.distributions import Beta
from torch.nn.functional import softplus

from tnbbeta_vae.distributions import TNBBetaSpherical, TNBBetaUnivariate
from tnbbeta_vae.distributions.tnbbeta_spherical import _log_surface_area
from tnbbeta_vae.distributions.tnbbeta_univariate import _log_density, _sample_logit

F64 = torch.float64
EPS_GRID = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 10.0, 100.0)
CLAMP = 1e-6
PQ_GRID = [
    (p, q)
    for p in (CLAMP, 0.015, 0.5, 0.9, 1 - CLAMP)
    for q in (CLAMP, 3e-3, 0.5, 0.99, 1 - CLAMP)
]


def _t(x: float) -> Tensor:
    return torch.tensor(x, dtype=F64)


def _grid(p: float, eps: float, points: int = 40001) -> tuple[Tensor, Tensor]:
    """Quadrature nodes ``t`` and weights ``dt`` (refined around ``logit(p)``)."""
    centre = math.log(p) - math.log1p(-p)
    scale = 1e-5
    reach = 60 / eps + 800.0
    top = math.asinh(reach / scale)
    u = torch.linspace(-top, top, points, dtype=F64)
    return centre + scale * torch.sinh(u), scale * torch.cosh(u) * (u[1] - u[0])


def _log_f_t(t: Tensor, p: float, q: float, eps: float) -> Tensor:
    """``log`` of the density of ``t = logit(y)``."""
    ly, l1y = -softplus(-t), -softplus(t)
    return _log_density(ly, l1y, _t(p), _t(q), _t(eps)) + ly + l1y


def _sphere_log_prob(t: Tensor, p: float, q: float, eps: float, dim: int) -> Tensor:
    """Spherical log-density at latitude logit ``t`` (no vector round-off)."""
    dist = TNBBetaSpherical(
        torch.eye(dim, dtype=F64)[0], _t(p), _t(q), _t(eps)
    )  # mean direction e_1
    return dist._log_prob_from_logs(-softplus(-t), -softplus(t))  # noqa: SLF001


def _sphere_log_f_t(t: Tensor, p: float, q: float, eps: float, dim: int) -> Tensor:
    """``log`` density of ``t`` under the sphere: density x area x measure x dw/dt."""
    ly, l1y = -softplus(-t), -softplus(t)
    log_w = ly + l1y + 2 * math.log(2)  # log(1 - w^2)
    return (
        _sphere_log_prob(t, p, q, eps, dim)
        + _log_surface_area(dim - 1)
        + (dim - 3) / 2 * log_w
        + math.log(2)
        + ly
        + l1y
    )


def _mass(log_f: Tensor, weight: Tensor) -> float:
    return float((log_f.exp() * weight).sum())


# ---- (a) normalisation -----------------------------------------------------


@pytest.mark.parametrize("eps", EPS_GRID)
def test_univariate_density_integrates_to_one(eps: float) -> None:
    """exp(log_prob) has mass 1 on every (p, q), including the clamps."""
    for p, q in PQ_GRID:
        t, w = _grid(p, eps)
        assert _mass(_log_f_t(t, p, q, eps), w) == pytest.approx(1.0, abs=2e-5), (
            p,
            q,
            eps,
        )


@pytest.mark.parametrize("dim", [3, 8, 16])
@pytest.mark.parametrize("eps", EPS_GRID)
def test_spherical_density_integrates_to_one(eps: float, dim: int) -> None:
    """The spherical density has unit mass in the latitude variable."""
    for p, q in PQ_GRID[::2]:
        t, w = _grid(p, eps)
        mass = _mass(_sphere_log_f_t(t, p, q, eps, dim), w)
        assert mass == pytest.approx(1.0, abs=2e-5), (p, q, eps, dim)


def test_public_log_prob_matches_logit_space_density() -> None:
    """The vector ``log_prob`` (chord form) equals the logit-space density."""
    dim = 8
    torch.manual_seed(0)
    mu = torch.randn(dim, dtype=F64)
    mu = mu / mu.norm()
    for eps in (0.3, 1.0, 10.0):
        dist = TNBBetaSpherical(mu, _t(0.7), _t(0.6), _t(eps))
        z = dist.sample((500,))
        w = z @ mu
        t = torch.log1p(w) - torch.log1p(-w)  # logit((1 + w) / 2)
        expected = _sphere_log_prob_mu(dist, t)
        torch.testing.assert_close(dist.log_prob(z), expected, rtol=1e-6, atol=1e-5)


def _sphere_log_prob_mu(dist: TNBBetaSpherical, t: Tensor) -> Tensor:
    return dist._log_prob_from_logs(-softplus(-t), -softplus(t))  # noqa: SLF001


# ---- (b) sampler vs density ------------------------------------------------


def _ks_vs_density(samples_t: Tensor, log_f: Tensor, t: Tensor, w: Tensor) -> float:
    """KS statistic of ``samples_t`` against the quadrature CDF (midpoint rule)."""
    mass = log_f.exp() * w
    cdf = (torch.cumsum(mass, 0) - 0.5 * mass).numpy()
    grid = t.numpy()
    return float(
        stats.kstest(samples_t.numpy(), lambda x: np.interp(x, grid, cdf)).statistic
    )


@pytest.mark.parametrize("eps", EPS_GRID)
def test_sampler_matches_density_ks(eps: float) -> None:
    """KS distance between sampled logits and the quadrature CDF is at noise level."""
    torch.manual_seed(1)
    n = 20000
    critical = 1.95 / math.sqrt(n)  # ~0.1% level, fixed seed
    for p, q in [(0.015, 3e-3), (0.5, 0.5), (0.9, 0.99), (0.3, 1 - CLAMP)]:
        t, w = _grid(p, eps)
        samples = _sample_logit(
            torch.full((n,), p, dtype=F64),
            torch.full((n,), q, dtype=F64),
            torch.full((n,), eps, dtype=F64),
        )
        ks = _ks_vs_density(samples, _log_f_t(t, p, q, eps), t, w)
        assert ks < critical, (p, q, eps, ks)


@pytest.mark.parametrize("dim", [3, 8, 16])
@pytest.mark.parametrize("eps", [0.3, 1.0, 10.0, 100.0])
def test_spherical_sampler_matches_latitude_density_ks(eps: float, dim: int) -> None:
    """Public ``rsample`` (vectors, random mean direction) matches the density."""
    torch.manual_seed(2)
    n = 20000
    mu = torch.randn(dim, dtype=F64)
    mu = mu / mu.norm()
    p, q = 0.7, 0.8
    dist = TNBBetaSpherical(mu, _t(p), _t(q), _t(eps))
    z = dist.rsample((n,))
    assert torch.allclose(z.norm(dim=-1), torch.ones(n, dtype=F64), atol=1e-12)
    w_lat = z @ mu
    t_s = torch.log1p(w_lat) - torch.log1p(-w_lat)
    t, w = _grid(p, eps)
    ks = _ks_vs_density(t_s, _log_f_t(t, p, q, eps), t, w)
    assert ks < 1.95 / math.sqrt(n), ks
    # The fibre direction is uniform: no preferred direction orthogonal to mu.
    resid = z - w_lat[:, None] * mu
    resid = resid / resid.norm(dim=-1, keepdim=True)
    assert float(resid.mean(0).norm()) < 6 / math.sqrt(n)


def test_tiny_epsilon_puts_mass_at_the_boundary_not_at_p() -> None:
    """eps = 1e-4: ~all mass near y in {0, 1} with the exact split, none at y = p."""
    torch.manual_seed(3)
    n = 40000
    for p, q in [(0.9, 0.5), (0.015, 3e-3)]:
        dist = TNBBetaUnivariate(_t(p), _t(q), _t(1e-4))
        y = dist.rsample((n,))
        t, w = _grid(p, 1e-4)
        exact_upper = float((_log_f_t(t, p, q, 1e-4).exp() * w)[t > 0].sum())
        assert 0.45 < exact_upper < 0.55
        assert float(((y > 0.999) | (y < 0.001)).double().mean()) > 0.99
        assert float(((y - p).abs() / (p * (1 - p)) < 0.05).double().mean()) < 0.005
        # P(y > 1/2) = P(t > 0): equals the quadrature value, not 0.066 / 0.934.
        assert float((y > 0.5).double().mean()) == pytest.approx(
            exact_upper, abs=5 / math.sqrt(n)
        )


# ---- (c) the failing quantity: E_q[log q] and the KL ------------------------


def _exact_elq(p: float, q: float, eps: float, dim: int) -> float:
    t, w = _grid(p, eps, 80001)
    log_f = _sphere_log_f_t(t, p, q, eps, dim)
    return float((log_f.exp() * _sphere_log_prob(t, p, q, eps, dim) * w).sum())


def test_elq_at_the_failing_parameters_matches_quadrature() -> None:
    """p=0.015, q=3e-3, eps=1e-4, d=3: exact E_q[log q] ~ +9973 (library gave -7.0)."""
    torch.manual_seed(4)
    p, q, eps, n = 0.015, 3e-3, 1e-4, 20000
    exact = _exact_elq(p, q, eps, 3)
    assert exact == pytest.approx(9973, rel=0.01)
    t = _sample_logit(
        torch.full((n,), p, dtype=F64),
        torch.full((n,), q, dtype=F64),
        torch.full((n,), eps, dtype=F64),
    )
    lp = _sphere_log_prob(t, p, q, eps, 3)
    sem = float(lp.std() / math.sqrt(n))
    assert abs(float(lp.mean()) - exact) < 5 * sem + 0.002 * abs(exact)


@pytest.mark.parametrize("dim", [3, 8])
@pytest.mark.parametrize("eps", [1e-3, 1e-2, 0.1, 0.3, 1.0, 10.0])
def test_elq_logit_space_matches_quadrature(eps: float, dim: int) -> None:
    """MC E_q[log q] (sampler + density, logit space) equals quadrature."""
    torch.manual_seed(5)
    n = 40000
    for p, q in [(0.015, 3e-3), (0.5, 0.5), (0.9, 0.99)]:
        exact = _exact_elq(p, q, eps, dim)
        t = _sample_logit(
            torch.full((n,), p, dtype=F64),
            torch.full((n,), q, dtype=F64),
            torch.full((n,), eps, dtype=F64),
        )
        lp = _sphere_log_prob(t, p, q, eps, dim)
        sem = float(lp.std() / math.sqrt(n))
        assert abs(float(lp.mean()) - exact) < 5 * sem + 1e-3 * abs(exact), (
            p,
            q,
            eps,
        )


@pytest.mark.parametrize("dim", [3, 8, 16])
def test_public_mc_kl_to_uniform_is_nonnegative(dim: int) -> None:
    """KL(q || Uniform) by MC through the public API is >= 0 within MC error."""
    torch.manual_seed(6)
    n = 20000
    mu = torch.randn(dim, dtype=F64)
    mu = mu / mu.norm()
    for eps in (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 10.0):
        for p, q in [(0.015, 3e-3), (0.5, 0.5), (0.9, 0.99), (CLAMP, 0.5)]:
            dist = TNBBetaSpherical(mu, _t(p), _t(q), _t(eps))
            z = dist.rsample((n,))
            kl = dist.log_prob(z) + _log_surface_area(dim)
            sem = float(kl.std() / math.sqrt(n))
            assert float(kl.mean()) > -4 * sem - 1e-6, (dim, eps, p, q)


def test_public_kl_at_failing_parameters_is_large_and_positive() -> None:
    """Vector round-off caps the pole density, but KL is no longer ~ -7."""
    torch.manual_seed(7)
    mu = torch.tensor([0.0, 0.6, 0.8], dtype=F64)
    dist = TNBBetaSpherical(mu, _t(0.015), _t(3e-3), _t(1e-4))
    kl = dist.log_prob(dist.rsample((20000,))) + _log_surface_area(3)
    assert float(kl.mean()) > 30.0


@pytest.mark.parametrize("dim", [3, 8])
@pytest.mark.parametrize("eps", [0.1, 1.0, 10.0])
def test_public_elq_matches_quadrature_where_vectors_resolve(
    eps: float, dim: int
) -> None:
    """For eps >= 0.1 the public (vector) MC E_q[log q] agrees with quadrature."""
    torch.manual_seed(8)
    n = 60000
    mu = torch.randn(dim, dtype=F64)
    mu = mu / mu.norm()
    p, q = 0.8, 0.7
    dist = TNBBetaSpherical(mu, _t(p), _t(q), _t(eps))
    lp = dist.log_prob(dist.rsample((n,)))
    sem = float(lp.std() / math.sqrt(n))
    assert abs(float(lp.mean()) - _exact_elq(p, q, eps, dim)) < 5 * sem + 1e-3


# ---- agreement with the previous implementation for eps >= 1 ----------------


def _old_log_prob(p: Tensor, q: Tensor, eps: Tensor, value: Tensor) -> Tensor:
    """Frozen copy of the pre-fix ``TNBBetaUnivariate.log_prob`` (ec2677e)."""
    y = value.clamp(CLAMP, 1.0 - CLAMP)
    denom = p * (1 - y) + (1 - p) * y
    log_gamma = (
        torch.log(y)
        + torch.log1p(-y)
        + torch.log(p)
        + torch.log1p(-p)
        - 2 * torch.log(denom)
    )
    gamma = log_gamma.exp()
    log_beta_const = 2 * torch.lgamma(eps) - torch.lgamma(2 * eps)
    return (
        -torch.log(y)
        - torch.log1p(-y)
        - log_beta_const
        + eps * torch.log1p(-q)
        + eps * log_gamma
        - (eps + 0.5) * torch.log1p(-4 * q * gamma)
    )


def _old_rsample(p: Tensor, q: Tensor, eps: Tensor) -> Tensor:
    """Frozen copy of the pre-fix ``TNBBetaUnivariate.rsample`` (ec2677e)."""
    z = Beta(eps, eps).rsample()
    s = 2 * z - 1
    u = (1 + s * torch.sqrt((1 - q) / (1 - q * s**2))) / 2
    return p * u / ((1 - p) * (1 - u) + p * u)


def test_log_prob_equals_previous_for_eps_ge_one() -> None:
    """Float64 log_prob agrees to 1e-6 relative on a (p, q, eps, y) grid, eps >= 1."""
    ys = torch.tensor([1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 1 - 1e-3], dtype=F64)
    for eps in (1.0, 1.5, 3.0, 10.0, 100.0):
        for p in (0.01, 0.3, 0.5, 0.9, 0.999):
            for q in (1e-6, 0.1, 0.5, 0.9, 0.999):
                new = TNBBetaUnivariate(_t(p), _t(q), _t(eps)).log_prob(ys)
                old = _old_log_prob(_t(p), _t(q), _t(eps), ys)
                torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-9)


def test_samples_match_previous_distribution_for_eps_ge_one() -> None:
    """Two-sample KS between old and new samplers (eps >= 1), uni- and spherical."""
    torch.manual_seed(9)
    n = 50000
    for eps in (1.0, 3.0, 30.0):
        for p, q in [(0.5, 0.5), (0.9, 0.99), (0.2, 0.1)]:
            args = (torch.full((n,), v, dtype=F64) for v in (p, q, eps))
            pp, qq, ee = args
            old = _old_rsample(pp, qq, ee)
            new = TNBBetaUnivariate(pp, qq, ee).rsample()
            ks = stats.ks_2samp(old.numpy(), new.numpy()).statistic  # pyright: ignore[reportAttributeAccessIssue]
            assert ks < 1.95 * math.sqrt(2 / n), (eps, p, q, ks)


def test_mean_reparameterised_gradient_matches_previous_for_eps_ge_one() -> None:
    """Mean pathwise gradient of a smooth function agrees (within MC error)."""
    torch.manual_seed(10)
    n = 200000

    def grads(sampler: object, p0: float, q0: float, e0: float) -> list[Tensor]:
        p = torch.full((n,), p0, dtype=F64, requires_grad=True)
        q = torch.full((n,), q0, dtype=F64, requires_grad=True)
        e = torch.full((n,), e0, dtype=F64, requires_grad=True)
        y = sampler(p, q, e)  # type: ignore[operator]
        (torch.sin(3 * y) + y**2).sum().backward()
        return [x.grad for x in (p, q, e)]  # type: ignore[misc]

    def new(p: Tensor, q: Tensor, e: Tensor) -> Tensor:
        return TNBBetaUnivariate(p, q, e).rsample()

    for p0, q0, e0 in [(0.4, 0.5, 1.0), (0.7, 0.9, 10.0), (0.5, 0.2, 100.0)]:
        for g_old, g_new in zip(
            grads(_old_rsample, p0, q0, e0), grads(new, p0, q0, e0), strict=True
        ):
            se = math.sqrt(float(g_old.var() / n) + float(g_new.var() / n))
            assert abs(float(g_old.mean()) - float(g_new.mean())) < 5 * se + 1e-9


# ---- (d) gradients ----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rsample_gradients_are_finite_everywhere(dtype: torch.dtype) -> None:
    """Pathwise gradients are finite for tiny/huge eps and p, q at the clamps."""
    torch.manual_seed(11)
    for eps in (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e3):
        for p, q in [(CLAMP, CLAMP), (1 - CLAMP, 1 - CLAMP), (0.5, 0.5)]:
            pp, qq, ee = (
                torch.full((512,), v, dtype=dtype, requires_grad=True)
                for v in (p, q, eps)
            )
            y = TNBBetaUnivariate(pp, qq, ee).rsample()
            y.sum().backward()
            mu = torch.nn.functional.normalize(
                torch.randn(512, 5, dtype=dtype), dim=-1
            ).requires_grad_()
            p2, q2, e2 = (
                torch.full((512,), v, dtype=dtype, requires_grad=True)
                for v in (p, q, eps)
            )
            z = TNBBetaSpherical(mu, p2, q2, e2).rsample()
            (z * torch.arange(5, dtype=dtype)).sum().backward()
            for g in (pp.grad, qq.grad, ee.grad, p2.grad, q2.grad, e2.grad, mu.grad):
                assert g is not None
                assert torch.isfinite(g).all(), (dtype, eps, p, q)


# ---- (e) float32 / float64 finiteness and precision --------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_samples_and_log_prob_are_finite_at_extremes(dtype: torch.dtype) -> None:
    """No inf/NaN for eps in [1e-6, 1e3], p, q at their clamps (incl. 1 - 1e-6)."""
    torch.manual_seed(12)
    for eps in (1e-6, 1e-4, 1e-2, 1.0, 1e1, 1e3):
        for p in (CLAMP, 0.5, 1 - CLAMP):
            for q in (CLAMP, 0.5, 1 - CLAMP):
                pt, qt, et = (torch.tensor(v, dtype=dtype) for v in (p, q, eps))
                uni = TNBBetaUnivariate(pt, qt, et)
                y = uni.rsample((256,))
                assert torch.isfinite(y).all()
                assert torch.isfinite(uni.log_prob(y)).all(), (dtype, eps, p, q)
                mu = torch.nn.functional.normalize(torch.randn(6, dtype=dtype), dim=-1)
                sph = TNBBetaSpherical(mu, pt, qt, et)
                z = sph.rsample((256,))
                assert torch.isfinite(z).all()
                assert torch.isfinite(sph.log_prob(z)).all(), (dtype, eps, p, q)
                # exactly at, and one ulp off, both poles of the mean axis
                pole = torch.stack([mu, -mu, mu * (1 - 1e-7), -mu * (1 - 1e-7)])
                pole = torch.nn.functional.normalize(pole, dim=-1)
                assert torch.isfinite(sph.log_prob(pole)).all(), (dtype, eps, p, q)


def test_float32_log_prob_near_q_one_matches_float64() -> None:
    """q = 1 - 1e-6 used to lose 1.4-8 nats in float32 (1 - 4 q gamma cancellation)."""
    ys = torch.linspace(0.05, 0.95, 19)
    for eps in (8.0, 30.0, 120.0):
        for p in (0.5, 1 - CLAMP):
            p32, q32, e32 = (torch.tensor(v) for v in (p, 1 - CLAMP, eps))
            got = TNBBetaUnivariate(p32, q32, e32).log_prob(ys).double()
            # float32 inputs are themselves rounded: the reference is the float64
            # density at those rounded parameters and points.
            ref = TNBBetaUnivariate(p32.double(), q32.double(), e32.double()).log_prob(
                ys.double()
            )
            assert float((got - ref).abs().max()) < 0.05, (eps, p)


def test_float32_log_prob_resolves_points_next_to_the_pole() -> None:
    """On the sphere 1 - |w| comes from the chord, so z next to mu is resolved."""
    dim = 8
    mu = torch.nn.functional.normalize(torch.randn(dim, dtype=F64), dim=-1)
    for delta in (1e-3, 1e-4, 1e-5, 1e-6):
        # a point at angular distance delta from mu
        orth = torch.randn(dim, dtype=F64)
        orth = orth - (orth @ mu) * mu
        orth = orth / orth.norm()
        z64 = math.cos(delta) * mu + math.sin(delta) * orth
        for eps in (0.5, 5.0):
            d64 = TNBBetaSpherical(mu, _t(0.9), _t(0.5), _t(eps))
            d32 = TNBBetaSpherical(
                mu.float(), _t(0.9).float(), _t(0.5).float(), _t(eps).float()
            )
            lp64 = float(d64.log_prob(z64))
            lp32 = float(d32.log_prob(z64.float()))
            # float32 jitter of z (6e-8) is 6e-8/delta relative in the chord
            assert abs(lp32 - lp64) < 8 * 6e-8 / delta * (abs(eps - 1) + 3) + 0.02, (
                delta,
                eps,
                lp32,
                lp64,
            )


# ---- rsample_with_log_prob ---------------------------------------------------


def test_rsample_with_log_prob_matches_log_prob_when_vectors_resolve() -> None:
    """For eps >= 1 the logit-space log-density equals ``log_prob`` of the sample."""
    torch.manual_seed(13)
    mu = torch.nn.functional.normalize(torch.randn(4, 8, dtype=F64), dim=-1)
    dist = TNBBetaSpherical(mu, _t(0.7), _t(0.6), _t(3.0))
    z, lp = dist.rsample_with_log_prob((50,))
    assert z.shape == (50, 4, 8)
    assert lp.shape == (50, 4)
    torch.testing.assert_close(lp, dist.log_prob(z), rtol=1e-6, atol=1e-6)


def test_rsample_with_log_prob_gives_exact_elq_at_the_failing_parameters() -> None:
    """p=0.015, q=3e-3, eps=1e-4 (d=3): MC E_q[log q] equals the quadrature value."""
    torch.manual_seed(14)
    n = 20000
    mu = torch.tensor([0.0, 0.6, 0.8], dtype=F64)
    dist = TNBBetaSpherical(mu, _t(0.015), _t(3e-3), _t(1e-4))
    z, lp = dist.rsample_with_log_prob((n,))
    assert z.shape == (n, 3)
    exact = _exact_elq(0.015, 3e-3, 1e-4, 3)
    sem = float(lp.std() / math.sqrt(n))
    assert abs(float(lp.mean()) - exact) < 5 * sem + 0.002 * abs(exact)
    assert float(lp.mean()) + _log_surface_area(3) > 9000  # KL, not -4.4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rsample_with_log_prob_is_finite_with_finite_gradients(
    dtype: torch.dtype,
) -> None:
    """Value and pathwise gradients stay finite for tiny eps and q near 1."""
    torch.manual_seed(15)
    for eps in (1e-6, 1e-4, 1e-2, 1.0, 1e3):
        for p, q in [(CLAMP, 1 - CLAMP), (1 - CLAMP, CLAMP), (0.5, 0.5)]:
            mu = torch.nn.functional.normalize(torch.randn(256, 5, dtype=dtype), dim=-1)
            p_, q_, e_ = (
                torch.full((256,), v, dtype=dtype, requires_grad=True)
                for v in (p, q, eps)
            )
            z, lp = TNBBetaSpherical(mu, p_, q_, e_).rsample_with_log_prob((4,))
            assert torch.isfinite(z).all()
            assert torch.isfinite(lp).all(), (dtype, eps, p, q)
            lp.mean().backward()
            for g in (p_.grad, q_.grad, e_.grad):
                assert g is not None
                assert torch.isfinite(g).all(), (dtype, eps, p, q)
