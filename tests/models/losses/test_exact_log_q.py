"""Tests that the MC ELBO and KL use the exact TNBBeta log q at tiny epsilon."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.distributions import kl_divergence

from apps.synthetic.tnb_low_epsilon_check import exact_elq
from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.heads import (
    LatentFamily,
    head_size,
    posterior_from_raw,
    standard_prior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.importance_weighted import importance_weighted_metrics
from tnbbeta_vae.models.losses.sampling import sample_and_kl, sample_with_log_prob

F64 = torch.float64
_LOG_AREA_S2 = math.log(4 * math.pi)


def _tnbbeta(
    p: float, q: float, eps: float, rows: int, dim: int = 3
) -> TNBBetaSpherical:
    mu = torch.zeros(rows, dim, dtype=F64)
    mu[:, 0] = 1.0
    full = lambda v: torch.full((rows,), v, dtype=F64)  # noqa: E731
    return TNBBetaSpherical(mu, full(p), full(q), full(eps))


def _zero_decoder(z: Tensor) -> Tensor:
    return torch.zeros(z.shape[0], 2, dtype=z.dtype)


def _old_monte_carlo_kl(posterior: Any, prior: Any, num_samples: int) -> Tensor:
    """The pre-change estimator: ``rsample`` then ``log_prob`` at the vector."""
    kls = []
    for _ in range(num_samples):
        z = posterior.rsample()
        kls.append(posterior.log_prob(z) - prior.log_prob(z))
    return torch.stack(kls).mean(0)


def _mean_and_sem(per_row: Tensor) -> tuple[float, float]:
    return float(per_row.mean()), float(per_row.std() / math.sqrt(per_row.numel()))


# ---- (i) exact E_q[log q] at tiny epsilon ----------------------------------


def test_mc_elbo_kl_matches_quadrature_at_the_failing_parameters() -> None:
    """p=0.015, q=3e-3, eps=1e-4, d=3: exact KL ~ 1e4; the vector path gave ~60."""
    torch.manual_seed(0)
    p, q, eps, rows = 0.015, 3e-3, 1e-4, 4000
    exact = float(
        exact_elq(torch.tensor([p]), torch.tensor([q]), torch.tensor([eps]))
    ) + (_LOG_AREA_S2)
    posterior = _tnbbeta(p, q, eps, rows)
    prior = standard_prior("tnbbeta", 3, torch.device("cpu"))
    x = torch.zeros(rows, 2, dtype=F64)

    new = monte_carlo_elbo(x, posterior, prior, _zero_decoder, num_samples=8)["kl"]
    mean, sem = _mean_and_sem(new)
    assert abs(mean - exact) < 6 * sem + 0.01 * exact

    old = _old_monte_carlo_kl(posterior, prior, 8)
    assert float(old.mean()) < 0.2 * exact  # the vector path under-states it


@pytest.mark.parametrize("eps", [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 10.0])
@pytest.mark.parametrize("p_q", [(0.015, 3e-3), (0.5, 0.5), (0.9, 0.99)])
def test_mc_elbo_kl_matches_quadrature_across_epsilon(
    eps: float, p_q: tuple[float, float]
) -> None:
    torch.manual_seed(1)
    p, q = p_q
    rows = 3000
    exact = float(
        exact_elq(torch.tensor([p]), torch.tensor([q]), torch.tensor([eps]))
    ) + (_LOG_AREA_S2)
    posterior = _tnbbeta(p, q, eps, rows)
    prior = standard_prior("tnbbeta", 3, torch.device("cpu"))
    kl = monte_carlo_elbo(
        torch.zeros(rows, 2, dtype=F64),
        posterior,
        prior,
        _zero_decoder,
        num_samples=8,
    )["kl"]
    mean, sem = _mean_and_sem(kl)
    assert abs(mean - exact) < 6 * sem + 0.01 * abs(exact) + 1e-3
    assert mean > -6 * sem  # a KL is non-negative


@pytest.mark.parametrize("dim", [3, 8])
@pytest.mark.parametrize("eps", [1e-4, 1e-3, 1e-2, 0.1, 1.0])
def test_importance_weighted_kl_is_nonnegative_at_small_epsilon(
    dim: int, eps: float
) -> None:
    """The paper's Table 1 KL (MC ``log q - log p``) is >= 0 at any epsilon."""
    torch.manual_seed(2)
    rows = 200
    posterior = _tnbbeta(0.015, 3e-3, eps, rows, dim)
    prior = standard_prior("tnbbeta", dim, torch.device("cpu"))

    class _Model:
        def posterior_and_prior(self, x: Tensor) -> tuple[Any, Any]:
            return posterior, prior

        def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
            return torch.zeros(z.shape[:-1], dtype=z.dtype)

    metrics = importance_weighted_metrics(
        _Model(), torch.zeros(rows, 1), num_samples=200, chunk_size=50
    )
    mean, sem = _mean_and_sem(metrics["kl"])
    assert mean > -6 * sem
    if dim == 3:
        exact = float(
            exact_elq(torch.tensor([0.015]), torch.tensor([3e-3]), torch.tensor([eps]))
        )
        assert abs(mean - (exact + _LOG_AREA_S2)) < 6 * sem + 0.02 * abs(exact)


def test_large_epsilon_agrees_with_the_vector_path_to_mc_error() -> None:
    torch.manual_seed(3)
    rows = 6000
    posterior = _tnbbeta(0.7, 0.4, 3.0, rows)
    prior = standard_prior("tnbbeta", 3, torch.device("cpu"))
    new = monte_carlo_elbo(
        torch.zeros(rows, 2, dtype=F64), posterior, prior, _zero_decoder, num_samples=4
    )["kl"]
    old = _old_monte_carlo_kl(posterior, prior, 4)
    (m_new, s_new), (m_old, s_old) = _mean_and_sem(new), _mean_and_sem(old)
    assert abs(m_new - m_old) < 6 * math.hypot(s_new, s_old)


def test_analytic_kl_path_is_untouched_and_skips_log_prob() -> None:
    """``analytic_kl=True`` calls ``kl_divergence`` and never evaluates ``log q``."""
    torch.manual_seed(4)
    posterior = posterior_from_raw("vmf", torch.randn(5, 4), 3)
    prior = standard_prior("vmf", 3, torch.device("cpu"))
    x = torch.zeros(5, 2)
    terms = monte_carlo_elbo(
        x, posterior, prior, lambda z: torch.zeros(5, 2), analytic_kl=True
    )
    assert torch.equal(terms["kl"], kl_divergence(posterior, prior))

    gaussian = posterior_from_raw("gaussian", torch.randn(5, 6), 3)
    g_prior = standard_prior("gaussian", 3, torch.device("cpu"))
    g_terms = monte_carlo_elbo(
        x, gaussian, g_prior, lambda z: torch.zeros(5, 2), analytic_kl=True
    )
    assert torch.equal(g_terms["kl"], kl_divergence(gaussian, g_prior))

    # TNBBeta still has no closed form: analytic_kl=True still raises.
    tnb = _tnbbeta(0.5, 0.5, 1.0, 3)
    with pytest.raises(NotImplementedError):
        monte_carlo_elbo(
            torch.zeros(3, 2, dtype=F64),
            tnb,
            standard_prior("tnbbeta", 3, torch.device("cpu")),
            _zero_decoder,
            analytic_kl=True,
        )


# ---- (ii) other families bit-identical -------------------------------------


@pytest.mark.parametrize("family", ["vmf", "power_spherical", "gaussian"])
@pytest.mark.parametrize("analytic", [False, True])
def test_other_families_monte_carlo_elbo_is_bit_identical(
    family: LatentFamily, analytic: bool
) -> None:
    dim = 4
    raw = torch.randn(
        6, head_size(family, dim), generator=torch.Generator().manual_seed(5)
    )
    x = torch.randn(6, 2)

    def run() -> dict[str, Tensor]:
        posterior = posterior_from_raw(family, raw, dim)
        prior = standard_prior(family, dim, torch.device("cpu"))
        w = torch.linspace(-1, 1, dim * 2).reshape(dim, 2)
        torch.manual_seed(11)
        return monte_carlo_elbo(
            x, posterior, prior, lambda z: z @ w, num_samples=3, analytic_kl=analytic
        )

    def reference() -> dict[str, Tensor]:
        # The code before the change, verbatim.
        posterior = posterior_from_raw(family, raw, dim)
        prior = standard_prior(family, dim, torch.device("cpu"))
        w = torch.linspace(-1, 1, dim * 2).reshape(dim, 2)
        torch.manual_seed(11)
        exact = kl_divergence(posterior, prior) if analytic else None
        lls, kls = [], []
        for _ in range(3):
            z = posterior.rsample()
            lls.append(pixel_log_likelihood(x, z @ w, "gaussian", 1.0))
            if exact is None:
                kls.append(posterior.log_prob(z) - prior.log_prob(z))
        ll = torch.stack(lls).mean(0)
        kl = exact if exact is not None else torch.stack(kls).mean(0)
        return {"elbo": ll - kl, "log_likelihood": ll, "kl": kl}

    new, old = run(), reference()
    for key in ("elbo", "log_likelihood", "kl"):
        assert torch.equal(new[key], old[key])


@pytest.mark.parametrize("family", ["vmf", "power_spherical", "gaussian"])
@pytest.mark.parametrize("reparameterize", [True, False])
def test_sample_with_log_prob_is_bit_identical_for_other_families(
    family: LatentFamily, reparameterize: bool
) -> None:
    dim = 4
    raw = torch.randn(
        6, head_size(family, dim), generator=torch.Generator().manual_seed(6)
    )
    posterior = posterior_from_raw(family, raw, dim)
    shape = torch.Size([5])
    torch.manual_seed(12)
    z, log_q = sample_with_log_prob(posterior, shape, reparameterize=reparameterize)
    torch.manual_seed(12)
    z_ref = posterior.rsample(shape) if reparameterize else posterior.sample(shape)
    assert torch.equal(z, z_ref)
    assert log_q is not None
    assert torch.equal(log_q, posterior.log_prob(z_ref))
    torch.manual_seed(12)
    z0, none = sample_with_log_prob(posterior, shape, need_log_prob=False)
    assert none is None
    assert torch.equal(z0, z_ref)


@pytest.mark.parametrize("family", ["vmf", "power_spherical"])
def test_sample_and_kl_is_bit_identical_for_closed_form_families(
    family: LatentFamily,
) -> None:
    dim = 4
    raw = torch.randn(
        6, head_size(family, dim), generator=torch.Generator().manual_seed(7)
    )
    posterior = posterior_from_raw(family, raw, dim)
    prior = standard_prior(family, dim, torch.device("cpu"))
    torch.manual_seed(13)
    z, kl = sample_and_kl(posterior, prior)
    torch.manual_seed(13)
    assert torch.equal(z, posterior.rsample())
    assert torch.equal(kl, kl_divergence(posterior, prior))


def test_tnbbeta_helper_log_prob_is_the_method_output() -> None:
    posterior = _tnbbeta(0.2, 0.1, 1e-3, 5)
    torch.manual_seed(14)
    z, log_q = sample_with_log_prob(posterior, torch.Size([7]))
    torch.manual_seed(14)
    z_ref, log_q_ref = posterior.rsample_with_log_prob(torch.Size([7]))
    assert torch.equal(z, z_ref)
    assert log_q is not None
    assert torch.equal(log_q, log_q_ref)


# ---- (iii) gradients -------------------------------------------------------


@pytest.mark.parametrize("eps_raw", [-12.0, -7.0, -3.0, 0.0, 2.0])
def test_gradients_flow_and_are_finite_through_the_exact_elbo(eps_raw: float) -> None:
    torch.manual_seed(15)
    dim = 3
    rows = 16
    raw = torch.randn(rows, dim + 3, dtype=F64)
    raw[:, -1] = eps_raw
    raw.requires_grad_(True)
    posterior = posterior_from_raw("tnbbeta", raw, dim)
    prior = standard_prior("tnbbeta", dim, torch.device("cpu"))
    x = torch.randn(rows, 2, dtype=F64)
    w = torch.randn(dim, 2, dtype=F64)
    terms = monte_carlo_elbo(x, posterior, prior, lambda z: z @ w, num_samples=4)
    (-terms["elbo"]).sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad.abs().sum()) > 0
    # the epsilon column receives gradient too
    assert float(raw.grad[:, -1].abs().sum()) > 0


def test_mlp_vae_training_step_uses_the_exact_log_q() -> None:
    """End to end: a TNBBeta MlpVAE at the epsilon floor reports the exact huge KL."""
    torch.manual_seed(16)
    model = MlpVAE(MlpVAEConfig(family="tnbbeta", latent_dim=3, input_dim=6))
    with torch.no_grad():
        model.posterior_head.bias[-1] = -30.0  # epsilon -> its floor, about 1e-4
    x = torch.randn(64, 6)
    out = model.training_step(x)
    out["loss"].backward()
    assert float(out["kl"]) > 100.0  # the vector path gave a few tens at most
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.posterior_head.parameters()
    )
