"""Tests for the four-class ring task's sampler, exact posterior and KL."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats
import torch

from tnbbeta_vae.data.ring_classes import (
    BAND_EDGES,
    NUM_CLASSES,
    exact_class_kl,
    ring_class_task,
)
from tnbbeta_vae.distributions import VonMisesFisher


def _draw(design: int, num: int, seed: int = 0):
    task = ring_class_task(design)
    generator = torch.Generator().manual_seed(seed)
    return task, *task.sample(num, generator)


def test_posterior_normalises_against_scipy_quad() -> None:
    task, x, _, cls, _ = _draw(2, 4)
    x = x.double()
    for i in range(4):

        def surface_density(w: float, i: int = i) -> float:
            value = task.log_posterior_at_latitude(
                x[i : i + 1], torch.tensor([w], dtype=torch.float64)
            )
            return float(value.exp())

        # dA = dw dphi on S^2, so the posterior integrates to 2 pi * int p(w) dw.
        total, _ = integrate.quad(surface_density, -1, 1, limit=200)
        assert abs(2 * math.pi * total - 1.0) < 1e-4
    theta, density = task.latitude_density(x)
    assert torch.allclose(
        (density * math.pi / task.grid_size).sum(-1), torch.ones(4, dtype=torch.float64)
    )
    del theta, cls


def test_grid_has_at_least_4001_nodes() -> None:
    assert ring_class_task(1).grid_size >= 4001


def test_prior_samples_are_uniform_on_the_sphere_in_both_designs() -> None:
    for design in (1, 2):
        _, _, z, _, w = _draw(design, 6000, seed=3)
        assert torch.allclose(z.norm(dim=-1), torch.ones(6000))
        assert stats.kstest(w.numpy(), stats.uniform(-1, 2).cdf).pvalue > 1e-3
        for axis in range(3):
            assert (
                stats.kstest(z[:, axis].numpy(), stats.uniform(-1, 2).cdf).pvalue > 1e-3
            )


def test_posterior_sampler_matches_density_via_ks() -> None:
    task, x, _, cls, _ = _draw(2, 4, seed=5)
    x = x.double()
    generator = torch.Generator().manual_seed(11)
    z = task.sample_posterior(x, cls, 4000, generator)
    axis = task.axis_of(cls)
    assert z.shape == (4000, 4, 3)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4000, 4, dtype=torch.float64))
    angle = torch.arccos(((z * axis).sum(-1)).clamp(-1, 1))
    for i in range(4):
        # Evaluate the exact latitude CDF at the sample's angles, which must be uniform.
        u = task.latitude_cdf(x[i : i + 1], angle[:, i])[0]
        assert stats.kstest(u.numpy(), "uniform").pvalue > 1e-3


def test_joint_sampler_is_calibrated_against_the_exact_posterior() -> None:
    # The true latitude's posterior CDF value is Uniform(0, 1) if (x, z) is drawn from
    # the model whose posterior the quadrature computes (a probability integral
    # transform).
    for design in (1, 2):
        task, x, _, _, w = _draw(design, 3000, seed=7)
        angle = torch.arccos(w.double().clamp(-1, 1))
        cdf = torch.stack(
            [
                task.latitude_cdf(x[i : i + 1].double(), angle[i : i + 1])[0, 0]
                for i in range(600)
            ]
        )
        assert stats.kstest(cdf.numpy(), "uniform").pvalue > 1e-3


def test_kl_uniform_to_posterior_matches_closed_form_and_monte_carlo() -> None:
    task, x, _, _, _ = _draw(1, 3, seed=2)
    x = x.double()
    exact = task.kl_uniform_to_posterior(x)
    # Closed form with scipy: E_U[E] + log E_U[exp(-E)] with E = ||x - h(w)||^2 / 2s^2.
    for i in range(3):

        def energy(w: float, i: int = i) -> float:
            value = task.features(torch.tensor([w], dtype=torch.float64))[0]
            return float(((x[i] - value) ** 2).sum()) / (2 * task.sigma**2)

        mean_energy = integrate.quad(energy, -1, 1)[0] / 2
        evidence = integrate.quad(lambda w: math.exp(-energy(w)), -1, 1)[0] / 2
        assert abs(float(exact[i]) - (mean_energy + math.log(evidence))) < 1e-6
    # Monte Carlo KL(U || p) = E_U[log U - log p].
    generator = torch.Generator().manual_seed(4)
    z = torch.randn(300_000, 3, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    cls = torch.zeros(3, dtype=torch.long)
    log_p = task.log_posterior(x, z[:, None, :].expand(-1, 3, -1), cls)
    kl = -math.log(4 * math.pi) - log_p
    standard_error = kl.std(0) / math.sqrt(kl.shape[0])
    assert (kl.mean(0) - exact).abs().max() < float(4 * standard_error.max()) + 1e-3


def test_evidence_is_consistent_with_bayes_rule_and_monte_carlo() -> None:
    for design in (1, 2):
        task, x, _, cls, _ = _draw(design, 5, seed=1)
        x = x.double()
        generator = torch.Generator().manual_seed(9)
        z = torch.randn(7, 5, 3, generator=generator, dtype=torch.float64)
        z = z / z.norm(dim=-1, keepdim=True)
        log_evidence = task.log_evidence(x)
        # log p(x) = log p(x | z) + log p(z) - log p(z | x) for every z.
        identity = (
            task.log_likelihood(x, z, cls)
            - math.log(4 * math.pi)
            - task.log_posterior(x, z, cls)
        )
        assert torch.allclose(identity, log_evidence.expand_as(identity), atol=1e-6)
    z = torch.randn(1_000_000, 3, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    task, x, _, cls, _ = _draw(1, 1, seed=1)
    likelihood = task.log_likelihood(
        x.double().expand(z.shape[0], -1), z, cls.expand(z.shape[0])
    ).exp()
    estimate = math.log(float(likelihood.mean()))
    assert abs(estimate - float(task.log_evidence(x.double())[0])) < 0.02


def test_design_one_bands_have_equal_mass_and_uniform_aggregate() -> None:
    task, x, z, cls, w = _draw(1, 40_000, seed=13)
    assert torch.equal(cls, task.band_of(w))
    assert torch.allclose(w, z[:, 0])
    counts = torch.bincount(cls, minlength=NUM_CLASSES).double() / 40_000
    assert (counts - 0.25).abs().max() < 0.01
    # Each band's w is uniform on its edges.
    for c in range(NUM_CLASSES):
        lo, hi = BAND_EDGES[c], BAND_EDGES[c + 1]
        wc = w[cls == c].numpy()
        assert wc.min() >= lo and wc.max() <= hi
        assert stats.kstest(wc, stats.uniform(lo, hi - lo).cdf).pvalue > 1e-3
    assert stats.kstest(w.numpy(), stats.uniform(-1, 2).cdf).pvalue > 1e-3
    assert task.input_dim == 3
    assert x.shape == (40_000, 3)


def test_design_two_axis_selection_and_class_independence() -> None:
    task, x, z, cls, w = _draw(2, 40_000, seed=17)
    axes = task.axes()
    assert torch.allclose(axes.norm(dim=-1), torch.ones(4, dtype=torch.float64))
    gram = axes @ axes.T
    off_diagonal = gram[~torch.eye(4, dtype=torch.bool)]
    assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, -1 / 3))
    # The latitude uses the class's own axis.
    expected = (z.double() * axes[cls]).sum(-1).float()
    assert torch.allclose(w, expected, atol=1e-6)
    decoded = task.decode(z.double(), cls)
    assert torch.allclose(decoded, task.features(w.double()), atol=1e-6)
    assert torch.allclose(task.encoder_input(x, cls)[:, 3:].sum(-1), torch.ones(40_000))
    assert task.input_dim == 7
    # The class is uniform and independent of z: z's distribution is the same per class.
    counts = torch.bincount(cls, minlength=NUM_CLASSES).double() / 40_000
    assert (counts - 0.25).abs().max() < 0.01
    for axis in range(3):
        samples = [z[cls == c, axis].numpy() for c in range(NUM_CLASSES)]
        for other in samples[1:]:
            assert stats.ks_2samp(samples[0], other).pvalue > 1e-3  # pyright: ignore[reportAttributeAccessIssue]
    # ... but x depends on the class: the same z maps to different x across classes.
    balanced = task.sample(100, torch.Generator().manual_seed(0), balanced=True)
    assert torch.equal(balanced[2], torch.arange(100) % 4)


def test_exact_class_kl_is_positive_for_a_cap_and_matches_uniform_kl_for_a_flat_q() -> (
    None
):
    task, x, _, cls, _ = _draw(2, 6, seed=21)
    x = x.double()
    # A concentrated cap is far from a ring: KL clearly positive.
    axis = task.axis_of(cls)
    cap = VonMisesFisher(axis, torch.full((6, 1), 20.0, dtype=torch.float64))
    kl, se = exact_class_kl(cap, task, x, cls, num_samples=1000)
    assert (kl - 4 * se > 0).all()
    # A vMF with kappa ~ 0 is the uniform distribution, so its KL is KL(U || p).
    near_uniform = VonMisesFisher(axis, torch.full((6, 1), 1e-3, dtype=torch.float64))
    kl_u, se_u = exact_class_kl(near_uniform, task, x, cls, num_samples=4000)
    gap = (kl_u - task.kl_uniform_to_posterior(x)).abs()
    assert (gap < 5 * se_u + 0.05).all()


def test_posterior_has_no_second_mode_above_five_percent() -> None:
    # Pre-registered claim (illustration script): 0.000 of 2000 test x have a second
    # latitude mode with more than 5% of the mass.
    for design in (1, 2):
        task, x, _, _, _ = _draw(design, 20_000, seed=29)
        mass = task.second_mode_mass(x.double())
        assert float((mass > 0.05).double().mean()) == 0.0
    assert np.isfinite(mass.numpy()).all()
