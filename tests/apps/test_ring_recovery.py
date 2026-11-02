"""Tests for the pure helpers of apps.synthetic.ring_recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from apps.synthetic import ring_recovery as rr
from tnbbeta_vae.data.ring_posterior import (
    RingPosteriorTask,
    exact_kl,
    ring_task,
    sphere_log_area,
)
from tnbbeta_vae.distributions import TNBBetaSpherical


def test_latitude_density_matches_spherical_log_prob() -> None:
    dim = 6
    p, q, eps = (torch.tensor([v], dtype=torch.float64) for v in (0.7, 0.3, 2.5))
    theta = torch.tensor([0.4, 1.1, 2.0], dtype=torch.float64)
    mu = torch.zeros(1, dim, dtype=torch.float64)
    mu[0, 0] = 1.0
    z = torch.zeros(3, 1, dim, dtype=torch.float64)
    z[..., 0] = torch.cos(theta)[:, None]
    z[..., 1] = torch.sin(theta)[:, None]
    sphere = TNBBetaSpherical(mu, p, q, eps).log_prob(z)[:, 0]
    expected = (
        sphere + sphere_log_area(dim - 1) + (dim - 2) * torch.log(torch.sin(theta))
    )
    got = rr._tnbbeta_log_latitude_density(p, q, eps, theta)[0]  # noqa: SLF001
    assert torch.allclose(got, expected, atol=1e-9)


def test_quadrature_kl_matches_monte_carlo() -> None:
    dim = 5
    task = RingPosteriorTask(
        dim,
        0.4,
        rr._axis_at(dim, 0.0),  # noqa: SLF001
    )
    x = task.features(torch.tensor([math.cos(math.radians(60))], dtype=torch.float64))
    theta, delta = rr._grid(task)  # noqa: SLF001
    _, density = task.latitude_density(x)
    log_p = density[0].log()
    raw = torch.tensor(
        [[math.log(0.75 / 0.25), -3.0, math.log(12.0)]], dtype=torch.float64
    )
    quadrature = rr._tnbbeta_quadrature_kl(raw, theta, delta, log_p)[0]  # noqa: SLF001
    p, q, eps = rr._unpack_tnbbeta(raw[0])  # noqa: SLF001
    mu = torch.zeros(1, dim, dtype=torch.float64)
    mu[0, 0] = 1.0
    torch.manual_seed(0)
    mc, se = exact_kl(
        TNBBetaSpherical(mu, p.reshape(1), q.reshape(1), eps.reshape(1)),
        task,
        x,
        num_samples=300_000,
    )
    assert abs(float(mc[0]) - float(quadrature)) < 5 * float(se[0]) + 2e-3


def test_truth_stats_ordering_and_fixed_epsilon_variants() -> None:
    task = ring_task(8, 0.2)
    x = task.features(torch.tensor([0.5, -0.2])).double()
    lo, hi, mean, sd = rr._truth_stats(task, x)  # noqa: SLF001
    assert torch.all(lo < mean)
    assert torch.all(mean < hi)
    assert torch.all(sd > 0)
    assert rr._stage3_fixed_epsilon("vmf", task) is None  # noqa: SLF001
    assert rr._stage3_fixed_epsilon("tnbbeta", task) is None  # noqa: SLF001
    assert rr._stage3_fixed_epsilon("tnbbeta_fixed_m_neg", task) == 3.5 - 0.5  # noqa: SLF001


def test_stage3_new_options_run_and_record(tmp_path: Path) -> None:
    """The new stage-3 options (samples, MC KL, warm-up) run and are recorded."""
    out = tmp_path / "stage3"
    args = [
        "stage3",
        "--dim",
        "3",
        "--condition",
        "learned",
        "--family",
        "vmf",
        "--seed",
        "0",
        "--epochs",
        "2",
        "--num-train",
        "256",
        "--num-test",
        "20",
        "--device",
        "cpu",
        "--num-elbo-samples",
        "3",
        "--mc-kl",
        "--warmup-epochs",
        "1",
        "--out-dir",
        str(out),
    ]
    rr.main(args)
    result = json.loads((out / "d3_learned_vmf_s0.json").read_text())
    assert result["num_elbo_samples"] == 3
    assert result["mc_kl"] is True
    assert len(result["loss_curve_full"]) == 2
    assert {"recon_model", "kl_prior_model_mc"} <= set(result)
    assert math.isclose(
        result["elbo_model"],
        result["recon_model"] - result["kl_prior_model_mc"],
        abs_tol=1e-4,
    )
