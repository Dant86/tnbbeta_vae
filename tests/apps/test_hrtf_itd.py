"""Tests for the pure helpers of apps.synthetic.hrtf_itd."""

from __future__ import annotations

import math

import numpy as np
import torch

from apps.synthetic import hrtf_itd as hi
from tnbbeta_vae.data.hrtf import HrtfTable, make_split


def _table(num_subjects: int = 7, rows: int = 60) -> HrtfTable:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(num_subjects * rows, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    itd = 600 * z[:, 1] + rng.normal(scale=20, size=len(z))
    subject = np.repeat(np.arange(3, 3 + num_subjects), rows)
    weight = np.full(len(z), 1.0 / rows)
    return HrtfTable(subject, z, itd, weight)


def test_exact_kl_matches_two_dimensional_quadrature() -> None:
    torch.manual_seed(0)
    model = hi.HrtfModel("V", "vmf").double()
    sigma = 0.4
    x = torch.tensor([[0.3], [-1.0]], dtype=torch.float64)
    raw = torch.tensor([[0.5, 1.0, -0.2, 2.0], [-1.0, 0.3, 0.8, 1.0]]).double()
    posterior = hi.posterior_from_raw("vmf", raw, 3)
    nodes = -1 + (torch.arange(4001, dtype=torch.float64) + 0.5) * (2 / 4001)
    with torch.no_grad():
        g_nodes = model.g_of_latitude(nodes)
        kl, se = hi._exact_kl(model, posterior, x, g_nodes, sigma, 4001, 40_000)
        # Independent: quadrature over (cos t, phi) on a fine grid.
        ct = -1 + (torch.arange(400, dtype=torch.float64) + 0.5) * (2 / 400)
        phi = (torch.arange(400, dtype=torch.float64) + 0.5) * (2 * math.pi / 400)
        st = (1 - ct**2).sqrt()
        z = torch.stack(
            [
                st[:, None] * phi.cos()[None],
                st[:, None] * phi.sin()[None],
                ct[:, None].expand(-1, 400),
            ],
            -1,
        ).reshape(-1, 3)
        area = (2 / 400) * (2 * math.pi / 400)
        for i in range(2):
            energy = (x[i, 0] - model.decode(z)[:, 0]).pow(2) / (2 * sigma**2)
            log_z = torch.log((-energy).exp().sum() * area)
            log_p = -energy - log_z
            q = posterior.log_prob(z[:, None, :].expand(-1, 2, -1))[:, i]
            reference = (q.exp() * (q - log_p)).sum() * area
            assert abs(float(kl[i]) - float(reference)) < 4 * float(se[i]) + 2e-3


def test_evaluate_and_summarize_run_on_a_tiny_split() -> None:
    split = make_split(_table(), 0)
    for arm in ("V", "L"):
        torch.manual_seed(0)
        model = hi.HrtfModel(arm, "tnbbeta").double().eval()
        arrays, extra = hi.evaluate(model, split, 20, 50, 201, chunk=64)
        summary = hi.summarize_run(arrays, extra, split)
        n = len(split.test_weight)
        assert arrays["abs_cos"].shape == (n,)
        assert 0 <= summary["cov50"] <= 1 and 0 <= summary["cov95"] <= 1
        assert summary["metric_name"] == ("kl" if arm == "V" else "nll")
        assert math.isfinite(summary["metric"])
        assert set(summary["baselines"]) == {"uniform", "oracle_ring", "oracle_cap"}
        assert summary["baselines"]["oracle_ring"] < summary["baselines"]["uniform"]


def test_axis_sign_follows_training_correlation() -> None:
    split = make_split(_table(), 1)
    model = hi.HrtfModel("V", "vmf").double()
    with torch.no_grad():
        model.raw_u.copy_(torch.tensor([0.0, 1.0, 0.0]))
    assert hi._axis_sign(model, split) == 1.0
    with torch.no_grad():
        model.raw_u.copy_(torch.tensor([0.0, -1.0, 0.0]))
    assert hi._axis_sign(model, split) == -1.0


def test_enumerated_wilcoxon_matches_scipy() -> None:
    diff = np.array([0.3, 0.1, -0.05, 0.4, 0.2, 0.15, 0.35, 0.25, 0.5, 0.45])
    assert (
        abs(hi._wilcoxon_greater(diff) - hi._wilcoxon_greater_enumerated(diff)) < 1e-9
    )
    assert hi._wilcoxon_greater_enumerated(np.ones(10)) == 2**-10


def test_family_spec_and_run_path(tmp_path) -> None:  # noqa: ANN001
    assert hi.family_spec("tnbbeta_init0") == {
        "family": "tnbbeta",
        "p_init_offset": 0.0,
    }
    assert hi.run_path(tmp_path, "xcorr", "V", "vmf", 3) == (
        tmp_path / "xcorr" / "armV" / "vmf" / "s3"
    )
