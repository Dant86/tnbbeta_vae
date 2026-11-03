"""Tests for apps.synthetic.hrtf_itd_armv_followup (small synthetic inputs, no h5py)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from apps.synthetic import hrtf_itd_armv_followup as fu
from tnbbeta_vae.data.hrtf import HrtfTable, make_split
from tnbbeta_vae.models.heads import posterior_from_raw, tnbbeta_posterior


def _table(num_subjects: int = 7, rows: int = 60) -> HrtfTable:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(num_subjects * rows, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    itd = 600 * z[:, 1] + rng.normal(scale=20, size=len(z))
    subject = np.repeat(np.arange(3, 3 + num_subjects), rows)
    weight = np.full(len(z), 1.0 / rows)
    return HrtfTable(subject, z, itd, weight)


def test_fixed_axis_keeps_u_exactly_plus_y_through_training() -> None:
    split = make_split(_table(), 0)
    torch.manual_seed(0)
    model = fu.FollowupModel("fixed_axis", "vmf")
    assert "raw_u" not in dict(model.named_parameters())
    before = [p.detach().clone() for p in model.g.parameters()]
    sigma0 = float(model.learned_scale().detach())
    assert math.isclose(sigma0, fu.INITIAL_SIGMA, rel_tol=1e-6)
    fu.train_model(model, split, 3, 2, 64, 2, torch.device("cpu"))
    assert torch.equal(model.u, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(
        model.double().u, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.g.parameters(), strict=True)
    )
    assert float(model.learned_scale().detach()) != sigma0


def test_learned_axis_moves_u_and_shares_initialisation() -> None:
    split = make_split(_table(), 0)
    torch.manual_seed(1)
    learned = fu.FollowupModel("learned_axis", "tnbbeta")
    torch.manual_seed(1)
    fixed = fu.FollowupModel("fixed_axis", "tnbbeta")
    fixed_params = dict(fixed.named_parameters())
    for name, a in learned.named_parameters():
        if name != "raw_u":
            assert torch.equal(a, fixed_params[name])
    u0 = learned.u.detach().clone()
    fu.train_model(learned, split, 2, 2, 64, 2, torch.device("cpu"))
    assert not torch.equal(learned.u, u0)


@pytest.mark.parametrize("family", ["vmf", "tnbbeta"])
def test_iwae_bound_matches_quadrature_marginal(family: str) -> None:
    torch.manual_seed(0)
    model = fu.FollowupModel("learned_axis", family, initial_sigma=1.5).double().eval()  # pyright: ignore[reportArgumentType]
    x = torch.tensor([[0.3], [-0.8]], dtype=torch.float64)
    # Fixed, non-trivial posterior parameters via the head output.
    raw = torch.tensor(
        [[0.5, 1.0, -0.2, 0.3, -0.4, 0.2], [-1.0, 0.3, 0.8, -0.2, 0.5, 0.1]],
        dtype=torch.float64,
    )
    if family == "vmf":
        raw = torch.tensor(
            [[0.5, 1.0, -0.2, 2.0], [-1.0, 0.3, 0.8, 1.0]], dtype=torch.float64
        )
    model.posterior = lambda _x: (  # type: ignore[method-assign]
        posterior_from_raw("vmf", raw, 3)
        if family == "vmf"
        else tnbbeta_posterior(raw, 3, epsilon_floor=1e-4)
    )
    bound = fu.iwae_arrays(model, x, num_samples=100_000, chunk=2)["ll"]
    # Independent: log p(x) = log integral N(x; g(u . z), sigma^2) dz / (4 pi).
    n = 400
    ct = -1 + (torch.arange(n, dtype=torch.float64) + 0.5) * (2 / n)
    phi = (torch.arange(n, dtype=torch.float64) + 0.5) * (2 * math.pi / n)
    st = (1 - ct**2).sqrt()
    z = torch.stack(
        [
            st[:, None] * phi.cos()[None],
            st[:, None] * phi.sin()[None],
            ct[:, None].expand(-1, n),
        ],
        -1,
    ).reshape(-1, 3)
    area = (2 / n) * (2 * math.pi / n)
    sigma = float(model.learned_scale().detach())
    with torch.no_grad():
        mean = model.decode(z)[:, 0]
    for i in range(2):
        density = torch.distributions.Normal(mean, sigma).log_prob(x[i, 0]).exp()
        reference = math.log(float(density.sum()) * area / (4 * math.pi))
        assert abs(float(bound[i]) - reference) < 0.03
    # K = 64 bound is below the marginal (up to Monte Carlo noise) and finite.
    small = fu.iwae_arrays(model, x, num_samples=64, chunk=2)
    assert np.all(np.isfinite(small["ll"]))
    assert np.all(small["ll"] >= small["elbo"] - 0.5)


def test_collapsed_flag_threshold() -> None:
    assert fu.is_collapsed(0.5001)
    assert not fu.is_collapsed(0.5)
    assert not fu.is_collapsed(0.1)
    assert not fu.is_collapsed(float("nan"))


def _record(iwae: float, sigma: float, diverged: bool = False) -> dict[str, Any]:
    return {
        "iwae": iwae,
        "sigma": sigma,
        "collapsed": fu.is_collapsed(sigma),
        "diverged": diverged,
    }


def test_restricted_pair_selection() -> None:
    tnb = {
        0: _record(0, 0.1),
        1: _record(0, 0.9),
        2: _record(0, 0.2),
        3: _record(0, 0.1),
    }
    cap = {
        0: _record(0, 0.2),
        1: _record(0, 0.1),
        2: _record(0, 0.8),
        4: _record(0, 0.1),
    }
    assert fu.restricted_seeds(tnb, cap) == [0]
    tnb[3] = _record(0, 0.5)
    cap[3] = _record(0, 0.5)
    assert fu.restricted_seeds(tnb, cap) == [0, 3]


def _runs(gap: dict[str, float], noise: float = 0.0, n: int = 10) -> fu.Cell:
    rng = np.random.default_rng(0)
    runs: fu.Cell = {}
    for decoder in fu.DECODERS:
        base = rng.normal(size=n) * noise
        runs[(decoder, "tnbbeta")] = {s: _record(-1.0 + base[s], 0.2) for s in range(n)}
        for other in fu.COMPARATORS:
            runs[(decoder, other)] = {
                s: _record(-1.0 + base[s] - gap[other] - 0.001 * s, 0.2)
                for s in range(n)
            }
    return runs


def test_holm_and_margin_logic_on_synthetic_data() -> None:
    runs = _runs({"vmf": 0.05, "power_spherical": 0.05})
    tests = fu.paired_tests(runs)
    fu.apply_holm(tests)
    assert len(tests) == 4
    assert all(abs(t["p"] - 2**-10) < 1e-12 for t in tests)
    assert all(abs(t["p"] - t["p_enum"]) < 1e-9 for t in tests)
    assert all(t["holm_reject"] and t["margin"] for t in tests)
    assert sorted(t["threshold"] for t in tests) == pytest.approx(
        [0.0125, 0.05 / 3, 0.025, 0.05]
    )
    assert fu.verdict(tests)["rule_met"]
    # Significant but below the 0.02-nat margin: Holm passes, margin fails.
    tests = fu.paired_tests(_runs({"vmf": 0.05, "power_spherical": 0.005}))
    fu.apply_holm(tests)
    assert all(t["holm_reject"] for t in tests)
    decision = fu.verdict(tests)
    assert decision["holm_all"] and not decision["margin_all"]
    assert not decision["rule_met"]
    # TNBBeta worse than one comparator: Holm fails for it and stops the rest.
    tests = fu.paired_tests(_runs({"vmf": 0.05, "power_spherical": -0.05}))
    fu.apply_holm(tests)
    assert not fu.verdict(tests)["holm_all"]
    bad = [t for t in tests if t["other"] == "power_spherical"]
    assert all(not t["holm_reject"] for t in bad)


def test_diverged_run_counts_as_worst_and_restricted_tests_have_no_holm() -> None:
    runs = _runs({"vmf": 0.05, "power_spherical": 0.05})
    runs[("fixed_axis", "vmf")][3] = _record(float("nan"), 0.2, diverged=True)
    assert fu.bound_of(runs[("fixed_axis", "vmf")][3]) == float("-inf")
    runs[("learned_axis", "power_spherical")][0] = _record(-1.5, 0.9)
    restricted = fu.paired_tests(runs, restricted=True)
    sizes = {(t["decoder"], t["other"]): t["n"] for t in restricted}
    assert sizes[("learned_axis", "power_spherical")] == 9
    assert sizes[("fixed_axis", "vmf")] == 10
    assert all("threshold" not in t for t in restricted)


def test_wilcoxon_enumeration_edge_cases() -> None:
    assert fu.wilcoxon_greater(np.zeros(5)) == 1.0
    assert fu.wilcoxon_greater_enumerated(np.zeros(5)) == 1.0
    assert math.isnan(fu.wilcoxon_greater(np.zeros(0)))
    diff = np.array([0.3, 0.1, -0.05, 0.4, 0.2, 0.15, 0.35, 0.25, 0.5, 0.45])
    assert abs(fu.wilcoxon_greater(diff) - fu.wilcoxon_greater_enumerated(diff)) < 1e-9


def test_train_evaluate_and_bound_summary_on_a_tiny_split() -> None:
    split = make_split(_table(), 0)
    for decoder in fu.DECODERS:
        torch.manual_seed(0)
        model = fu.FollowupModel(decoder, "tnbbeta")
        fu.train_model(model, split, 1, 1, 64, 2, torch.device("cpu"))
        model.double().eval()
        arrays, extra = fu.evaluate(model, split, 20, 50, 201, chunk=64)
        summary = fu.summarize_run(arrays, extra, split)
        arrays.update(fu.iwae_arrays(model, split, 8))
        summary.update(fu.bound_summary(arrays, 8))
        assert math.isfinite(summary["iwae"])
        assert summary["iwae"] >= -summary["neg_elbo"] - 1.0
        assert 0 <= summary["cov50"] <= 1
        if decoder == "fixed_axis":
            assert summary["u_abs_cos_y"] == 1.0


def test_oriented_latitude_check_flips_a_mirrored_latent() -> None:
    n = 200
    itd = np.linspace(-1, 1, n)
    w_true = itd.copy()
    arrays = {
        "itd_us": itd,
        "weight": np.full(n, 1.0 / n),
        "lat_q50": -w_true,
        "lat_q025": -w_true - 0.1,
        "lat_q25": -w_true - 0.02,
        "lat_q75": -w_true + 0.02,
        "lat_q975": -w_true + 0.1,
        "w_true": w_true,
    }
    out = fu.oriented_latitude_check(arrays)
    assert out["flipped"] == 1.0
    assert out["cov50"] == pytest.approx(1.0) and out["cov95"] == pytest.approx(1.0)
    assert out["median_abs_error"] == pytest.approx(0.0)
    arrays["w_true"] = -w_true
    for key in ("lat_q50", "lat_q025", "lat_q25", "lat_q75", "lat_q975"):
        arrays[key] = -arrays[key]
    assert fu.oriented_latitude_check(arrays)["flipped"] == 0.0
