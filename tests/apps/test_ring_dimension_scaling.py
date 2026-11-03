"""Tests for the ring dimension-scaling study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.synthetic import ring_dimension_scaling as study
from tnbbeta_vae.data.ring_dimension import dimension_ring_task


def _tiny_train(tmp_path: Path, condition: str, dim: int, arm: str, seed: int) -> None:
    study.main(
        [
            "train",
            "--condition",
            condition,
            "--dim",
            str(dim),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--epochs",
            "2",
            "--num-train",
            "256",
            "--num-test",
            "100",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
        ]
    )


@pytest.mark.parametrize("condition", study.CONDITIONS)
@pytest.mark.parametrize("dim", [3, 9])
@pytest.mark.parametrize("arm", study.ARMS)
def test_model_shapes_and_loss(condition: str, dim: int, arm: str) -> None:
    torch.manual_seed(0)
    model = study.DimensionRingVAE(condition, dim, arm, dimension_ring_task(dim))
    x = torch.randn(8, 3)
    posterior = model.posterior(x)
    z = posterior.rsample(torch.Size([5]))
    assert z.shape == (5, 8, dim)
    assert model.decode(z).shape == (5, 8, 3)
    assert model.log_likelihood(x, z).shape == (5, 8)
    loss = model.negative_elbo(x, 1.0, 2)
    assert torch.isfinite(loss)
    loss.backward()


def test_condition_a_decoder_is_the_true_one_and_frozen() -> None:
    task = dimension_ring_task(5)
    model = study.DimensionRingVAE("A", 5, "vmf", task)
    z = torch.randn(4, 5)
    z = z / z.norm(dim=-1, keepdim=True)
    assert torch.allclose(model.decode(z), task.decode(z))
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith(("decoder", "link", "learned_scale")) for n in names)
    assert float(model.sigma()) == pytest.approx(0.2)


def test_encoder_and_decoder_inits_match_across_arms() -> None:
    task = dimension_ring_task(9)
    reference: dict[str, torch.Tensor] | None = None
    for arm in study.ARMS:
        torch.manual_seed(3)
        model = study.DimensionRingVAE("B", 9, arm, task)
        state = {
            k: v
            for k, v in model.state_dict().items()
            if k.startswith(("encoder", "decoder"))
        }
        if reference is None:
            reference = state
        for key, value in state.items():
            assert torch.equal(value, reference[key])


def test_dimension_aware_head_inits() -> None:
    task = dimension_ring_task(9)
    torch.manual_seed(0)
    vmf = study.DimensionRingVAE("B", 9, "vmf", task)
    kappa = vmf.posterior(torch.zeros(2, 3)).scale  # pyright: ignore[reportAttributeAccessIssue]
    assert torch.allclose(kappa, torch.full_like(kappa, 9.0), atol=0.5)
    torch.manual_seed(0)
    eps = study.DimensionRingVAE("B", 9, "tnbbeta_epsinit", task)
    epsilon = eps.posterior(torch.zeros(2, 3)).epsilon  # pyright: ignore[reportAttributeAccessIssue]
    assert torch.allclose(epsilon, torch.full_like(epsilon, 4.0), atol=0.5)
    torch.manual_seed(0)
    plain = study.DimensionRingVAE("B", 9, "tnbbeta", task)
    assert float(plain.posterior(torch.zeros(2, 3)).epsilon.max()) < 2.0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("condition", study.CONDITIONS)
@pytest.mark.parametrize("arm", ["vmf", "tnbbeta"])
def test_tiny_train_writes_a_finite_summary(
    tmp_path: Path, condition: str, arm: str
) -> None:
    _tiny_train(tmp_path, condition, 5, arm, 0)
    stem = study.run_path(tmp_path, condition, 5, arm, 0)
    record = json.loads(stem.with_suffix(".json").read_text())
    assert not record["diverged"]
    assert np.isfinite(record["gap_elbo"])
    assert 0.0 <= record["ring_r_median"] <= 1.0
    if condition == "A":
        assert np.isfinite(record["kl_exact"])
    runs = study.load_runs(tmp_path)
    assert (condition, 5, arm) in runs


def test_decoder_axis_recovers_a_planted_single_index_decoder() -> None:
    task = dimension_ring_task(9)
    model = study.DimensionRingVAE("B", 9, "vmf", task).double()
    planted = torch.zeros(9, dtype=torch.float64)
    planted[2] = 1.0
    layers = [m for m in model.decoder_net.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for layer in layers:
            layer.weight.zero_()
            layer.bias.zero_()
        layers[0].weight[0] = planted
        layers[1].weight[0, 0] = 1.0
        layers[2].weight[:, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    axis, fraction = study.decoder_axis(model)
    assert abs(float(axis[2])) > 0.99
    assert fraction > 0.95


def test_wilcoxon_scipy_matches_enumeration() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        diff = rng.normal(0.3, 1.0, 10)
        assert study.wilcoxon_greater(diff) == pytest.approx(
            study.wilcoxon_greater_enumerated(diff), abs=1e-9
        )


def test_holm_family_and_margin() -> None:
    runs: study.Runs = {}
    for cond in ("A",):
        for dim in study.DIMENSIONS:
            for arm, base in (("tnbbeta", 0.0), ("vmf", 0.5), ("power_spherical", 0.5)):
                runs[(cond, dim, arm)] = {
                    s: {"diverged": False, "gap_elbo": base + 0.01 * s + 0.001 * dim}
                    for s in range(10)
                }
    tests = study.confirmatory_tests(runs, "A")
    assert len(tests) == 8
    assert all(t["confirmed"] for t in tests)
    assert all(t["p"] == pytest.approx(1 / 1024) for t in tests)


def test_exact_vmf_sampler_matches_the_theoretical_mean_cosine() -> None:
    from scipy.special import ive

    torch.manual_seed(0)
    dim, kappa = 17, 17.0
    loc = torch.zeros(1, dim, dtype=torch.float64)
    loc[0, 0] = 1.0
    scale = torch.full((1, 1), kappa, dtype=torch.float64)
    z = study.ExactVonMisesFisher(loc, scale).sample(torch.Size([60_000]))
    theory = ive(dim / 2, kappa) / ive(dim / 2 - 1, kappa)
    assert float(z[..., 0].mean()) == pytest.approx(theory, abs=0.004)
