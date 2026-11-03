"""Tests for the linear-decoder ring study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.synthetic import linear_decoder_ring as study


def _tiny_train(
    tmp_path: Path, decoder: str, dim: int, arm: str, seed: int = 0
) -> Path:
    study.main(
        [
            "train", "--decoder", decoder, "--dim", str(dim), "--sigma", "0.2",
            "--arm", arm, "--seed", str(seed), "--epochs", "2", "--num-train", "256",
            "--num-test", "100", "--device", "cpu", "--out-dir", str(tmp_path),
        ]
    )  # fmt: skip
    return study.run_path(tmp_path, decoder, dim, 0.2, arm, seed)


@pytest.mark.parametrize("decoder", study.DECODERS)
@pytest.mark.parametrize("arm", study.ARMS)
def test_model_shapes_and_loss(decoder: str, arm: str) -> None:
    torch.manual_seed(0)
    model = study.LinearRingVAE(decoder, 5, arm)
    y = torch.randn(8)
    z = model.posterior(y).rsample(torch.Size([3]))
    assert z.shape == (3, 8, 5)
    assert model.decode(z).shape == (3, 8, 1)
    assert model.log_likelihood(y, z).shape == (3, 8)
    loss = model.negative_elbo(y, 1.0, 2)
    assert torch.isfinite(loss)
    loss.backward()


def test_linear_decoder_is_the_normalised_axis() -> None:
    model = study.LinearRingVAE("L", 5, "vmf")
    z = torch.nn.functional.normalize(torch.randn(4, 5), dim=-1)
    expected = (z * model.axis()).sum(-1, keepdim=True)
    assert torch.allclose(model.decode(z), expected)
    assert float(model.axis().detach().norm()) == pytest.approx(1.0)
    assert not any(name == "bias" for name, _ in model.named_parameters())


def test_initial_kappa_is_dimension_aware() -> None:
    model = study.LinearRingVAE("L", 9, "vmf")
    y = torch.zeros(2)
    kappa = model.posterior(y).scale  # pyright: ignore[reportAttributeAccessIssue]
    assert torch.allclose(kappa, torch.full_like(kappa, 9.0), atol=2.0)


def test_encoder_decoder_init_identical_across_arms() -> None:
    torch.manual_seed(3)
    first = study.LinearRingVAE("L", 5, "vmf")
    torch.manual_seed(3)
    second = study.LinearRingVAE("L", 5, "tnbbeta")
    assert torch.equal(first.axis_raw, second.axis_raw)
    for key, value in first.encoder.state_dict().items():
        assert torch.equal(value, second.encoder.state_dict()[key])


def test_grid_has_960_unique_cells_with_l_first() -> None:
    cells = study.grid_cells()
    assert len(cells) == 960
    assert len(set(cells)) == 960
    assert all(c[0] == "L" for c in cells[:480])
    assert {c[2] for c in cells if c[0] != "L"} == {0.2}


@pytest.mark.parametrize(
    ("decoder", "arm"),
    [("L", "tnbbeta"), ("F", "vmf"), ("M", "power_spherical"), ("Lb", "tnbbeta_init0")],
)
def test_tiny_run_writes_outputs(tmp_path: Path, decoder: str, arm: str) -> None:
    stem = _tiny_train(tmp_path, decoder, 5, arm)
    record = json.loads(stem.with_suffix(".json").read_text())
    assert not record["diverged"]
    assert 0 <= record["abs_cos_axis"] <= 1
    assert record["gap_elbo"] > -0.5
    if decoder != "F":
        gap = record["model_misfit"] + record["kl_to_model_posterior"]
        assert gap == pytest.approx(record["gap_elbo"], abs=1e-9)
    arrays = np.load(stem.with_suffix(".npz"))
    assert arrays["elbo"].shape == (100,)
    assert stem.with_suffix(".pt").exists()


def test_summarize_runs_on_tiny_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for arm in study.ARMS:
        for seed in range(5):
            _tiny_train(tmp_path, "L", 3, arm, seed)
    study.summarize(tmp_path)
    out = capsys.readouterr().out
    assert "Confirmatory tests" in out
    assert "WARNING" not in out
    runs = study.load_runs(tmp_path)
    assert len(runs) == 4
    tests = study.confirmatory_tests(runs)
    assert len(tests) == 2
    assert all(abs(t["p"] - t["p_enum"]) < 1e-9 for t in tests)


def test_wilcoxon_enumeration_matches_scipy() -> None:
    diff = np.array([0.3, 0.1, -0.05, 0.4, 0.22, 0.15, -0.01, 0.5, 0.31, 0.07])
    assert study.wilcoxon_greater(diff) == pytest.approx(
        study.wilcoxon_greater_enumerated(diff)
    )


def test_learned_axis_of_free_decoder_recovers_linear_map() -> None:
    model = study.LinearRingVAE("F", 5, "vmf").double()
    linear = torch.nn.Linear(5, 1, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]))
    model.decoder_net = torch.nn.Sequential(linear)
    axis, fraction = study.learned_axis(model)
    assert abs(float(axis[2])) == pytest.approx(1.0, abs=1e-9)
    assert fraction == pytest.approx(1.0)
