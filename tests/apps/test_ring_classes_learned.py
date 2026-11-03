"""Tests for the learned-decoder four-class ring study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.synthetic import ring_classes_learned as study


def _tiny_train(
    tmp_path: Path, setting: str, decoder: str, arm: str, seed: int
) -> None:
    study.main(
        [
            "train",
            "--setting",
            setting,
            "--decoder",
            decoder,
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


@pytest.mark.parametrize("setting", study.SETTINGS)
@pytest.mark.parametrize("decoder", study.DECODERS)
@pytest.mark.parametrize("arm", ["vmf", "tnbbeta"])
def test_model_shapes_and_loss(setting: str, decoder: str, arm: str) -> None:
    torch.manual_seed(0)
    model = study.LearnedRingVAE(setting, decoder, **study.arm_spec(arm))
    x = torch.randn(8, 3)
    cls = torch.arange(8) % 4
    posterior = model.posterior(x, cls)
    z = posterior.rsample(torch.Size([5]))
    assert model.decode(z, cls).shape == (5, 8, 3)
    assert model.log_likelihood(x, z, cls).shape == (5, 8)
    loss = model.negative_elbo(x, cls, 1.0, 2)
    assert torch.isfinite(loss)
    loss.backward()


def test_encoder_and_decoder_inits_match_across_families() -> None:
    weights = []
    for arm in ("vmf", "tnbbeta"):
        torch.manual_seed(3)
        model = study.LearnedRingVAE("D2g", "mlp", **study.arm_spec(arm))
        weights.append(
            (
                model.encoder.state_dict()["0.weight"],
                model.decoder_net.state_dict()["0.0.weight"],
            )
        )
    assert torch.equal(weights[0][0], weights[1][0])
    assert torch.equal(weights[0][1], weights[1][1])


def test_decoder_single_index_recovers_a_known_axis() -> None:
    torch.manual_seed(0)
    model = study.LearnedRingVAE("D1", "single_index", "vmf").double()
    with torch.no_grad():
        model.axis_raw.copy_(torch.tensor([[0.3, -0.5, 0.8]]))
    u_star, eta2 = study.decoder_single_index(model, per_class=False)
    truth = torch.nn.functional.normalize(model.axis_raw, dim=-1).detach().numpy()
    assert abs(float((u_star * truth).sum())) > 0.99
    assert eta2[0] > 0.95


def test_decoder_single_index_of_a_two_axis_decoder_is_lower() -> None:
    torch.manual_seed(1)
    model = study.LearnedRingVAE("D1", "mlp", "vmf").double()
    _, eta2 = study.decoder_single_index(model, per_class=False)
    assert 0.0 < eta2[0] <= 1.0


def test_enumerated_wilcoxon_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        diff = rng.normal(0.3, 1.0, 10)
        assert study._wilcoxon_greater(diff) == pytest.approx(
            study._wilcoxon_greater_enumerated(diff), abs=1e-9
        )


def test_train_writes_outputs_and_summarize_reports_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tiny_train(tmp_path, "D2l", "mlp", "vmf", 0)
    stem = study.run_path(tmp_path, "D2l", "mlp", "vmf", 0)
    for suffix in (".json", ".npz", ".pt"):
        assert stem.with_suffix(suffix).exists()
    record = json.loads(stem.with_suffix(".json").read_text())
    assert record["diverged"] is False
    assert "knn_accuracy" in record
    assert record["sigma"] > 0
    arrays = np.load(stem.with_suffix(".npz"))
    assert arrays["elbo"].shape == (100,)
    assert arrays["log_ref"].shape == (100,)
    rng = np.random.default_rng(0)
    for arm, shift in (("vmf", 0.3), ("power_spherical", 0.2), ("tnbbeta", 0.0)):
        for seed in range(10):
            copy = dict(record, arm=arm, seed=seed)
            copy["gap_elbo"] = 1.0 + shift + 0.01 * rng.normal()
            copy["gap_iwae"] = copy["gap_elbo"] - 0.1
            out = study.run_path(tmp_path, "D2l", "mlp", arm, seed).with_suffix(".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(copy))
    study.summarize(tmp_path)
    text = capsys.readouterr().out
    assert "Holm reject" in text
    assert "| D2l | vmf | 10 |" in text
    assert "0.00098" in text
