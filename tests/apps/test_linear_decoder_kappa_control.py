# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
"""Tests for the linear-decoder tuned-kappa control study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.synthetic import linear_decoder_kappa_control as kc


def test_arm_settings_parse() -> None:
    pin = kc.arm_settings("vmf_pin30", 5)
    assert pin["kappa_pin"] == 30.0 and pin["kappa_init"] is None
    init = kc.arm_settings("power_spherical_k20d", 9)
    assert init["kappa_init"] == 180.0 and init["epochs"] == 300
    fast = kc.arm_settings("vmf_H", 5)
    assert (
        fast["epochs"] == 600 and fast["head_lr"] == 1e-2 and fast["kappa_init"] == 5.0
    )
    assert kc.arm_settings("tnbbeta_H", 5)["base"] == "tnbbeta"
    with pytest.raises(ValueError):
        kc.arm_settings("nope", 3)


def test_grid_size_and_order() -> None:
    cells = kc.grid_cells()
    assert len(cells) == 17 * 10 * 12
    assert len(set(cells)) == len(cells)
    assert not cells[0][2].endswith("_H") and cells[-1][2].endswith("_H")


@pytest.mark.parametrize("family", ["vmf", "power_spherical"])
def test_pinned_posterior_has_constant_kappa(family: str) -> None:
    torch.manual_seed(0)
    model = kc.ControlVAE(5, f"{family}_pin30")
    assert model.posterior_head.out_features == 5
    posterior = model.posterior(torch.randn(7))
    kappa = posterior.kappa if hasattr(posterior, "kappa") else posterior.scale  # pyright: ignore[reportAttributeAccessIssue]
    assert torch.allclose(kappa.reshape(-1), torch.full((7,), 30.0))


def test_initial_kappa_is_applied_and_trainable() -> None:
    torch.manual_seed(0)
    model = kc.ControlVAE(5, "vmf_k5d")
    posterior = model.posterior(torch.zeros(3))
    assert torch.allclose(
        posterior.scale.reshape(-1), torch.full((3,), 25.0), rtol=0.05
    )  # pyright: ignore[reportAttributeAccessIssue]
    assert model.posterior_head.out_features == 6


def test_encoder_decoder_init_matches_parent_arm() -> None:
    torch.manual_seed(3)
    a = kc.ControlVAE(5, "vmf_pin10")
    torch.manual_seed(3)
    b = kc.ControlVAE(5, "vmf_H")
    assert torch.equal(a.axis_raw, b.axis_raw)
    assert torch.equal(a.encoder[0].weight, b.encoder[0].weight)  # pyright: ignore[reportAttributeAccessIssue]


def test_train_smoke_and_summarize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kc.main(
        [
            "train",
            "--dim",
            "3",
            "--sigma",
            "0.2",
            "--arm",
            "vmf_pin10",
            "--seed",
            "0",
            "--epochs",
            "2",
            "--num-train",
            "256",
            "--num-test",
            "50",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
        ]
    )
    record = json.loads((tmp_path / "L/d3_s0.2/vmf_pin10/s0.json").read_text())
    assert not record["diverged"] and len(record["trajectory"]["kappa"]) == 2
    assert record["trajectory"]["kappa"][0] == pytest.approx(10.0)
    kc.main(
        [
            "train",
            "--dim",
            "3",
            "--sigma",
            "0.2",
            "--arm",
            "tnbbeta_H",
            "--seed",
            "0",
            "--epochs",
            "2",
            "--num-train",
            "256",
            "--num-test",
            "50",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
        ]
    )
    record = json.loads((tmp_path / "L/d3_s0.2/tnbbeta_H/s0.json").read_text())
    assert set(record["trajectory"]) >= {"p", "q", "epsilon", "sigma_hat"}
    kc.summarize(tmp_path, tmp_path)
    assert "vmf_pin10" in capsys.readouterr().out


def _fake_runs(cap_offset: float) -> kc.Runs:
    rng = np.random.default_rng(0)
    runs: kc.Runs = {}
    for dim in kc.DIMS:
        for sigma in kc.SIGMAS:
            base = rng.uniform(0.01, 0.02, 10)
            arms = {
                "tnbbeta": base,
                "vmf": base + cap_offset + 1.0,
                "power_spherical": base + cap_offset + 1.0,
                "vmf_pin30": base + cap_offset,
                "power_spherical_pin30": base + cap_offset,
            }
            for arm, gaps in arms.items():
                runs[(dim, sigma, arm)] = {
                    s: {"diverged": False, "gap_elbo": float(g)}
                    for s, g in enumerate(gaps)
                }
    return runs


def test_tuned_tests_confirm_and_margin() -> None:
    tests = kc.tuned_tests(_fake_runs(0.2), leave_one_out=False)
    assert len(tests) == 24 and all(t["arm"].endswith("pin30") for t in tests)
    assert all(t["confirmed"] for t in tests)
    assert all(abs(t["p"] - t["p_enum"]) < 1e-9 for t in tests)
    near = kc.tuned_tests(_fake_runs(0.01), leave_one_out=True)
    assert not any(t["confirmed"] for t in near)  # significant but below the margin
    assert all(t["holm_reject"] for t in near)
