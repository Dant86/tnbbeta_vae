"""Tests for the axial Bingham study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.synthetic import axial_bingham as study
from tnbbeta_vae.data.axial_bingham import AxialBinghamTask


def _train(tmp_path: Path, dim: int, decoder: str, arm: str, seed: int = 0) -> dict:
    study.main(
        [
            "train",
            "--dim",
            str(dim),
            "--sigma",
            "0.5",
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
            "60",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
        ]
    )
    path = study.run_path(tmp_path, dim, 0.5, decoder, arm, seed).with_suffix(".json")
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("dim", "decoder", "arm"),
    [
        (2, "fixed", "tnbbeta"),
        (3, "linear", "vmf_sym"),
        (3, "mlp", "power_spherical"),
        (4, "fixed", "ps_sym"),
    ],
)
def test_tiny_run_writes_finite_summary(
    tmp_path: Path, dim: int, decoder: str, arm: str
) -> None:
    record = _train(tmp_path, dim, decoder, arm)
    assert not record["diverged"]
    assert np.isfinite(record["gap_elbo"])
    # Unbiased ELBO below the true evidence on average.
    assert record["gap_elbo"] > -0.5
    stem = study.run_path(tmp_path, dim, 0.5, decoder, arm, 0)
    assert stem.with_suffix(".npz").exists()
    assert stem.with_suffix(".pt").exists()
    if dim in study.EXACT_DIMS:
        assert record["quad_true_model_maxerr"] < 1e-6
        assert record["kl_to_model_min"] > -0.5


def test_fixed_decoder_gap_equals_exact_kl_to_bingham(tmp_path: Path) -> None:
    _train(tmp_path, 2, "fixed", "vmf")
    arrays = np.load(
        study.run_path(tmp_path, 2, 0.5, "fixed", "vmf", 0).with_suffix(".npz")
    )
    gap = arrays["log_ref"] - arrays["elbo"]
    assert np.allclose(gap, arrays["log_model"] - arrays["elbo"], atol=1e-6)


def test_arm_families() -> None:
    assert study.arm_base_family("vmf_sym") == "vmf"
    assert study.arm_base_family("ps_sym") == "power_spherical"
    assert study.arm_base_family("tnbbeta_init0") == "tnbbeta"
    assert study.arm_p_init("tnbbeta_init0") == 0.0
    with pytest.raises(ValueError):
        study.arm_base_family("nope")


def test_wilcoxon_enumeration_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        diff = rng.normal(0.3, 1.0, size=10)
        assert (
            abs(
                study._wilcoxon_greater(diff) - study._wilcoxon_greater_enumerated(diff)
            )
            < 1e-9
        )


def test_holm_step_down() -> None:
    tests = [
        {"p": 0.001, "diff": np.array([0.1, 0.2])},
        {"p": 0.02, "diff": np.array([0.1, 0.2])},
        {"p": 0.06, "diff": np.array([0.01, 0.02])},
    ]
    study._holm(tests)
    assert [t["holm_reject"] for t in tests] == [True, True, False]
    assert [t["margin"] for t in tests] == [True, True, False]


def test_task_log_likelihood_matches_decoder() -> None:
    task = AxialBinghamTask(3, 0.5)
    x, z = task.sample(4, torch.Generator().manual_seed(0))
    model = study.AxialVAE(3, 0.5, "fixed", "vmf")
    assert torch.allclose(
        model.log_likelihood(x, z), task.log_likelihood(x, z), atol=1e-5
    )
