"""Smoke test for apps.synthetic.sbm_recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from apps.synthetic import sbm_recovery


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))


def test_experiment_runs_for_all_four_families_and_writes_expected_keys(
    tmp_path: Path,
) -> None:
    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "3",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_smoke",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "out" / "sbm_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "power_spherical", "tnbbeta"}
    for family, metrics in results.items():
        assert "no_checkpoint" not in metrics
        assert math.isfinite(metrics["test_auc"])
        assert math.isfinite(metrics["test_ap"])
        assert "diverged" in metrics
        assert math.isfinite(metrics["entropy_mean"]) or math.isnan(
            metrics["entropy_mean"]
        )
        if family == "tnbbeta":
            assert {
                "p_mean",
                "q_mean",
                "epsilon_mean",
                "m_mean",
                "frac_bimodal",
            } <= set(metrics)
        else:
            assert "p_mean" not in metrics

    checkpoint_root = tmp_path / "ckpt"
    for family in ("gaussian", "vmf", "power_spherical", "tnbbeta"):
        assert (
            checkpoint_root / f"sbm_recovery_smoke_{family}_seed0" / "final.pt"
        ).exists()


def test_experiment_only_runs_the_requested_families(tmp_path: Path) -> None:
    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_smoke2",
            "--families",
            "gaussian",
            "tnbbeta",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "out" / "sbm_recovery.json").read_text())
    assert set(results) == {"gaussian", "tnbbeta"}
