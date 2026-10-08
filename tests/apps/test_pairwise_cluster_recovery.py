"""Smoke test for apps.synthetic.pairwise_cluster_recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path

from apps.synthetic import pairwise_cluster_recovery


def test_experiment_runs_for_all_four_families_and_writes_expected_keys(
    tmp_path: Path,
) -> None:
    pairwise_cluster_recovery.main(
        [
            "--out-dir",
            str(tmp_path),
            "--num-clusters",
            "3",
            "--points-per-cluster",
            "5",
            "--dim",
            "6",
            "--latent-dim",
            "4",
            "--hidden-dims",
            "8",
            "4",
            "--num-pairs",
            "40",
            "--test-fraction",
            "0.25",
            "--epochs",
            "3",
            "--seed",
            "0",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "pairwise_cluster_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "power_spherical", "tnbbeta"}
    for family, metrics in results.items():
        assert math.isfinite(metrics["test_auc"])
        assert math.isfinite(metrics["test_ap"])
        assert math.isfinite(metrics["entropy_mean"]) or math.isnan(
            metrics["entropy_mean"]
        )
        assert "diverged" in metrics
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


def test_experiment_only_runs_the_requested_families(tmp_path: Path) -> None:
    pairwise_cluster_recovery.main(
        [
            "--out-dir",
            str(tmp_path),
            "--num-clusters",
            "3",
            "--points-per-cluster",
            "5",
            "--dim",
            "6",
            "--latent-dim",
            "4",
            "--hidden-dims",
            "8",
            "4",
            "--num-pairs",
            "40",
            "--epochs",
            "2",
            "--families",
            "gaussian",
            "tnbbeta",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "pairwise_cluster_recovery.json").read_text())
    assert set(results) == {"gaussian", "tnbbeta"}
