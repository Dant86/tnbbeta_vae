"""Smoke test for apps.synthetic.sbm_recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

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


def test_graph_and_split_use_graph_seed_not_the_per_run_training_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SBM graph/split are generated once from ``--graph-seed``, independent of
    ``--seed`` (which only seeds model init/training, like ``run_once`` already
    does) -- so a seed sweep for the TNBBeta-stability check trains on the exact
    same graph and edge split every time, varying only initialization/negative
    sampling, not the task itself.
    """
    seen_seeds: list[int] = []
    real_sbm = sbm_recovery.stochastic_block_model

    def _spy(**kwargs: object) -> object:
        seen_seeds.append(kwargs["seed"])  # type: ignore[arg-type]
        return real_sbm(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sbm_recovery, "stochastic_block_model", _spy)

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
            "--seed",
            "7",
            "--graph-seed",
            "123",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_smoke3",
            "--families",
            "gaussian",
        ]  # fmt: skip
    )

    assert seen_seeds == [123]


def test_feature_noise_std_omitted_passes_none_to_stochastic_block_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Additive-only regression: not passing the new flag must leave the graph
    builder's ``feature_noise_std`` at its ``None`` default, unchanged."""
    seen_feature_noise_std: list[object] = []
    real_sbm = sbm_recovery.stochastic_block_model

    def _spy(**kwargs: object) -> object:
        seen_feature_noise_std.append(kwargs.get("feature_noise_std"))
        return real_sbm(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sbm_recovery, "stochastic_block_model", _spy)

    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_nofeaturenoise_spy",
            "--families",
            "gaussian",
        ]  # fmt: skip
    )

    assert seen_feature_noise_std == [None]


def test_feature_noise_std_flag_is_passed_through_and_trains_on_dense_features(
    tmp_path: Path,
) -> None:
    """``--feature-noise-std`` actually reaches ``stochastic_block_model`` (dense,
    noisy one-hot-community features, not the default sparse identity) and the whole
    pipeline still runs end to end on top of it."""
    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "3",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_featurenoise_smoke",
            "--families",
            "gaussian",
            "tnbbeta",
            "--feature-noise-std",
            "0.5",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "out" / "sbm_recovery.json").read_text())
    assert set(results) == {"gaussian", "tnbbeta"}
    for metrics in results.values():
        assert "no_checkpoint" not in metrics
        assert math.isfinite(metrics["test_auc"])


def test_no_aggregation_runs_for_all_four_families_and_writes_expected_keys(
    tmp_path: Path,
) -> None:
    """``--no-aggregation`` is the "aggregation OFF" arm: an end-to-end smoke run,
    same pattern as the default (aggregation-ON) smoke test above.
    """
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
            "sbm_recovery_noagg_smoke",
            "--no-aggregation",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "out" / "sbm_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "power_spherical", "tnbbeta"}
    for family, metrics in results.items():
        assert "no_checkpoint" not in metrics
        assert math.isfinite(metrics["test_auc"])
        assert math.isfinite(metrics["test_ap"])
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


def test_no_aggregation_passes_an_identity_encoder_adjacency_to_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirms the flag is actually wired to ``run_once``'s ``encoder_adjacency``
    (an identity matrix of the right size) rather than just accepted and ignored,
    and that omitting the flag still passes ``None`` (the no-op default).
    """
    seen_encoder_adjacency: list[object] = []
    real_run_once = sbm_recovery.run_once

    def _spy(*args: object, **kwargs: object) -> object:
        seen_encoder_adjacency.append(kwargs["encoder_adjacency"])
        return real_run_once(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sbm_recovery, "run_once", _spy)

    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_noagg_spy",
            "--no-aggregation",
            "--families",
            "gaussian",
        ]  # fmt: skip
    )

    assert len(seen_encoder_adjacency) == 1
    identity = seen_encoder_adjacency[0]
    assert identity is not None
    dense = torch.as_tensor(identity.toarray())  # type: ignore[union-attr]
    assert torch.equal(dense, torch.eye(30))

    seen_encoder_adjacency.clear()
    sbm_recovery.main(
        [
            "--out-dir",
            str(tmp_path / "out2"),
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "sbm_recovery_agg_spy",
            "--families",
            "gaussian",
        ]  # fmt: skip
    )

    assert seen_encoder_adjacency == [None]
