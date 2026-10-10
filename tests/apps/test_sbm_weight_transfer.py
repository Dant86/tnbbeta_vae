"""Tests for apps.eval.sbm_weight_transfer (tiny synthetic checkpoint/graphs only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from apps.eval import sbm_weight_transfer
from tnbbeta_vae.models import GraphVAE, GraphVAEConfig


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))


def _write_checkpoint(tmp_path: Path, run_name: str, config: GraphVAEConfig) -> None:
    model = GraphVAE(config)
    run_dir = tmp_path / "ckpt" / run_name
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "model_name": "graph_vae",
            "config": config.model_dump(),
            "model_state_dict": model.state_dict(),
        },
        run_dir / "final.pt",
    )


def test_writes_posterior_shape_summary_for_tnbbeta(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=30, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_transfer_smoke", config)

    sbm_weight_transfer.main(
        [
            "--run-name",
            "sbm_transfer_smoke",
            "--device",
            "cpu",
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--graph-seed",
            "0",
        ]  # fmt: skip
    )

    output = (
        tmp_path
        / "ckpt"
        / "sbm_transfer_smoke"
        / "weight_transfer_final_pout0.05_gseed0.json"
    )
    results = json.loads(output.read_text())
    assert results["num_nodes"] == 30
    assert results["model_name"] == "graph_vae"
    assert results["graph_p_out"] == 0.05
    assert results["graph_p_in"] == 0.6
    assert results["graph_seed"] == 0
    stats = results["posterior_stats"]
    assert {"p_mean", "q_mean", "epsilon_mean", "m_mean", "frac_bimodal"} <= set(stats)


def test_omits_tnbbeta_fields_for_vmf(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "vmf"), in_features=30, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_transfer_vmf_smoke", config)

    sbm_weight_transfer.main(
        [
            "--run-name",
            "sbm_transfer_vmf_smoke",
            "--device",
            "cpu",
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--p-out",
            "0.05",
        ]  # fmt: skip
    )

    output = (
        tmp_path
        / "ckpt"
        / "sbm_transfer_vmf_smoke"
        / "weight_transfer_final_pout0.05_gseed0.json"
    )
    stats = json.loads(output.read_text())["posterior_stats"]
    assert "p_mean" not in stats
    assert "entropy_mean" in stats


def test_different_graph_specs_write_distinct_output_files(tmp_path: Path) -> None:
    """Running the SAME checkpoint against two different graph specs (the actual
    transfer-test use case) must not clobber one output with the other."""
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=30, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_transfer_two_graphs", config)

    for p_out in (0.005, 0.05):
        sbm_weight_transfer.main(
            [
                "--run-name",
                "sbm_transfer_two_graphs",
                "--device",
                "cpu",
                "--num-communities",
                "3",
                "--nodes-per-community",
                "10",
                "--p-out",
                str(p_out),
            ]  # fmt: skip
        )

    run_dir = tmp_path / "ckpt" / "sbm_transfer_two_graphs"
    low = json.loads(
        (run_dir / "weight_transfer_final_pout0.005_gseed0.json").read_text()
    )
    high = json.loads(
        (run_dir / "weight_transfer_final_pout0.05_gseed0.json").read_text()
    )
    assert low["graph_p_out"] == 0.005
    assert high["graph_p_out"] == 0.05


def test_builds_the_batch_from_the_given_graph_spec_not_a_hardcoded_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirms the script actually builds its batch from the ``--p-out``/etc.
    arguments given on the command line (the "other condition's graph" the transfer
    test needs), not some fixed/training graph baked into the script by accident."""
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=30, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_transfer_spy", config)

    seen_kwargs: list[dict[str, object]] = []
    real_sbm = sbm_weight_transfer.stochastic_block_model

    def _spy(**kwargs: object) -> object:
        seen_kwargs.append(kwargs)
        return real_sbm(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sbm_weight_transfer, "stochastic_block_model", _spy)

    sbm_weight_transfer.main(
        [
            "--run-name",
            "sbm_transfer_spy",
            "--device",
            "cpu",
            "--num-communities",
            "3",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.4",
            "--p-out",
            "0.2",
            "--graph-seed",
            "5",
        ]  # fmt: skip
    )

    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["num_communities"] == 3
    assert seen_kwargs[0]["nodes_per_community"] == 10
    assert seen_kwargs[0]["p_in"] == 0.4
    assert seen_kwargs[0]["p_out"] == 0.2
    assert seen_kwargs[0]["seed"] == 5
