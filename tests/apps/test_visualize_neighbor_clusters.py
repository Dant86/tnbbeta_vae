"""Smoke test for apps.eval.visualize_neighbor_clusters (tiny synthetic graph only).

Illustrative, not a statistical test (see the module's own docstring) -- this only
confirms the script runs end to end and produces a figure with the right structure,
not that any particular clustering pattern appears.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from apps.eval import visualize_neighbor_clusters as vnc
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


def test_writes_a_figure_and_json_summary_for_tnbbeta(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=50, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_vis_smoke", config)

    vnc.main(
        [
            "--run-name",
            "sbm_vis_smoke",
            "--device",
            "cpu",
            "--num-communities",
            "5",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--graph-seed",
            "0",
            "--num-high",
            "2",
            "--num-low",
            "2",
            "--min-degree",
            "3",
        ]  # fmt: skip
    )

    run_dir = tmp_path / "ckpt" / "sbm_vis_smoke"
    html = run_dir / "neighbor_clusters_final.html"
    assert html.exists()
    assert html.stat().st_size > 0

    output = run_dir / "neighbor_clusters_final.json"
    results = json.loads(output.read_text())
    assert len(results["high_heterogeneity_nodes"]) == 2
    assert len(results["low_heterogeneity_nodes"]) == 2
    # Selected "high" nodes should have heterogeneity >= every selected "low" node's.
    min_high = min(n["heterogeneity"] for n in results["high_heterogeneity_nodes"])
    max_low = max(n["heterogeneity"] for n in results["low_heterogeneity_nodes"])
    assert min_high >= max_low


def test_works_for_a_non_tnbbeta_family_too(tmp_path: Path) -> None:
    """posterior_centre is defined for every family -- this diagnostic isn't
    TNBBeta-specific, unlike svae_latitude/graph_posterior_shape's TNBBeta-only
    fields."""
    config = GraphVAEConfig(
        family=cast("Any", "vmf"), in_features=50, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_vis_vmf_smoke", config)

    vnc.main(
        [
            "--run-name",
            "sbm_vis_vmf_smoke",
            "--device",
            "cpu",
            "--num-communities",
            "5",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--graph-seed",
            "0",
            "--num-high",
            "2",
            "--num-low",
            "2",
            "--min-degree",
            "3",
        ]  # fmt: skip
    )

    html = tmp_path / "ckpt" / "sbm_vis_vmf_smoke" / "neighbor_clusters_final.html"
    assert html.exists()


def test_supports_feature_noise_std_matching_a_feature_sweep_checkpoint(
    tmp_path: Path,
) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=5, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_vis_noisy_smoke", config)

    vnc.main(
        [
            "--run-name",
            "sbm_vis_noisy_smoke",
            "--device",
            "cpu",
            "--num-communities",
            "5",
            "--nodes-per-community",
            "10",
            "--p-in",
            "0.6",
            "--p-out",
            "0.05",
            "--graph-seed",
            "0",
            "--feature-noise-std",
            "0.5",
            "--num-high",
            "2",
            "--num-low",
            "2",
            "--min-degree",
            "3",
        ]  # fmt: skip
    )

    html = tmp_path / "ckpt" / "sbm_vis_noisy_smoke" / "neighbor_clusters_final.html"
    assert html.exists()


def test_raises_a_clear_error_when_min_degree_excludes_every_node(
    tmp_path: Path,
) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=50, hidden_dim=8, latent_dim=4
    )
    _write_checkpoint(tmp_path, "sbm_vis_toohigh_smoke", config)

    with pytest.raises(ValueError, match="min-degree"):
        vnc.main(
            [
                "--run-name",
                "sbm_vis_toohigh_smoke",
                "--device",
                "cpu",
                "--num-communities",
                "5",
                "--nodes-per-community",
                "10",
                "--p-in",
                "0.6",
                "--p-out",
                "0.05",
                "--graph-seed",
                "0",
                "--min-degree",
                "10000",
            ]  # fmt: skip
        )
