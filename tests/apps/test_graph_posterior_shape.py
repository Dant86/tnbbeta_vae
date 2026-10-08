"""Tests for apps.eval.graph_posterior_shape (tiny fake graph, no network/real data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import scipy.sparse as sp
import torch

from apps.eval import graph_posterior_shape
from tnbbeta_vae.data.planetoid import Graph as PlanetoidGraph
from tnbbeta_vae.models import GraphVAE, GraphVAEConfig


def _fake_planetoid_graph(*_args: object, **_kwargs: object) -> PlanetoidGraph:
    num_nodes, in_features = 6, 4
    adjacency = sp.csr_matrix(
        ([1.0] * 4, ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(num_nodes, num_nodes)
    )
    features = sp.random(num_nodes, in_features, density=0.5, format="csr").astype(
        "float32"
    )
    return PlanetoidGraph(adjacency, features)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(graph_posterior_shape, "load_planetoid", _fake_planetoid_graph)


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
        family=cast("Any", "tnbbeta"), in_features=4, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "planetoid_tnb_smoke", config)

    graph_posterior_shape.main(
        [
            "--run-name",
            "planetoid_tnb_smoke",
            "--dataset",
            "cora",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = tmp_path / "ckpt" / "planetoid_tnb_smoke" / "posterior_shape_final.json"
    results = json.loads(output.read_text())
    assert results["num_nodes"] == 6
    assert results["model_name"] == "graph_vae"
    stats = results["posterior_stats"]
    assert {"p_mean", "q_mean", "epsilon_mean", "m_mean", "frac_bimodal"} <= set(stats)


def test_omits_tnbbeta_fields_for_vmf(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "vmf"), in_features=4, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "planetoid_vmf_smoke", config)

    graph_posterior_shape.main(
        [
            "--run-name",
            "planetoid_vmf_smoke",
            "--dataset",
            "cora",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = tmp_path / "ckpt" / "planetoid_vmf_smoke" / "posterior_shape_final.json"
    stats = json.loads(output.read_text())["posterior_stats"]
    assert "p_mean" not in stats
    assert "entropy_mean" in stats
