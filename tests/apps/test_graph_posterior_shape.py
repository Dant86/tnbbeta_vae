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
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.data.stochastic_block_model import stochastic_block_model
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


def _fake_mag_graph(*_args: object, **_kwargs: object) -> SnapGraph:
    num_nodes, in_features = 6, 4
    adjacency = sp.csr_matrix(
        ([1.0] * 4, ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(num_nodes, num_nodes)
    )
    features = sp.random(num_nodes, in_features, density=0.5, format="csr").astype(
        "float32"
    )
    node_id_map = {node: node for node in range(num_nodes)}
    return SnapGraph(adjacency, features, node_id_map)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(graph_posterior_shape, "load_planetoid", _fake_planetoid_graph)
    monkeypatch.setattr(graph_posterior_shape, "load_mag_coauthor", _fake_mag_graph)


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


def test_writes_posterior_shape_summary_for_a_mag_coauthor_dataset(
    tmp_path: Path,
) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=4, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "mag_cs_tnb_smoke", config)

    graph_posterior_shape.main(
        [
            "--run-name",
            "mag_cs_tnb_smoke",
            "--dataset",
            "mag_cs",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = tmp_path / "ckpt" / "mag_cs_tnb_smoke" / "posterior_shape_final.json"
    results = json.loads(output.read_text())
    assert results["num_nodes"] == 6
    assert results["dataset"] == "mag_cs"


def test_graph_to_batch_handles_a_synthetic_graph_with_sparse_identity_features() -> (
    None
):
    """``graph_to_batch`` plugs a non-dataset-dispatch ``Graph`` in directly.

    Exercises the reuse path ``apps/synthetic/sbm_recovery.py`` depends on: a
    ``Graph``-shaped object with torch.sparse identity features that never goes
    through ``_load_batch``'s dataset-name dispatch at all.
    """
    graph = stochastic_block_model(
        num_communities=3, nodes_per_community=4, p_in=0.5, p_out=0.05, seed=0
    )

    batch = graph_posterior_shape.graph_to_batch(graph, torch.device("cpu"))

    assert batch.num_nodes == 12
    assert batch.features.shape == (12, 12)
    assert batch.positive_edges.shape == (2, 0)
