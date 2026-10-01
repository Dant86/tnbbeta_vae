"""Tests for apps.link_prediction.main's checkpoint-saving (``--run-name``)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from apps.link_prediction.main import run_once
from tnbbeta_vae.data.planetoid import Graph, LinkSplit, normalized_adjacency
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint

_NUM_NODES = 8


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))


def _tiny_graph() -> Graph:
    rng = np.random.default_rng(0)
    rows = np.arange(_NUM_NODES)
    cols = np.roll(rows, 1)
    adjacency = sp.coo_matrix(
        (np.ones(_NUM_NODES), (rows, cols)), shape=(_NUM_NODES, _NUM_NODES)
    )
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    features = rng.normal(size=(_NUM_NODES, 4)).astype(np.float32)
    return Graph(adjacency, sp.csr_matrix(features))


def _tiny_split(graph: Graph) -> LinkSplit:
    upper = sp.triu(graph.adjacency, k=1).tocoo()
    edges = np.stack([upper.row, upper.col])
    return LinkSplit(
        train_adjacency=graph.adjacency,
        val_positive=edges[:, :1],
        val_negative=np.array([[0], [3]]),
        test_positive=edges[:, 1:2],
        test_negative=np.array([[1], [4]]),
    )


def test_run_once_without_run_name_saves_no_checkpoint(tmp_path: Path) -> None:
    graph = _tiny_graph()
    split = _tiny_split(graph)
    config = GraphVAEConfig(family="tnbbeta", in_features=4, latent_dim=3)

    run_once(
        graph, split, config, lr=0.01, epochs=3, seed=0, device=torch.device("cpu")
    )

    assert not checkpoint_dir().exists()


def test_run_once_with_run_name_saves_a_loadable_checkpoint(tmp_path: Path) -> None:
    graph = _tiny_graph()
    split = _tiny_split(graph)
    config = GraphVAEConfig(family="tnbbeta", in_features=4, latent_dim=3)

    run_once(
        graph,
        split,
        config,
        lr=0.01,
        epochs=3,
        seed=0,
        device=torch.device("cpu"),
        run_name="dblp_tnbbeta",
    )

    run_dir = checkpoint_dir() / "dblp_tnbbeta_seed0"
    assert (run_dir / "final.pt").exists()

    loaded, checkpoint = load_model_checkpoint(run_dir / "final.pt", "cpu")
    model = cast("GraphVAE", loaded)
    assert checkpoint["model_name"] == "graph_vae"
    assert checkpoint["config"]["family"] == "tnbbeta"
    # The loaded model actually runs: build a batch the same way run_once does.
    features = torch.as_tensor(graph.features.toarray(), dtype=torch.float32)
    batch = GraphBatch(
        features=features,
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.zeros((2, 0), dtype=torch.long),
    )
    posterior, _ = model.posterior_and_prior(batch)
    assert posterior.rsample().shape == (_NUM_NODES, 3)


def test_run_once_multiple_seeds_each_get_their_own_checkpoint() -> None:
    graph = _tiny_graph()
    split = _tiny_split(graph)
    config = GraphVAEConfig(family="vmf", in_features=4, latent_dim=3)

    for seed in (0, 1):
        run_once(
            graph,
            split,
            config,
            lr=0.01,
            epochs=2,
            seed=seed,
            device=torch.device("cpu"),
            run_name="dblp_vmf",
        )

    assert (checkpoint_dir() / "dblp_vmf_seed0" / "final.pt").exists()
    assert (checkpoint_dir() / "dblp_vmf_seed1" / "final.pt").exists()
