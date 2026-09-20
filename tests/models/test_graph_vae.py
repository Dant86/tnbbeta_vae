"""Tests for GraphVAE, the ranking metrics and the link-prediction driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from apps.link_prediction import main as link_main
from apps.link_prediction import table as link_table
from tnbbeta_vae.data.planetoid import Graph, normalized_adjacency, split_edges
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.registry import list_registered_models

_FAMILIES = ["gaussian", "vmf", "tnbbeta"]


def _community_graph(seed: int = 0) -> Graph:
    """Six dense communities with informative features: links are easy to predict."""
    rng = np.random.default_rng(seed)
    communities, size = 6, 20
    community = np.repeat(np.arange(communities), size)
    same = community[:, None] == community[None, :]
    probability = np.where(same, 0.4, 0.005)
    upper = np.triu(rng.random((len(community), len(community))) < probability, k=1)
    adjacency = sp.csr_matrix((upper | upper.T).astype(np.float32))
    features = np.eye(communities, dtype=np.float32)[community]
    features += 0.1 * rng.standard_normal(features.shape).astype(np.float32)
    return Graph(adjacency, sp.csr_matrix(features))


def _batch(graph: Graph, seed: int = 0) -> tuple[GraphBatch, Any]:
    split = split_edges(graph.adjacency, seed=seed)
    upper = np.stack(np.nonzero(np.triu(split.train_adjacency.toarray(), k=1)))
    batch = GraphBatch(
        torch.as_tensor(graph.features.toarray(), dtype=torch.float32),
        normalized_adjacency(split.train_adjacency),
        torch.as_tensor(upper, dtype=torch.long),
    )
    return batch, split


def test_roc_auc_and_average_precision_match_hand_computed_values() -> None:
    positive = np.array([0.9, 0.8, 0.4])
    negative = np.array([0.7, 0.3, 0.2])

    assert roc_auc(positive, negative) == pytest.approx(8 / 9)
    assert roc_auc(positive, positive) == pytest.approx(0.5)
    assert average_precision(positive, negative) == pytest.approx((1 + 1 + 3 / 4) / 3)
    assert roc_auc(np.array([1.0]), np.array([0.0])) == 1.0


def test_registered_and_batch_moves_devices() -> None:
    assert "graph_vae" in list_registered_models()
    batch, _ = _batch(_community_graph())

    moved = batch.to(torch.device("cpu"))

    assert moved.num_nodes == 120


@pytest.mark.parametrize("family", _FAMILIES)
def test_training_step_gradients_and_embeddings(family: Any) -> None:
    torch.manual_seed(0)
    batch, _ = _batch(_community_graph())
    model = GraphVAE(GraphVAEConfig(family=family, in_features=6, latent_dim=4))

    out = model.training_step(batch)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert all(p.grad is not None for p in model.first.parameters())
    assert all(p.grad is not None for p in model.second.parameters())
    embeddings = model.embeddings(batch)
    assert embeddings.shape == (120, 4)
    if family != "gaussian":
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(120), atol=1e-4)


def test_temperature_is_learned_by_default_and_fixed_when_given() -> None:
    learned = GraphVAE(GraphVAEConfig(family="vmf", in_features=6, latent_dim=4))
    fixed = GraphVAE(
        GraphVAEConfig(family="vmf", in_features=6, latent_dim=4, fixed_temperature=1.0)
    )

    assert learned.log_temperature.requires_grad
    assert learned.temperature().item() == pytest.approx(5.0)
    assert not fixed.log_temperature.requires_grad
    assert fixed.temperature().item() == 1.0
    z = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    edges = torch.tensor([[0, 1], [2, 3]])
    inner = (z[edges[0]] * z[edges[1]]).sum(-1)
    assert torch.allclose(fixed.link_logits(z, edges), inner)
    assert torch.allclose(learned.link_logits(z, edges), 5.0 * inner, atol=1e-5)


@pytest.mark.parametrize("family", _FAMILIES)
def test_models_learn_to_predict_links_on_a_community_graph(family: Any) -> None:
    graph = _community_graph()
    split = split_edges(graph.adjacency, seed=0)

    result = link_main.run_once(
        graph,
        split,
        GraphVAEConfig(family=family, in_features=6, latent_dim=8),
        lr=0.01,
        epochs=150,
        seed=0,
        device=torch.device("cpu"),
    )

    assert result["val_auc"] > 0.8
    assert result["test_auc"] > 0.8
    assert 0.0 <= result["test_ap"] <= 1.0


def test_driver_grid_search_selects_by_validation_auc_and_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(link_main, "load_planetoid", lambda *_: _community_graph())

    for family in _FAMILIES:
        link_main.main(
            [
                "--dataset",
                "cora",
                "--family",
                family,
                "--lrs",
                "0.01",
                "--dropouts",
                "0",
                "0.2",
                "--latent-dims",
                "4",
                "--epochs",
                "20",
                "--seeds",
                "0",
                "1",
                "--device",
                "cpu",
            ]  # fmt: skip
        )

    result = json.loads(
        (tmp_path / "ckpt" / "link_prediction" / "cora_tnbbeta.json").read_text()
    )
    assert len(result["configurations"]) == 2
    best = max(result["configurations"], key=lambda item: item["val_auc"])
    assert result["selected"] == best
    assert result["selected"]["test_auc_std"] >= 0

    capsys.readouterr()
    link_table.main(["--datasets", "cora", "pubmed"])
    rows = capsys.readouterr().out.strip().splitlines()
    assert "TNBBeta-VGAE" in rows[0]
    assert rows[2].startswith("| cora | AUC | ")
    assert "±" in rows[2] and "| - | - | - |" in rows[4]
