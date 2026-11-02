"""Tests for GraphVAE, the ranking metrics and the link-prediction driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from apps.link_prediction import main as link_main
from apps.link_prediction import table as link_table
from tnbbeta_vae.data.planetoid import Graph, normalized_adjacency, split_edges
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.heads import head_size
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


def test_log_temperature_stays_a_top_level_state_dict_key() -> None:
    """Pins GraphVAE's state_dict shape so existing checkpoints keep loading.

    ``log_temperature`` is a top-level ``nn.Parameter`` baked into every
    real cluster checkpoint's ``state_dict`` (see CLAUDE.md). Extracting
    ``link_logits``'s formula into ``pairwise_logits`` must not move or
    rename it -- this guards against that regression specifically.
    """
    model = GraphVAE(GraphVAEConfig(family="vmf", in_features=6, latent_dim=4))

    assert "log_temperature" in model.state_dict()


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


@pytest.mark.parametrize("family", _FAMILIES)
def test_a_featureless_isolated_node_does_not_break_training(family: Any) -> None:
    """Citeseer has such nodes: the GCN gives them an exactly zero raw output."""
    torch.manual_seed(0)
    graph = _community_graph()
    size = graph.adjacency.shape[0]
    padded = sp.block_diag([graph.adjacency, sp.csr_matrix((1, 1))], format="csr")
    features = sp.vstack([graph.features, sp.csr_matrix((1, 6))], format="csr")
    batch, _ = _batch(Graph(padded, features))
    model = GraphVAE(GraphVAEConfig(family=family, in_features=6, latent_dim=4))

    out = model.training_step(batch)
    out["loss"].backward()
    embeddings = model.embeddings(batch)

    assert torch.isfinite(out["loss"])
    assert embeddings.shape == (size + 1, 4)
    assert torch.isfinite(embeddings).all()
    if family != "gaussian":
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(size + 1), atol=1e-4)


def test_a_diverging_run_stops_and_is_reported_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _community_graph()
    split = split_edges(graph.adjacency, seed=0)
    real_step = GraphVAE.training_step
    calls = {"count": 0}

    def unstable_step(self: GraphVAE, batch: GraphBatch, kl_weight: float = 1.0) -> Any:
        calls["count"] += 1
        out = real_step(self, batch, kl_weight)
        if calls["count"] > 5:
            out["loss"] = out["loss"] * float("nan")
        return out

    monkeypatch.setattr(GraphVAE, "training_step", unstable_step)

    result = link_main.run_once(
        graph,
        split,
        GraphVAEConfig(family="gaussian", in_features=6, latent_dim=4),
        lr=0.01,
        epochs=50,
        seed=0,
        device=torch.device("cpu"),
    )

    assert result["diverged"] == 1.0
    assert calls["count"] == 6  # stopped at the first non-finite step
    assert 0.0 <= result["best_epoch"] <= 4.0  # best of the 5 healthy epochs is kept
    assert 0.0 <= result["val_auc"] <= 1.0


def test_a_run_that_diverges_immediately_reports_chance_level_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _community_graph()
    split = split_edges(graph.adjacency, seed=0)
    real_step = GraphVAE.training_step

    def always_nan(self: GraphVAE, batch: GraphBatch, kl_weight: float = 1.0) -> Any:
        out = real_step(self, batch, kl_weight)
        out["loss"] = out["loss"] * float("nan")
        return out

    monkeypatch.setattr(GraphVAE, "training_step", always_nan)

    result = link_main.run_once(
        graph,
        split,
        GraphVAEConfig(family="vmf", in_features=6, latent_dim=4),
        lr=0.01,
        epochs=10,
        seed=0,
        device=torch.device("cpu"),
    )

    assert result["diverged"] == 1.0 and result["best_epoch"] == -1.0
    assert result["val_auc"] == result["test_auc"] == 0.5
    summary = link_main.summarize([result, {**result, "diverged": 0.0}])
    assert summary["num_diverged"] == 1.0


def _sparse_identity_batch(num_nodes: int, seed: int = 0) -> tuple[GraphBatch, Any]:
    """Creates a small synthetic graph with sparse identity features."""
    # Create a simple ring graph: node i connects to nodes (i-1) % n and (i+1) % n.
    rows = np.repeat(np.arange(num_nodes), 2)
    cols = np.concatenate(
        [
            (np.arange(num_nodes) - 1) % num_nodes,
            (np.arange(num_nodes) + 1) % num_nodes,
        ]
    )
    adjacency = sp.csr_matrix(
        (np.ones_like(rows, dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )
    adjacency.eliminate_zeros()
    # Sparse identity features (no real node features, just self-loops as placeholders).
    features = torch.sparse_coo_tensor(
        indices=torch.arange(num_nodes).unsqueeze(0).repeat(2, 1),
        values=torch.ones(num_nodes, dtype=torch.float32),
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()
    split = split_edges(adjacency, seed=seed)
    upper = np.stack(np.nonzero(np.triu(split.train_adjacency.toarray(), k=1)))
    batch = GraphBatch(
        features=features,
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(upper, dtype=torch.long),
    )
    return batch, split


def test_feature_reconstruction_weight_cli_flag_is_threaded_into_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(link_main, "load_planetoid", lambda *_: _community_graph())

    link_main.main(
        [
            "--dataset",
            "cora",
            "--family",
            "tnbbeta",
            "--lrs",
            "0.01",
            "--dropouts",
            "0",
            "--latent-dims",
            "4",
            "--epochs",
            "3",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--feature-reconstruction-weight",
            "0.5",
            "--run-name",
            "feature_recon_cli_test",
        ]  # fmt: skip
    )

    checkpoint = torch.load(
        tmp_path / "ckpt" / "feature_recon_cli_test_seed0" / "final.pt",
        weights_only=False,
    )
    assert checkpoint["config"]["feature_reconstruction_weight"] == 0.5


def test_default_config_state_dict_keys_are_unchanged() -> None:
    """Pins GraphVAE's state_dict key set so real cluster checkpoints keep loading.

    ``feature_reconstruction_weight`` defaults to ``0.0`` (no feature decoder), so a
    default-config model's parameters must be byte-for-byte the pre-change set --
    not just "no feature_decoder key", but nothing else either.
    """
    model = GraphVAE(GraphVAEConfig())

    assert set(model.state_dict().keys()) == {
        "first.weight",
        "log_temperature",
        "second.weight",
    }


def test_feature_reconstruction_weight_zero_never_builds_decoder() -> None:
    model = GraphVAE(
        GraphVAEConfig(
            family="tnbbeta",
            in_features=6,
            latent_dim=4,
            feature_reconstruction_weight=0.0,
        )
    )

    assert model.feature_decoder is None


def test_feature_reconstruction_weight_zero_training_step_keys_are_unchanged() -> None:
    batch, _ = _batch(_community_graph())
    model = GraphVAE(
        GraphVAEConfig(
            family="tnbbeta",
            in_features=6,
            latent_dim=4,
            feature_reconstruction_weight=0.0,
        )
    )

    out = model.training_step(batch)

    assert set(out.keys()) == {"loss", "link_loss", "kl"}


@pytest.mark.parametrize("family", _FAMILIES)
def test_feature_reconstruction_weight_positive_builds_decoder_and_trains(
    family: Any,
) -> None:
    torch.manual_seed(0)
    batch, _ = _batch(_community_graph())
    model = GraphVAE(
        GraphVAEConfig(
            family=family,
            in_features=6,
            latent_dim=4,
            feature_reconstruction_weight=1.0,
        )
    )

    assert isinstance(model.feature_decoder, torch.nn.Linear)

    out = model.training_step(batch)
    out["loss"].backward()

    assert "feature_loss" in out
    assert torch.isfinite(out["feature_loss"])
    assert torch.isfinite(out["loss"])
    # log_temperature is excluded: it never enters the Gaussian family's link logit
    # (see pairwise_logits), so it has no gradient regardless of this change.
    gradient_bearing = (
        *model.first.parameters(),
        *model.second.parameters(),
        *cast("torch.nn.Linear", model.feature_decoder).parameters(),
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in gradient_bearing
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_sparse_identity_features_train_without_densifying(family: Any) -> None:
    """Verifies sparse features work without densifying (no OOM on large graphs)."""
    torch.manual_seed(0)
    batch, _ = _sparse_identity_batch(num_nodes=100)
    model = GraphVAE(
        GraphVAEConfig(
            family=family, in_features=100, hidden_dim=16, latent_dim=4, dropout=0.0
        )
    )

    out = model.training_step(batch)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["link_loss"])
    assert torch.isfinite(out["kl"])
    assert all(p.grad is not None for p in model.first.parameters())
    assert all(p.grad is not None for p in model.second.parameters())
    embeddings = model.embeddings(batch)
    assert embeddings.shape == (100, 4)
    if family != "gaussian":
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(100), atol=1e-4)


def test_fixed_epsilon_shrinks_head_and_pins_epsilon() -> None:
    free = GraphVAE(GraphVAEConfig(family="tnbbeta", in_features=6, latent_dim=4))
    fixed = GraphVAE(
        GraphVAEConfig(family="tnbbeta", in_features=6, latent_dim=4, fixed_epsilon=2.5)
    )

    assert fixed.second.out_features == free.second.out_features - 1
    raw = torch.randn(5, fixed.second.out_features)
    posterior = fixed._posterior(raw)  # noqa: SLF001
    assert torch.allclose(posterior.epsilon, torch.full((5,), 2.5))  # type: ignore[attr-defined]


def test_fixed_epsilon_none_leaves_state_dict_shape_unchanged() -> None:
    config = GraphVAEConfig(family="tnbbeta", in_features=6, latent_dim=4)

    assert config.fixed_epsilon is None
    assert GraphVAE(config).second.out_features == head_size("tnbbeta", 4)


def test_results_name_keeps_a_single_run_from_clobbering_the_grid_search_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path))
    grid_search_json = tmp_path / "link_prediction" / "cora_tnbbeta.json"
    grid_search_json.parent.mkdir(parents=True)
    grid_search_json.write_text("original")
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(link_main, "load_planetoid", lambda *_: _community_graph())

    link_main.main(
        [
            "--dataset", "cora", "--family", "tnbbeta", "--lrs", "0.01",
            "--dropouts", "0", "--latent-dims", "4", "--epochs", "2",
            "--seeds", "0", "--device", "cpu", "--results-name", "one_cell",
        ]
    )  # fmt: skip

    assert grid_search_json.read_text() == "original"
    assert (tmp_path / "link_prediction" / "one_cell.json").exists()


@pytest.mark.parametrize("family", ["vmf", "power_spherical"])
def test_fixed_kappa_shrinks_head_and_pins_kappa(family: str) -> None:
    config = {"family": family, "in_features": 6, "latent_dim": 4}
    free = GraphVAE(GraphVAEConfig(**config))  # type: ignore[arg-type]
    fixed = GraphVAE(GraphVAEConfig(**config, fixed_kappa=9.0))  # type: ignore[arg-type]

    assert fixed.second.out_features == free.second.out_features - 1
    posterior = fixed._posterior(torch.randn(5, fixed.second.out_features))  # noqa: SLF001
    kappa = posterior.scale if family == "vmf" else posterior.kappa  # type: ignore[attr-defined]
    assert torch.allclose(kappa.reshape(-1), torch.full((5,), 9.0))
    assert posterior.rsample().shape == (5, 4)


@pytest.mark.parametrize("family", ["tnbbeta", "gaussian"])
def test_fixed_kappa_rejects_families_without_a_kappa(family: str) -> None:
    with pytest.raises(ValueError, match="fixed_kappa"):
        GraphVAE(
            GraphVAEConfig(family=family, in_features=6, latent_dim=4, fixed_kappa=3.0)  # type: ignore[arg-type]
        )


def test_fixed_kappa_out_of_range_is_rejected() -> None:
    model = GraphVAE(
        GraphVAEConfig(family="vmf", in_features=6, latent_dim=4, fixed_kappa=0.5)
    )
    with pytest.raises(ValueError, match="kappa"):
        model._posterior(torch.randn(2, model.second.out_features))  # noqa: SLF001


@pytest.mark.parametrize("family", ["gaussian", "vmf", "tnbbeta"])
def test_decoder_bias_adds_a_parameter_only_when_requested(family: str) -> None:
    default = GraphVAE(GraphVAEConfig(family=family, in_features=10, latent_dim=4))  # type: ignore[arg-type]
    biased = GraphVAE(
        GraphVAEConfig(family=family, in_features=10, latent_dim=4, decoder_bias=True)  # type: ignore[arg-type]
    )

    assert "bias" not in default.state_dict()
    assert set(biased.state_dict()) - set(default.state_dict()) == {"bias"}
    assert float(default.logit_bias()) == 0.0


def test_decoder_bias_shifts_every_link_logit() -> None:
    config = GraphVAEConfig(
        family="vmf", in_features=10, latent_dim=4, decoder_bias=True
    )
    model = GraphVAE(config)
    z = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
    edges = torch.tensor([[0, 1, 2], [3, 4, 5]])
    before = model.link_logits(z, edges)
    with torch.no_grad():
        assert model.bias is not None
        model.bias.fill_(-2.0)

    assert torch.allclose(model.link_logits(z, edges), before - 2.0)


@pytest.mark.parametrize("family", ["vmf", "power_spherical"])
def test_initial_kappa_starts_kappa_near_the_target_and_leaves_it_trainable(
    family: str,
) -> None:
    torch.manual_seed(0)
    config = GraphVAEConfig(
        family=family,  # type: ignore[arg-type]
        in_features=20,
        latent_dim=16,
        initial_kappa=16.0,
    )
    model = GraphVAE(config)
    batch = GraphBatch(
        features=torch.eye(20),
        norm_adjacency=torch.eye(20).to_sparse(),
        positive_edges=torch.tensor([[0, 1], [1, 2]]),
    )
    posterior, _ = model.posterior_and_prior(batch)

    kappa = posterior.scale if family == "vmf" else posterior.kappa  # type: ignore[attr-defined]
    assert 10.0 < float(kappa.mean()) < 22.0
    assert model.second.weight.requires_grad
    assert head_size(family, 16) == model.second.out_features  # type: ignore[arg-type]


def test_initial_kappa_rejects_other_families_and_fixed_kappa() -> None:
    with pytest.raises(ValueError, match="initial_kappa"):
        GraphVAE(GraphVAEConfig(family="tnbbeta", in_features=5, initial_kappa=4.0))
    with pytest.raises(ValueError, match="exclusive"):
        GraphVAE(
            GraphVAEConfig(
                family="vmf", in_features=5, initial_kappa=4.0, fixed_kappa=4.0
            )
        )


@pytest.mark.parametrize("family", ["gaussian", "vmf", "power_spherical", "tnbbeta"])
def test_head_bias_adds_second_bias_only_when_requested(family: str) -> None:
    kwargs: dict[str, Any] = {"family": family, "in_features": 10, "latent_dim": 4}
    default = GraphVAE(GraphVAEConfig(**kwargs))
    biased = GraphVAE(GraphVAEConfig(head_bias=True, **kwargs))

    assert set(default.state_dict()) == {
        "first.weight",
        "log_temperature",
        "second.weight",
    }
    assert set(biased.state_dict()) - set(default.state_dict()) == {"second.bias"}
    assert biased.second.bias.shape == (default.second.out_features,)  # type: ignore[union-attr]
    # An old (bias-free) state dict still loads strictly into a default model.
    GraphVAE(GraphVAEConfig(head_bias=False, **kwargs)).load_state_dict(
        default.state_dict(), strict=True
    )


def test_head_bias_is_added_after_aggregation_and_trains() -> None:
    graph = _community_graph()
    batch, _ = _batch(graph)
    config = GraphVAEConfig(
        family="gaussian", in_features=6, latent_dim=4, head_bias=True
    )
    model = GraphVAE(config)
    bias = model.second.bias
    assert bias is not None
    with torch.no_grad():
        bias.copy_(torch.arange(float(bias.numel())))
    posterior, _ = model.posterior_and_prior(batch)
    # Gaussian head: [mean (4), raw scale (4)]; the mean's offset is exactly bias[:4]
    # at every node, not scaled by the node's adjacency row sum.
    zero = GraphVAE(config)
    zero.load_state_dict(model.state_dict())
    with torch.no_grad():
        zero.second.bias.zero_()  # type: ignore[union-attr]
    shift = posterior.base_dist.loc - zero.posterior_and_prior(batch)[0].base_dist.loc  # type: ignore[attr-defined]
    assert torch.allclose(shift, torch.arange(4.0).expand_as(shift), atol=1e-5)
    model.training_step(batch)["loss"].backward()
    assert bias.grad is not None
    assert float(bias.grad.abs().sum()) > 0


@pytest.mark.parametrize(
    "extra",
    [{}, {"fixed_epsilon": 1.5}],
)
def test_p_init_offset_sets_only_the_raw_p_bias_entry(extra: dict[str, Any]) -> None:
    latent_dim = 5
    base = {"family": "tnbbeta", "in_features": 10, "latent_dim": latent_dim}
    torch.manual_seed(0)
    plain = GraphVAE(GraphVAEConfig(head_bias=True, **base, **extra))
    torch.manual_seed(0)
    shifted = GraphVAE(
        GraphVAEConfig(head_bias=True, p_init_offset=1.1, **base, **extra)
    )
    assert plain.second.bias is not None and shifted.second.bias is not None
    assert float(shifted.second.bias[latent_dim]) == pytest.approx(1.1)
    changed = (plain.second.bias != shifted.second.bias).nonzero().flatten().tolist()
    assert changed == [latent_dim]
    assert torch.equal(plain.second.weight, shifted.second.weight)


def test_p_init_offset_is_ignored_without_head_bias_or_for_other_families() -> None:
    torch.manual_seed(0)
    a = GraphVAE(GraphVAEConfig(in_features=10, latent_dim=4))
    torch.manual_seed(0)
    b = GraphVAE(GraphVAEConfig(in_features=10, latent_dim=4, p_init_offset=1.1))
    assert set(a.state_dict()) == set(b.state_dict())
    assert torch.equal(a.second.weight, b.second.weight)
    torch.manual_seed(0)
    c = GraphVAE(
        GraphVAEConfig(family="vmf", in_features=10, latent_dim=4, head_bias=True)
    )
    torch.manual_seed(0)
    d = GraphVAE(
        GraphVAEConfig(
            family="vmf",
            in_features=10,
            latent_dim=4,
            head_bias=True,
            p_init_offset=1.1,
        )
    )
    assert torch.equal(c.second.bias, d.second.bias)  # type: ignore[arg-type]
