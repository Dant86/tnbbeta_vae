"""Tests for apps.link_prediction.main's --ignore-features flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from apps.link_prediction import main as link_prediction_main
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.models import GraphVAEConfig
from tnbbeta_vae.paths import checkpoint_dir

_NUM_NODES = 20
_NUM_FEATURES = 5


def _fake_mag_graph(*_args: object, **_kwargs: object) -> SnapGraph:
    # Same density reasoning as tests/apps/test_link_prediction_mag_dataset.py: enough
    # edges that split_edges' floor()ed val/test counts are >= 1, but not a complete
    # graph (no non-edges left for negative sampling).
    upper = sp.random(_NUM_NODES, _NUM_NODES, density=0.3, rng=0, format="coo")
    adjacency = ((upper + upper.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    rng = np.random.default_rng(0)
    features = sp.csr_matrix(
        rng.integers(0, 5, size=(_NUM_NODES, _NUM_FEATURES)).astype(np.float32)
    )
    node_id_map = {node: node for node in range(_NUM_NODES)}
    return SnapGraph(adjacency, features, node_id_map)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(link_prediction_main, "load_mag_coauthor", _fake_mag_graph)


def _run(argv: list[str]) -> list[tuple[Any, GraphVAEConfig]]:
    """Runs main() with run_once spied on; returns each call's (graph, config)."""
    calls: list[tuple[Any, GraphVAEConfig]] = []
    original = link_prediction_main.run_once

    def spy(graph: Any, split: Any, config: GraphVAEConfig, **kwargs: Any) -> Any:
        calls.append((graph, config))
        return original(graph, split, config, **kwargs)

    link_prediction_main.run_once = spy  # type: ignore[assignment]
    try:
        link_prediction_main.main(argv)
    finally:
        link_prediction_main.run_once = original  # type: ignore[assignment]
    return calls


_BASE_ARGS = [
    "--dataset", "mag_cs", "--family", "tnbbeta",
    "--lrs", "0.01", "--dropouts", "0", "--latent-dims", "3",
    "--epochs", "1", "--seeds", "0", "--device", "cpu",
]  # fmt: skip


def test_default_behavior_is_unchanged_without_the_flag() -> None:
    calls = _run(_BASE_ARGS)

    assert len(calls) == 1
    graph, config = calls[0]
    assert graph.features.shape == (_NUM_NODES, _NUM_FEATURES)
    assert config.in_features == _NUM_FEATURES

    output = checkpoint_dir() / "link_prediction" / "mag_cs_tnbbeta.json"
    assert output.exists()
    assert not (
        checkpoint_dir() / "link_prediction" / "mag_cs_nofeat_tnbbeta.json"
    ).exists()


def test_ignore_features_builds_an_identity_feature_batch_of_the_right_shape() -> None:
    calls = _run([*_BASE_ARGS, "--ignore-features"])

    assert len(calls) == 1
    graph, config = calls[0]
    # The batch actually built has identity features of the right shape -- not just
    # trusted plumbing.
    features = graph.features
    dense = features.to_dense() if features.is_sparse else features
    assert dense.shape == (_NUM_NODES, _NUM_NODES)
    assert torch.allclose(torch.as_tensor(np.asarray(dense)), torch.eye(_NUM_NODES))

    # in_features tracks num_nodes, not the original feature count, with no
    # special-casing needed: it's read from graph.features.shape[1] after the
    # override.
    assert config.in_features == _NUM_NODES
    assert config.in_features != _NUM_FEATURES


def test_ignore_features_uses_the_nofeat_suffix_in_the_output_path() -> None:
    _run([*_BASE_ARGS, "--ignore-features"])

    output = checkpoint_dir() / "link_prediction" / "mag_cs_nofeat_tnbbeta.json"
    assert output.exists()
    # And never collides with the features-on counterpart's path.
    assert not (checkpoint_dir() / "link_prediction" / "mag_cs_tnbbeta.json").exists()


def test_ignore_features_off_and_on_runs_write_separate_files_without_collision() -> (
    None
):
    _run(_BASE_ARGS)
    _run([*_BASE_ARGS, "--ignore-features"])

    directory = checkpoint_dir() / "link_prediction"
    assert (directory / "mag_cs_tnbbeta.json").exists()
    assert (directory / "mag_cs_nofeat_tnbbeta.json").exists()
