"""Tests for tnbbeta_vae.data.planetoid (fabricated raw Planetoid files)."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tnbbeta_vae.data.planetoid import (
    _LegacyUnpickler,
    load_planetoid,
    normalized_adjacency,
    split_edges,
)


def _write_raw(root: Path, name: str, *, missing_test_node: bool = False) -> None:
    """A 10-node ring: nodes 0-5 are train (x/allx), 6-9 test, 3 features."""
    num_features = 3
    rng = np.random.default_rng(0)
    features = rng.integers(0, 2, size=(10, num_features)).astype(np.float32)
    test_order = [8, 6, 9, 7]  # deliberately unsorted, as in the real files
    if missing_test_node:
        test_order = [8, 6, 9]
    graph = {node: [(node - 1) % 10, (node + 1) % 10] for node in range(10)}
    parts = {
        "x": sp.csr_matrix(features[:3]),
        "allx": sp.csr_matrix(features[:6]),
        "tx": sp.csr_matrix(features[test_order]),
        "graph": graph,
    }
    for suffix, value in parts.items():
        with (root / f"ind.{name}.{suffix}").open("wb") as file:
            pickle.dump(value, file)
    (root / f"ind.{name}.test.index").write_text("\n".join(map(str, test_order)))


def test_load_planetoid_restores_test_order_and_builds_a_symmetric_graph(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path, "cora")
    rng = np.random.default_rng(0)
    expected = rng.integers(0, 2, size=(10, 3)).astype(np.float32)

    graph = load_planetoid(tmp_path, "cora")

    assert graph.adjacency.shape == (10, 10)
    assert (graph.adjacency != graph.adjacency.T).nnz == 0
    assert graph.adjacency.diagonal().sum() == 0
    assert graph.adjacency.sum() == 20
    np.testing.assert_array_equal(graph.features.toarray(), expected)


def test_load_planetoid_pads_isolated_citeseer_test_nodes(tmp_path: Path) -> None:
    _write_raw(tmp_path, "citeseer", missing_test_node=True)
    rng = np.random.default_rng(0)
    expected = rng.integers(0, 2, size=(10, 3)).astype(np.float32)

    graph = load_planetoid(tmp_path, "citeseer")

    assert graph.features.shape == (10, 3)  # 6 train nodes + the test range 6..9
    dense = graph.features.toarray()
    for node in (0, 1, 2, 3, 4, 5, 6, 8, 9):
        np.testing.assert_array_equal(dense[node], expected[node])
    np.testing.assert_array_equal(dense[7], np.zeros(3))  # the isolated, absent node


def test_legacy_unpickler_remaps_removed_scipy_module_paths() -> None:
    import io

    unpickler = _LegacyUnpickler(io.BytesIO(b""))

    assert unpickler.find_class("scipy.sparse.csr", "csr_matrix") is sp.csr_matrix


def _ring(n: int) -> sp.csr_matrix:
    rows = list(range(n)) + [(i + 1) % n for i in range(n)]
    cols = [(i + 1) % n for i in range(n)] + list(range(n))
    return sp.csr_matrix((np.ones(2 * n, dtype=np.float32), (rows, cols)), shape=(n, n))


def test_split_edges_partitions_edges_and_negatives_are_true_non_edges() -> None:
    adjacency = _ring(100)

    split = split_edges(adjacency, val_fraction=0.1, test_fraction=0.2, seed=1)

    assert split.val_positive.shape[1] == 10 and split.test_positive.shape[1] == 20
    assert split.val_negative.shape == split.val_positive.shape
    assert split.test_negative.shape == split.test_positive.shape
    train_edges = split.train_adjacency.sum() / 2
    assert train_edges == 100 - 10 - 20
    held_out = {tuple(edge) for edge in split.val_positive.T} | {
        tuple(edge) for edge in split.test_positive.T
    }
    assert not any(split.train_adjacency[a, b] for a, b in held_out)
    for a, b in np.concatenate([split.val_negative, split.test_negative], axis=1).T:
        assert adjacency[a, b] == 0 and a != b
    negatives = [tuple(pair) for pair in split.val_negative.T] + [
        tuple(pair) for pair in split.test_negative.T
    ]
    assert len(set(negatives)) == len(negatives)
    same = split_edges(adjacency, val_fraction=0.1, test_fraction=0.2, seed=1)
    assert np.array_equal(same.val_positive, split.val_positive)


def test_normalized_adjacency_is_symmetric_with_unit_row_norm_scaling() -> None:
    dense = normalized_adjacency(_ring(6)).to_dense()

    assert torch.allclose(dense, dense.T)
    assert torch.allclose(dense.sum(1), torch.ones(6))  # regular graph: rows sum to 1
    assert dense[0, 0] == pytest.approx(1 / 3)
