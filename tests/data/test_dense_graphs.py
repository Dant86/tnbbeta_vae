"""Tests for the dense-graph loaders on tiny fabricated files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sp

from tnbbeta_vae.data.dense_graphs import load_dense_graph

if TYPE_CHECKING:
    from pathlib import Path


def test_amazon_loader_symmetrises_and_cleans(tmp_path: Path) -> None:
    # Directed edges 0->1, 1->2, 2->2 (self-loop), 0->1 stored once; node 3 isolated.
    adjacency = sp.csr_matrix(
        ([2.0, 1.0, 1.0], ([0, 1, 2], [1, 2, 2])), shape=(4, 4), dtype=np.float32
    )
    features = sp.csr_matrix(
        ([3.0, 1.0], ([0, 2], [1, 0])), shape=(4, 2), dtype=np.float32
    )
    folder = tmp_path / "amazon_photo"
    folder.mkdir()
    np.savez(
        folder / "amazon_electronics_photo.npz",
        adj_data=adjacency.data,
        adj_indices=adjacency.indices,
        adj_indptr=adjacency.indptr,
        adj_shape=np.array(adjacency.shape),
        attr_data=features.data,
        attr_indices=features.indices,
        attr_indptr=features.indptr,
        attr_shape=np.array(features.shape),
    )
    graph = load_dense_graph(tmp_path, "amazon_photo")
    dense = graph.adjacency.toarray()
    np.testing.assert_array_equal(dense, dense.T)
    assert dense.diagonal().sum() == 0
    assert set(np.unique(dense)) == {0.0, 1.0}
    assert dense.sum() == 4  # edges {0,1} and {1,2}, each in both directions
    assert graph.adjacency.shape == (4, 4)
    assert set(np.unique(graph.features.data)) == {1.0}


def test_musae_loader_dedupes_and_drops_self_loops(tmp_path: Path) -> None:
    folder = tmp_path / "squirrel"
    folder.mkdir()
    (folder / "squirrel_edges.csv").write_text("id1,id2\n0,1\n1,0\n1,2\n2,2\n")
    (folder / "squirrel.json").write_text(
        json.dumps({"0": [0, 2], "1": [1], "2": [], "3": [2]})
    )
    graph = load_dense_graph(tmp_path, "squirrel")
    assert graph.adjacency.shape == (4, 4)
    assert graph.adjacency.sum() == 4  # {0,1} and {1,2}, both directions
    assert graph.adjacency.diagonal().sum() == 0
    assert graph.features.shape == (4, 3)
    assert graph.features.nnz == 4


def test_unknown_graph_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dense graph"):
        load_dense_graph(tmp_path, "nope")
