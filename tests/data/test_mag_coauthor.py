"""Tests for MAG co-authorship network loading (NOCD paper's SparseGraph .npz files).

The synthetic ``.npz`` fixtures here mirror the exact key structure verified against
the real ``mag_cs.npz`` (downloaded once, inspected, then deleted -- see
``src/tnbbeta_vae/data/mag_coauthor.py``'s module docstring): ``adj_matrix.*``,
``attr_matrix.*`` and ``labels.*`` scipy-CSR triples (data/indices/indptr/shape), plus
string lookup tables that this loader doesn't need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from tnbbeta_vae.data.mag_coauthor import (
    MAG_COAUTHOR_DATASETS,
    load_mag_coauthor,
    load_mag_communities,
)
from tnbbeta_vae.data.snap_community import Graph as SnapGraph


def _write_mag_npz(
    path: Path,
    *,
    adjacency: Any,
    features: Any,
    labels: Any,
) -> None:
    """Writes a tiny ``.npz`` file in the real format's exact key structure."""
    num_nodes = adjacency.shape[0]
    np.savez(
        path,
        **{
            "adj_matrix.data": adjacency.data,
            "adj_matrix.indices": adjacency.indices,
            "adj_matrix.indptr": adjacency.indptr,
            "adj_matrix.shape": np.array(adjacency.shape, dtype=np.int64),
            "attr_matrix.data": features.data,
            "attr_matrix.indices": features.indices,
            "attr_matrix.indptr": features.indptr,
            "attr_matrix.shape": np.array(features.shape, dtype=np.int64),
            "labels.data": labels.data,
            "labels.indices": labels.indices,
            "labels.indptr": labels.indptr,
            "labels.shape": np.array(labels.shape, dtype=np.int64),
            "node_names": np.array([f"n{i}" for i in range(num_nodes)]),
            "attr_names": np.array([f"kw{i}" for i in range(features.shape[1])]),
            "edge_attr_matrix": None,
            "edge_attr_names": None,
            "class_names": np.array(
                [f"class{i}" for i in range(labels.shape[1])], dtype=object
            ),
            "metadata": None,
            "type": "SparseGraph",
        },
        allow_pickle=True,
    )


def _tiny_mag_file(tmp_path: Path, name: str = "cs") -> Path:
    """A 6-node ring graph, 4 keyword features, 3 overlapping communities."""
    rows = np.arange(6)
    cols = np.roll(rows, 1)
    adjacency = sp.coo_matrix((np.ones(6), (rows, cols)), shape=(6, 6))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    rng = np.random.default_rng(0)
    features = sp.csr_matrix(rng.integers(0, 5, size=(6, 4)).astype(np.float32))

    # Overlapping memberships: node 0 is in two communities (a bridge node), node 5
    # is in none -- both are real possibilities the loader must handle.
    labels = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    path = tmp_path / f"mag_{name}.npz"
    _write_mag_npz(path, adjacency=adjacency, features=features, labels=labels)
    return path


def test_mag_coauthor_datasets_lists_the_four_mag_subjects() -> None:
    assert MAG_COAUTHOR_DATASETS == ("cs", "eng", "chem", "med")


def test_load_mag_coauthor_reuses_the_snap_community_graph_dataclass(
    tmp_path: Path,
) -> None:
    _tiny_mag_file(tmp_path)

    graph = load_mag_coauthor(tmp_path, "cs")

    assert isinstance(graph, SnapGraph)
    assert graph.adjacency.shape == (6, 6)
    assert graph.features.shape == (6, 4)
    # Adjacency is undirected (the ring's edges appear twice) and self-loop-free.
    assert graph.adjacency.sum() == 12
    assert graph.adjacency.diagonal().sum() == 0


def test_load_mag_coauthor_node_ids_are_already_dense_0_to_n_minus_1(
    tmp_path: Path,
) -> None:
    """Unlike SNAP's com-DBLP/com-amazon, MAG's row order needs no remapping."""
    _tiny_mag_file(tmp_path)

    graph = load_mag_coauthor(tmp_path, "cs")

    assert graph.node_id_map == {i: i for i in range(6)}


def test_load_mag_coauthor_features_are_sparse_not_binary_identity(
    tmp_path: Path,
) -> None:
    """Real MAG features are bag-of-keyword counts, unlike SNAP's identity stand-in."""
    _tiny_mag_file(tmp_path)

    graph = load_mag_coauthor(tmp_path, "cs")

    dense = graph.features.toarray()
    assert not np.allclose(dense, np.eye(6, 4))
    assert dense.shape == (6, 4)


def test_load_mag_communities_returns_node_id_to_community_set(
    tmp_path: Path,
) -> None:
    _tiny_mag_file(tmp_path)

    memberships = load_mag_communities(tmp_path, "cs")

    assert memberships[0] == {0, 1}  # Bridge node: two communities.
    assert memberships[1] == {0}
    assert memberships[2] == {1}
    assert memberships[3] == {1}
    assert memberships[4] == {2}
    assert 5 not in memberships  # No communities at all.


def test_load_mag_communities_keys_match_load_mag_coauthors_node_space(
    tmp_path: Path,
) -> None:
    """The membership dict's keys are already graph.node_id_map's internal IDs."""
    _tiny_mag_file(tmp_path)

    graph = load_mag_coauthor(tmp_path, "cs")
    memberships = load_mag_communities(tmp_path, "cs")

    assert set(memberships) <= set(graph.node_id_map.values())


def test_load_mag_coauthor_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mag_coauthor(tmp_path, "cs")
