"""Dense attributed graphs of the dense real-graph study.

Three graphs of mean degree above 24, loaded without unpickling anything:

* Amazon Computers and Amazon Photo (Shchur et al., "Pitfalls of Graph Neural Network
  Evaluation", MIT licence of ``github.com/shchur/gnn-benchmark``): a flat ``.npz`` of
  plain arrays (``adj_data/indices/indptr/shape``, ``attr_data/indices/indptr/shape``,
  ``labels``) that opens with ``numpy.load(..., allow_pickle=False)``. The stored
  adjacency is not symmetric; it is symmetrised here.
* Wikipedia Squirrel (MUSAE, Rozemberczki, Allen and Sarkar, 2019; data hosted with
  the ``benedekrozemberczki/MUSAE`` repository, GPL-3.0 for the code, and on SNAP):
  ``squirrel_edges.csv`` (two columns of node indices) and ``squirrel.json`` (node id
  to a list of feature indices).

Preprocessing is identical for every graph: symmetrise, binarise, drop self-loops,
deduplicate, keep all nodes (no largest-component restriction, isolated nodes stay).
Features are the binary indicator of the stored feature entries.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from tnbbeta_vae.data.planetoid import Graph

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DENSE_GRAPH_NAMES", "load_dense_graph"]

DENSE_GRAPH_NAMES = ("amazon_computers", "amazon_photo", "squirrel")
_AMAZON_FILES = {
    "amazon_computers": "amazon_electronics_computers.npz",
    "amazon_photo": "amazon_electronics_photo.npz",
}


def load_dense_graph(root: Path, name: str) -> Graph:
    """Loads one dense graph from ``root/<name>/``.

    Args:
        root: Directory holding one sub-directory per graph with the raw files.
        name: One of :data:`DENSE_GRAPH_NAMES`.

    Returns:
        The graph: symmetric binary adjacency without self-loops (CSR, float32) and
        binary sparse features (CSR, float32).

    Raises:
        ValueError: If ``name`` is unknown.
        FileNotFoundError: If a raw file is missing.
    """
    if name in _AMAZON_FILES:
        adjacency, features = _load_amazon(root / name / _AMAZON_FILES[name])
    elif name == "squirrel":
        adjacency, features = _load_musae(root / name, name)
    else:
        raise ValueError(f"Unknown dense graph {name!r}; expected {DENSE_GRAPH_NAMES}.")
    return Graph(_clean_adjacency(adjacency), _binary(features))


def _load_amazon(path: Path) -> tuple[Any, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    adjacency = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = sp.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    return adjacency, features


def _load_musae(directory: Path, name: str) -> tuple[Any, Any]:
    edges_path = directory / f"{name}_edges.csv"
    features_path = directory / f"{name}.json"
    for path in (edges_path, features_path):
        if not path.exists():
            raise FileNotFoundError(path)
    with edges_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header: id1,id2
        edges = np.array([[int(a), int(b)] for a, b in reader], dtype=np.int64)
    stored = json.loads(features_path.read_text())
    num_nodes = max(int(edges.max()), max(int(key) for key in stored)) + 1
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    rows = [int(key) for key, items in stored.items() for _ in items]
    cols = [int(item) for items in stored.values() for item in items]
    features = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(num_nodes, max(cols) + 1),
    ).tocsr()
    return adjacency, features


def _clean_adjacency(adjacency: Any) -> Any:
    """Symmetrises, binarises, removes self-loops and deduplicates."""
    matrix = sp.csr_matrix(adjacency)
    matrix = matrix + matrix.T
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.data = np.ones_like(matrix.data)
    return matrix.astype(np.float32)


def _binary(features: Any) -> Any:
    matrix = sp.csr_matrix(features)
    matrix.eliminate_zeros()
    matrix.data = np.ones_like(matrix.data)
    return matrix.astype(np.float32)
