"""Citation-network graphs (Cora, Citeseer, Pubmed) for link prediction.

Reads the raw Planetoid files (Yang et al. 2016) that Kipf and Welling's VGAE used, and
builds the edge split of that paper: 5% of the edges for validation and 10% for testing,
each with an equal number of sampled non-edges, and a training graph without them.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PLANETOID_DATASETS",
    "PLANETOID_FILE_SUFFIXES",
    "PLANETOID_URL",
    "Graph",
    "LinkSplit",
    "load_planetoid",
    "normalized_adjacency",
    "split_edges",
]

# scipy's sparse types are loosely typed, so matrices are annotated as ``Any``.
SparseMatrix = Any

PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_FILE_SUFFIXES = ("x", "tx", "allx", "graph", "test.index")
PLANETOID_DATASETS = ("cora", "citeseer", "pubmed")


@dataclass
class Graph:
    """An undirected graph with node features.

    Attributes:
        adjacency: Symmetric binary adjacency without self-loops, shape ``(n, n)``.
        features: Node features, shape ``(n, f)``.
    """

    adjacency: SparseMatrix
    features: SparseMatrix


@dataclass
class LinkSplit:
    """Train/validation/test edges for link prediction.

    Attributes:
        train_adjacency: Symmetric adjacency of the training edges.
        val_positive: Held-out edges, shape ``(2, n_val)``.
        val_negative: Sampled non-edges, shape ``(2, n_val)``.
        test_positive: Held-out edges, shape ``(2, n_test)``.
        test_negative: Sampled non-edges, shape ``(2, n_test)``.
    """

    train_adjacency: SparseMatrix
    val_positive: np.ndarray
    val_negative: np.ndarray
    test_positive: np.ndarray
    test_negative: np.ndarray


class _LegacyUnpickler(pickle.Unpickler):
    """Loads the old Planetoid pickles, whose scipy classes live at removed paths."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("scipy.sparse."):
            module = "scipy.sparse"
        return super().find_class(module, name)


def load_planetoid(root: Path, name: str) -> Graph:
    """Loads one Planetoid dataset from ``root/ind.<name>.*``.

    Args:
        root: Directory holding the raw files (see ``apps/data/download_planetoid.py``).
        name: ``"cora"``, ``"citeseer"`` or ``"pubmed"``.

    Returns:
        The graph, with test nodes put back in their original order.
    """
    objects = {}
    for suffix in ("x", "tx", "allx", "graph"):
        with (root / f"ind.{name}.{suffix}").open("rb") as file:
            objects[suffix] = _LegacyUnpickler(file, encoding="latin1").load()
    test_index = np.loadtxt(root / f"ind.{name}.test.index", dtype=int)
    test_sorted = np.sort(test_index)

    test_features = objects["tx"]
    if name == "citeseer":
        # Some test nodes are isolated and absent from the files; add empty rows so
        # that the test block spans the whole index range.
        extended = sp.lil_matrix(
            (test_sorted.max() - test_sorted.min() + 1, objects["x"].shape[1])
        )
        extended[test_sorted - test_sorted.min(), :] = test_features
        test_features = extended
    features: Any = sp.vstack((objects["allx"], test_features)).tolil()
    features[test_index, :] = features[test_sorted, :]

    rows, cols = [], []
    for node, neighbours in objects["graph"].items():
        rows += [node] * len(neighbours)
        cols += list(neighbours)
    size = features.shape[0]
    raw: Any = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    adjacency: Any = ((raw + raw.T) > 0).astype(np.float32).tolil()
    adjacency.setdiag(0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    all_features: Any = features
    return Graph(adjacency, all_features.tocsr().astype(np.float32))


def split_edges(
    adjacency: SparseMatrix,
    *,
    val_fraction: float = 0.05,
    test_fraction: float = 0.10,
    seed: int = 0,
) -> LinkSplit:
    """Holds out edges and samples equally many non-edges for each of val and test.

    Args:
        adjacency: Symmetric binary adjacency without self-loops.
        val_fraction: Fraction of edges held out for validation.
        test_fraction: Fraction of edges held out for testing.
        seed: Seed for the split.

    Returns:
        A :class:`LinkSplit`. The negatives are node pairs that are not edges of the
        full graph, drawn without duplicates and disjoint between val and test.
    """
    rng = np.random.default_rng(seed)
    upper: Any = sp.triu(adjacency, k=1).tocoo()
    edges = np.stack([upper.row, upper.col])
    order = rng.permutation(edges.shape[1])
    num_val = int(np.floor(edges.shape[1] * val_fraction))
    num_test = int(np.floor(edges.shape[1] * test_fraction))
    val_positive = edges[:, order[:num_val]]
    test_positive = edges[:, order[num_val : num_val + num_test]]
    train = edges[:, order[num_val + num_test :]]

    num_nodes = adjacency.shape[0]
    forbidden = {(int(a), int(b)) for a, b in edges.T}
    negatives: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(negatives) < num_val + num_test:
        a, b = sorted(rng.integers(num_nodes, size=2).tolist())
        if a != b and (a, b) not in forbidden and (a, b) not in seen:
            seen.add((a, b))
            negatives.append((a, b))
    negative = np.asarray(negatives).T

    one_direction: Any = sp.csr_matrix(
        (np.ones(train.shape[1], dtype=np.float32), (train[0], train[1])),
        shape=adjacency.shape,
    )
    train_adjacency: Any = one_direction + one_direction.T
    return LinkSplit(
        train_adjacency=train_adjacency.tocsr(),
        val_positive=val_positive,
        val_negative=negative[:, :num_val],
        test_positive=test_positive,
        test_negative=negative[:, num_val:],
    )


def normalized_adjacency(adjacency: SparseMatrix) -> torch.Tensor:
    """Returns ``D^-1/2 (A + I) D^-1/2`` as a sparse COO torch tensor."""
    with_loops: Any = adjacency + sp.eye(adjacency.shape[0])
    degree = np.asarray(with_loops.sum(axis=1)).flatten()
    inverse_root: Any = sp.diags(np.power(degree, -0.5))
    normalized: Any = (inverse_root @ with_loops @ inverse_root).tocoo()
    indices = torch.as_tensor(
        np.stack([normalized.row, normalized.col]), dtype=torch.long
    )
    values = torch.as_tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, values, normalized.shape, check_invariants=False
    ).coalesce()
