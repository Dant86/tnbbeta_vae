"""SNAP community-structured graphs (co-authorship, product categories, etc.).

Loads undirected graphs from SNAP's community detection archive, with node IDs remapped
to a contiguous 0..n-1 range and features as sparse identity matrices (suitable for
featureless graphs like co-DBLP). Provides community membership parsing and bridge-node
identification for targeted diagnostics.

The edge list format is two whitespace/tab-separated node IDs per line, with lines
starting with '#' treated as comments. Community file format is one community per line,
each line a whitespace/tab-separated list of node IDs in that community.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "SNAP_COMMUNITY_DATASETS",
    "Graph",
    "bridge_nodes",
    "load_snap_community",
    "load_snap_communities",
]

# Scipy's sparse types are loosely typed, so matrices are annotated as ``Any``.
SparseMatrix = Any

SNAP_COMMUNITY_DATASETS = ("dblp", "amazon")


@dataclass
class Graph:
    """An undirected graph with node features.

    Attributes:
        adjacency: Symmetric binary adjacency without self-loops, shape ``(n, n)``.
            Stored as scipy.sparse.csr_matrix.
        features: Node features, shape ``(n, n)`` or ``(n, in_features)``.
            Stored as scipy.sparse or torch.sparse. For SNAP graphs with identity
            features, use torch.sparse_coo_tensor.
    """

    adjacency: SparseMatrix
    features: Any  # scipy.sparse or torch.sparse tensor


def load_snap_community(root: Path, name: str) -> Graph:
    """Loads one SNAP community graph from ``root/com-<name>.*``.

    Node IDs in the raw SNAP files are not contiguous; they are remapped to a dense
    0..n-1 range consistently across adjacency and features.

    Args:
        root: Directory with raw files (see ``apps/data/download_snap_community.py``).
        name: Dataset name like ``"dblp"`` or ``"amazon"``.

    Returns:
        The graph with sparse identity features as torch.sparse_coo_tensor (no real node
        features in SNAP datasets). Adjacency is scipy.sparse for compatibility with
        split_edges/normalized_adjacency pipeline.
    """
    edge_file = root / f"com-{name}.ungraph.txt.gz"

    # Read edge list and remap node IDs to 0..n-1.
    edges = []
    node_id_map = {}
    next_id = 0

    with gzip.open(edge_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            # Remap node IDs to contiguous range.
            if a not in node_id_map:
                node_id_map[a] = next_id
                next_id += 1
            if b not in node_id_map:
                node_id_map[b] = next_id
                next_id += 1
            edges.append((node_id_map[a], node_id_map[b]))

    num_nodes = next_id
    if num_nodes == 0:
        raise ValueError(f"No edges found in {edge_file}")

    # Build sparse adjacency (symmetric, undirected graph).
    edges_array = np.array(edges, dtype=np.int64)
    rows = np.concatenate([edges_array[:, 0], edges_array[:, 1]])
    cols = np.concatenate([edges_array[:, 1], edges_array[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adjacency: Any = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency.eliminate_zeros()

    # Create sparse identity features as torch.sparse_coo_tensor.
    # This avoids materializing a dense num_nodes x num_nodes matrix, which would
    # OOM for large graphs like com-DBLP (317K nodes -> 402GB dense).
    indices = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    values = torch.ones(num_nodes, dtype=torch.float32)
    features = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()

    return Graph(adjacency, features)


def load_snap_communities(root: Path, name: str) -> dict[int, set[int]]:
    """Loads community membership from ``root/com-<name>.all.cmty.txt.gz``.

    Parses the community file into a mapping of node IDs (in the original SNAP ID space,
    before remapping) to the set of community indices they belong to. To use this with
    the remapped node IDs from load_snap_community, apply the remapping manually or
    pass the node_id_map from load_snap_community.

    Args:
        root: Directory holding the raw files.
        name: Dataset name (same as in load_snap_community).

    Returns:
        A dict mapping original (unmapped) node IDs to sets of community indices.
        A community index is just the order in which communities appear in the file.
    """
    community_file = root / f"com-{name}.all.cmty.txt.gz"

    memberships: dict[int, set[int]] = {}
    with gzip.open(community_file, "rt") as f:
        for community_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            node_ids = [int(x) for x in line.split()]
            for node_id in node_ids:
                if node_id not in memberships:
                    memberships[node_id] = set()
                memberships[node_id].add(community_idx)

    return memberships


def bridge_nodes(
    memberships: dict[int, set[int]], min_multiplicity: int = 2
) -> set[int]:
    """Returns node IDs that belong to at least min_multiplicity communities.

    These nodes are "bridge" nodes that connect multiple communities -- genuinely
    multi-community members (e.g., authors publishing in multiple venues).

    Args:
        memberships: Node ID -> set of community indices, as from load_snap_communities.
        min_multiplicity: Minimum number of communities a node must belong to.

    Returns:
        Set of node IDs that are members of at least min_multiplicity communities.
    """
    return {
        node_id
        for node_id, communities in memberships.items()
        if len(communities) >= min_multiplicity
    }
