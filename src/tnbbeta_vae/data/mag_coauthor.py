"""MAG co-authorship networks (NOCD paper's SparseGraph ``.npz`` releases).

Shchur and Gunnemann's "Overlapping Community Detection with Graph Neural
Networks" (NOCD) released four Microsoft Academic Graph co-authorship networks
(Computer Science, Engineering, Chemistry, Medicine): each a single large graph with
real sparse bag-of-keywords node features AND genuine overlapping ground-truth
community memberships -- unlike com-DBLP (:mod:`tnbbeta_vae.data.snap_community`,
overlapping communities but no real features) or Planetoid
(:mod:`tnbbeta_vae.data.planetoid`, real features but no overlapping communities).

Source: the paper's own repository (archived, but still served from GitHub's
raw-content host), ``https://github.com/shchur/overlapping-community-detection``,
data files directly under its ``data/`` folder as ``mag_<name>.npz``.

The real ``.npz`` key structure (verified directly against a downloaded
``mag_cs.npz``, then deleted -- not assumed from the file's lineage) is the same
"SparseGraph" pickled-sparse-matrix convention this research group used for its
earlier "Pitfalls of Graph Neural Network Evaluation" release:

* ``adj_matrix.data`` / ``.indices`` / ``.indptr`` / ``.shape``: a scipy CSR
  adjacency matrix (symmetric, binary {0, 1} data, no self-loops). On ``mag_cs.npz``:
  21,957 nodes, 96,750 edges.
* ``attr_matrix.data`` / ``.indices`` / ``.indptr`` / ``.shape``: a scipy CSR node
  feature matrix of bag-of-keyword counts (not binary -- values like 1.0-10.0+ on
  ``mag_cs.npz``), 7,793 keyword columns there.
* ``labels.data`` / ``.indices`` / ``.indptr`` / ``.shape``: a scipy CSR binary
  community-membership matrix, shape ``(num_nodes, num_communities)``. On
  ``mag_cs.npz``: 18 communities; every node belongs to at least one, up to 13 at
  once -- genuinely overlapping, unlike Planetoid's single-label classes.
* ``node_names``, ``attr_names``, ``class_names``: string lookup tables (each row's
  original opaque MAG author ID, the keyword vocabulary, and the community names).
  Not needed by this loader; kept in the file for completeness/debugging only.
* ``edge_attr_matrix``, ``edge_attr_names``, ``metadata``: always ``None`` in these
  four releases.
* ``type``: always the literal string ``"SparseGraph"``.

Row/column order in ``adj_matrix``/``attr_matrix``/``labels`` is already the graph's
internal ``0..n-1`` node space (verified: ``adj_matrix.indptr`` has length
``n + 1`` matching ``node_names``' length exactly, and every row index used by
``labels`` is in range) -- unlike SNAP's com-DBLP/com-amazon, whose raw author IDs
need :func:`tnbbeta_vae.data.snap_community.load_snap_community`'s remapping, MAG's
``node_names[i]`` is simply node ``i``'s original (opaque) MAG author ID string, never
read elsewhere in this codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from tnbbeta_vae.data.snap_community import Graph

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["MAG_COAUTHOR_DATASETS", "load_mag_coauthor", "load_mag_communities"]

MAG_COAUTHOR_DATASETS = ("cs", "eng", "chem", "med")


def load_mag_coauthor(root: Path, name: str) -> Graph:
    """Loads one MAG co-authorship network from ``root/mag_<name>.npz``.

    Args:
        root: Directory holding the raw ``.npz`` file (see
            ``apps/data/download_mag_coauthor.py``).
        name: One of :data:`MAG_COAUTHOR_DATASETS` (``"cs"``, ``"eng"``, ``"chem"``
            or ``"med"``).

    Returns:
        The graph, reusing :class:`~tnbbeta_vae.data.snap_community.Graph` since MAG's
        adjacency/features shapes fit it without modification: sparse CSR adjacency,
        sparse CSR real (non-identity) features, and an identity ``node_id_map``
        (MAG's row order is already the dense ``0..n-1`` range this codebase's other
        graph loaders have to build by remapping raw IDs).
    """
    data = _load_npz(root, name)
    adjacency = _csr_from_npz(data, "adj_matrix").astype(np.float32)
    features = _csr_from_npz(data, "attr_matrix").astype(np.float32)
    node_id_map = {node: node for node in range(adjacency.shape[0])}
    return Graph(adjacency, features, node_id_map)


def load_mag_communities(root: Path, name: str) -> dict[int, set[int]]:
    """Loads overlapping community membership for one MAG co-authorship network.

    Args:
        root: Directory holding the raw ``.npz`` file (same as
            :func:`load_mag_coauthor`).
        name: One of :data:`MAG_COAUTHOR_DATASETS`.

    Returns:
        Internal node ID (i.e. the same space as :func:`load_mag_coauthor`'s
        ``Graph.node_id_map`` values -- a no-op identity here, so no
        :func:`~tnbbeta_vae.data.snap_community.remap_communities` step is needed) ->
        the set of community indices that node belongs to, in the same shape
        :func:`~tnbbeta_vae.data.snap_community.load_snap_communities` returns. Nodes
        in zero communities (possible in general, though none occur on ``mag_cs.npz``)
        are omitted, exactly as a SNAP node never mentioned in any community line
        would be.
    """
    data = _load_npz(root, name)
    labels = _csr_from_npz(data, "labels")
    memberships: dict[int, set[int]] = {}
    for node in range(labels.shape[0]):
        start, end = labels.indptr[node], labels.indptr[node + 1]
        communities = {int(index) for index in labels.indices[start:end]}
        if communities:
            memberships[node] = communities
    return memberships


def _load_npz(root: Path, name: str) -> Any:
    """Opens ``root/mag_<name>.npz``, raising ``FileNotFoundError`` if it's missing."""
    path = root / f"mag_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _csr_from_npz(data: Any, prefix: str) -> Any:
    """Rebuilds one scipy CSR matrix from its ``<prefix>.{data,indices,indptr,shape}``.

    This is how every sparse matrix in this ``.npz`` format is stored.
    """
    return sp.csr_matrix(
        (data[f"{prefix}.data"], data[f"{prefix}.indices"], data[f"{prefix}.indptr"]),
        shape=tuple(data[f"{prefix}.shape"]),
    )
