"""A synthetic stochastic-block-model graph for isolating GCN + graph-structure effects.

This project's research has found ``TNBBetaSpherical``'s posterior goes genuinely
bimodal specifically on real ``GraphVAE`` link-prediction runs (Cora, Citeseer,
Pubmed, com-DBLP -- all of which combine a GCN encoder with a real graph), and never
on reconstruction-style VAE tasks (which have neither). Two prior ablations
(``apps/synthetic/pairwise_cluster_recovery.py``'s plain-MLP-encoder arm, and
``GraphVAE``'s own feature-reconstruction-decoder arm) each independently ruled out
one candidate explanation while leaving "the GCN architecture + a graph with
community-correlated structure, together" untested in isolation.

This module generates that missing data point: a small stochastic block model
(dense intra-community edges, sparse inter-community edges) with identity node
features -- the same featureless setup as com-DBLP, at ``pairwise_cluster_recovery``'s
exact item/community count (10 x 50 = 500 by default) for direct comparability --
so ``apps/synthetic/sbm_recovery.py`` can train the real, unmodified ``GraphVAE`` on
it and see whether the GCN + community structure alone is sufficient to reproduce
bimodality.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import torch

from tnbbeta_vae.data.snap_community import Graph

__all__ = ["sbm_community_labels", "stochastic_block_model"]


def stochastic_block_model(
    num_communities: int = 10,
    nodes_per_community: int = 50,
    p_in: float = 0.3,
    p_out: float = 0.01,
    seed: int = 0,
) -> Graph:
    """Generates a synthetic stochastic-block-model graph with identity features.

    Node ``i`` belongs to community ``i // nodes_per_community`` (see
    :func:`sbm_community_labels`). Every pair of distinct nodes is connected
    independently: with probability ``p_in`` if they share a community, ``p_out``
    otherwise. Features are a sparse identity matrix, built exactly as
    ``tnbbeta_vae.data.snap_community.load_snap_community`` builds them for
    featureless real graphs like com-DBLP (so a model trained on this graph sees the
    same kind of featureless input).

    Args:
        num_communities: Number of equal-sized communities.
        nodes_per_community: Number of nodes per community.
        p_in: Edge probability between two nodes in the same community.
        p_out: Edge probability between two nodes in different communities.
        seed: Seed for the edge sampling (reproducible given the same seed).

    Returns:
        A :class:`~tnbbeta_vae.data.snap_community.Graph` with
        ``num_communities * nodes_per_community`` nodes, a trivial identity
        ``node_id_map`` (``{i: i for i in range(num_nodes)}``, since there is no raw
        external ID space here), and sparse identity features as a
        ``torch.sparse_coo_tensor``.
    """
    num_nodes = num_communities * nodes_per_community
    labels = sbm_community_labels(num_communities, nodes_per_community)
    rng = np.random.default_rng(seed)

    rows, cols = np.triu_indices(num_nodes, k=1)
    same_community = labels[rows] == labels[cols]
    probabilities = np.where(same_community, p_in, p_out)
    connected = rng.random(probabilities.shape[0]) < probabilities
    edge_rows, edge_cols = rows[connected], cols[connected]

    # Mirror edges to both directions, matching load_snap_community's convention.
    all_rows = np.concatenate([edge_rows, edge_cols])
    all_cols = np.concatenate([edge_cols, edge_rows])
    data = np.ones(all_rows.shape[0], dtype=np.float32)
    adjacency: sp.csr_matrix = sp.csr_matrix(
        (data, (all_rows, all_cols)), shape=(num_nodes, num_nodes)
    )
    adjacency.eliminate_zeros()

    # Sparse identity features, built exactly as load_snap_community builds them for
    # featureless graphs (see that function's docstring for why: avoids
    # materializing a dense num_nodes x num_nodes matrix).
    indices = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    values = torch.ones(num_nodes, dtype=torch.float32)
    features = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce()

    node_id_map = {i: i for i in range(num_nodes)}
    return Graph(adjacency, features, node_id_map)


def sbm_community_labels(
    num_communities: int = 10, nodes_per_community: int = 50
) -> npt.NDArray[np.int64]:
    """Returns the ground-truth community index of each node.

    A pure function of the community sizes (not the edge sampling), for use only in
    a sanity check or report -- never fed to a model trained on the graph
    :func:`stochastic_block_model` returns.

    Args:
        num_communities: Number of equal-sized communities.
        nodes_per_community: Number of nodes per community.

    Returns:
        Array of shape ``(num_communities * nodes_per_community,)``: node ``i``'s
        community is ``i // nodes_per_community``.
    """
    return np.repeat(np.arange(num_communities), nodes_per_community)
