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
    feature_noise_std: float | None = None,
) -> Graph:
    """Generates a synthetic stochastic-block-model graph with identity features.

    Node ``i`` belongs to community ``i // nodes_per_community`` (see
    :func:`sbm_community_labels`). Every pair of distinct nodes is connected
    independently: with probability ``p_in`` if they share a community, ``p_out``
    otherwise. By default (``feature_noise_std=None``), features are a sparse
    identity matrix, built exactly as
    ``tnbbeta_vae.data.snap_community.load_snap_community`` builds them for
    featureless real graphs like com-DBLP (so a model trained on this graph sees the
    same kind of featureless input).

    Args:
        num_communities: Number of equal-sized communities.
        nodes_per_community: Number of nodes per community.
        p_in: Edge probability between two nodes in the same community.
        p_out: Edge probability between two nodes in different communities.
        seed: Seed for the edge sampling and, if requested, the feature noise
            (reproducible given the same seed).
        feature_noise_std: If given, overrides the default identity features with a
            tunable interpolation between pure noise and an exact community
            revelation: each node's feature vector becomes its true community as a
            one-hot vector (shape ``(num_nodes, num_communities)``, from
            :func:`sbm_community_labels`) plus i.i.d. ``N(0,
            feature_noise_std ** 2)`` noise, drawn from the same seeded generator as
            the edge sampling (so the whole graph -- edges and features alike -- is
            reproducible from one seed). ``None`` (the default) preserves today's
            sparse-identity-feature behavior exactly; this parameter is additive-only
            and does not change the edges sampled either way.

    Returns:
        A :class:`~tnbbeta_vae.data.snap_community.Graph` with
        ``num_communities * nodes_per_community`` nodes and a trivial identity
        ``node_id_map`` (``{i: i for i in range(num_nodes)}``, since there is no raw
        external ID space here). ``features`` is a sparse identity
        ``torch.sparse_coo_tensor`` if ``feature_noise_std`` is ``None``, or a dense
        ``torch.Tensor`` of shape ``(num_nodes, num_communities)`` otherwise.
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

    features = _node_features(
        num_nodes, num_communities, labels, feature_noise_std, rng
    )

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


def _node_features(
    num_nodes: int,
    num_communities: int,
    labels: npt.NDArray[np.int64],
    feature_noise_std: float | None,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Builds :func:`stochastic_block_model`'s node features.

    ``feature_noise_std=None`` returns the default sparse identity matrix; otherwise
    returns a dense noisy one-hot-community matrix (see
    :func:`stochastic_block_model`'s docstring for the exact construction).
    """
    if feature_noise_std is None:
        indices = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        values = torch.ones(num_nodes, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices,
            values,
            (num_nodes, num_nodes),
            dtype=torch.float32,
            check_invariants=False,
        ).coalesce()

    one_hot = np.zeros((num_nodes, num_communities), dtype=np.float32)
    one_hot[np.arange(num_nodes), labels] = 1.0
    noise = rng.normal(0.0, feature_noise_std, size=one_hot.shape).astype(np.float32)
    return torch.as_tensor(one_hot + noise, dtype=torch.float32)
