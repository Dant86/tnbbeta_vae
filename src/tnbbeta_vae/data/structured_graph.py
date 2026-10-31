"""Synthetic graphs without a block-constant affinity, with known edge probabilities.

Complements :mod:`tnbbeta_vae.data.block_graph`, whose graphs are all block models.
Every generator returns the same :class:`~tnbbeta_vae.data.block_graph.BlockGraph`
(``block`` is a coarse node partition used only to break statistics down, ``kind``
names each pair's type), so ``apps.synthetic.graph_families`` scores them unchanged.

* :func:`latent_sphere_graph`: nodes on a sphere, ``P = sigmoid(scale * cos + offset)``,
  the generative model a dot-product decoder on a sphere assumes.
* :func:`mixed_membership_graph`: overlapping communities (each node a Dirichlet
  mixture).
* :func:`scale_free_graph`: a Chung-Lu model with Pareto weights; heavy-tailed degrees
  and no communities.
* :func:`ring_graph`: a small-world ring, dense lattice neighbours plus sparse
  shortcuts; no communities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from tnbbeta_vae.data.block_graph import BlockGraph

__all__ = [
    "latent_sphere_graph",
    "mixed_membership_graph",
    "ring_graph",
    "scale_free_graph",
]


def latent_sphere_graph(
    num_nodes: int = 500,
    intrinsic_dim: int = 4,
    scale: float = 10.0,
    offset: float = -9.0,
    *,
    seed: int = 0,
) -> BlockGraph:
    """Nodes uniform on a sphere with ``P_ij = sigmoid(scale * cos + offset)``.

    Args:
        num_nodes: Number of nodes.
        intrinsic_dim: Ambient dimension of the latent sphere.
        scale: Slope on the cosine similarity.
        offset: Logit offset (sets the overall density).
        seed: Seed for the positions and the edges.

    Returns:
        The graph; ``block`` is the sign pattern of the first two coordinates (4 blocks)
        and ``kind`` is ``near`` (cos > 0.5), ``mid`` (0 to 0.5) or ``far`` (< 0).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((num_nodes, intrinsic_dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    cosine = z @ z.T
    probability = 1.0 / (1.0 + np.exp(-(scale * cosine + offset)))
    kind = np.where(cosine > 0.5, "near", np.where(cosine > 0.0, "mid", "far"))
    block = (z[:, 0] > 0).astype(np.int64) * 2 + (z[:, 1] > 0)
    return _sample(probability, block, kind, rng)


def mixed_membership_graph(
    num_nodes: int = 500,
    num_blocks: int = 5,
    concentration: float = 0.3,
    p_in: float = 0.15,
    p_out: float = 0.01,
    *,
    seed: int = 0,
) -> BlockGraph:
    """Overlapping communities: ``P_ij = pi_i' B pi_j`` with Dirichlet memberships.

    Args:
        num_nodes: Number of nodes.
        num_blocks: Number of communities.
        concentration: Dirichlet concentration (smaller is closer to hard membership).
        p_in: Edge probability between two nodes wholly in the same community.
        p_out: Edge probability between nodes in different communities.
        seed: Seed for the memberships and the edges.

    Returns:
        The graph; ``block`` is each node's dominant community and ``kind`` is
        ``pure-same`` / ``pure-diff`` (both nodes have a dominant weight above 0.6) or
        ``mixed`` (at least one node does not).
    """
    rng = np.random.default_rng(seed)
    membership = rng.dirichlet(np.full(num_blocks, concentration), size=num_nodes)
    affinity = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(affinity, p_in)
    probability = membership @ affinity @ membership.T
    block = membership.argmax(axis=1)
    pure = membership.max(axis=1) > 0.6
    both_pure = np.outer(pure, pure)
    same = block[:, None] == block[None, :]
    kind = np.where(both_pure, np.where(same, "pure-same", "pure-diff"), "mixed")
    return _sample(probability, block, kind, rng)


def scale_free_graph(
    num_nodes: int = 500,
    tail_index: float = 2.2,
    density: float = 0.04,
    *,
    seed: int = 0,
) -> BlockGraph:
    """Chung-Lu graph: ``P_ij = clip(density * w_i * w_j)`` with Pareto weights.

    Args:
        num_nodes: Number of nodes.
        tail_index: Pareto tail index of the weights (smaller is heavier-tailed).
        density: Scales the pair probabilities (mean weight is 1).
        seed: Seed for the weights and the edges.

    Returns:
        The graph; ``block`` is the node's weight quartile and ``kind`` is
        ``hub-hub`` / ``hub-leaf`` / ``leaf-leaf`` (a hub is in the top 10% of weights).
    """
    rng = np.random.default_rng(seed)
    weights = rng.pareto(tail_index, size=num_nodes) + 1.0
    weights /= weights.mean()
    probability = np.clip(density * np.outer(weights, weights), 0.0, 1.0)
    hub = weights >= np.quantile(weights, 0.9)
    kind = np.where(
        np.outer(hub, hub),
        "hub-hub",
        np.where(hub[:, None] | hub[None, :], "hub-leaf", "leaf-leaf"),
    )
    quartile = np.searchsorted(np.quantile(weights, [0.25, 0.5, 0.75]), weights)
    return _sample(probability, quartile.astype(np.int64), kind, rng)


def ring_graph(
    num_nodes: int = 500,
    half_width: int = 10,
    p_near: float = 0.6,
    p_shortcut: float = 0.01,
    num_arcs: int = 5,
    *,
    seed: int = 0,
) -> BlockGraph:
    """Small-world ring: dense links within ``half_width`` steps, sparse shortcuts.

    Args:
        num_nodes: Number of nodes, placed in order around a cycle.
        half_width: Ring distance up to which two nodes are lattice neighbours.
        p_near: Edge probability between lattice neighbours.
        p_shortcut: Edge probability between all other pairs.
        num_arcs: Number of contiguous arcs reported as ``block``.
        seed: Seed for the edges.

    Returns:
        The graph; ``kind`` is ``lattice`` or ``shortcut``. Node order carries no
        information to the model (features are the identity).
    """
    rng = np.random.default_rng(seed)
    index = np.arange(num_nodes)
    gap = np.abs(index[:, None] - index[None, :])
    distance = np.minimum(gap, num_nodes - gap)
    near = distance <= half_width
    probability = np.where(near, p_near, p_shortcut)
    kind = np.where(near, "lattice", "shortcut")
    block = index * num_arcs // num_nodes
    return _sample(probability, block.astype(np.int64), kind, rng)


def _sample(
    probability: npt.NDArray[np.float64],
    block: npt.NDArray[np.int64],
    kind: npt.NDArray[np.str_],
    rng: np.random.Generator,
) -> BlockGraph:
    """Draws a symmetric simple graph from ``probability`` (diagonal zeroed)."""
    probability = probability.copy()
    np.fill_diagonal(probability, 0.0)
    num_nodes = probability.shape[0]
    upper = np.triu(rng.random((num_nodes, num_nodes)) < probability, k=1)
    adjacency = sp.csr_matrix((upper | upper.T).astype(np.float64))
    return BlockGraph(adjacency, probability, block, kind.astype(object))
