"""Block-structured synthetic graphs with known edge probabilities.

A generalization of :mod:`tnbbeta_vae.data.stochastic_block_model` whose block affinity
matrix is arbitrary (assortative, bipartite / disassortative, hub-and-spoke, ...), with
optional degree heterogeneity, and which returns the *true* probability of every node
pair. Link-prediction metrics can then be compared against ground truth, not just
against held-out edges.

``P_ij = clip(w_i * w_j * B[block_i, block_j], 0, 1)`` for ``i != j``, with ``w = 1``
unless ``degree_exponent`` is given (a Pareto weight with that tail index, normalized to
mean 1), which makes it a degree-corrected block model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

__all__ = [
    "BlockGraph",
    "BlockSpec",
    "assortative_spec",
    "bipartite_spec",
    "block_graph",
    "hub_spoke_spec",
]


@dataclass
class BlockSpec:
    """A block model's structure.

    Attributes:
        affinity: Symmetric block edge probabilities, shape ``(k, k)``.
        sizes: Number of nodes per block, length ``k``.
        kinds: ``kinds[a][b]`` names the kind of pair a block pair ``(a, b)`` is (used
            to break metrics down, e.g. ``"leaf-leaf-same-hub"``), symmetric.
    """

    affinity: npt.NDArray[np.float64]
    sizes: list[int]
    kinds: list[list[str]]


@dataclass
class BlockGraph:
    """A sampled graph and its ground truth.

    Attributes:
        adjacency: Symmetric binary adjacency without self-loops.
        probability: True edge probability of every ordered pair, shape ``(n, n)``,
            symmetric with a zero diagonal.
        block: Block index of each node, shape ``(n,)``.
        kind: Kind name of each node pair, shape ``(n, n)`` (object array).
    """

    adjacency: sp.csr_matrix
    probability: npt.NDArray[np.float64]
    block: npt.NDArray[np.int64]
    kind: npt.NDArray[np.object_]


def block_graph(
    spec: BlockSpec, *, seed: int = 0, degree_exponent: float | None = None
) -> BlockGraph:
    """Samples a graph from ``spec``.

    Args:
        spec: Block affinities, sizes and pair kinds.
        seed: Seed for the degree weights and the edges.
        degree_exponent: If given, the tail index of Pareto degree weights (smaller is
            heavier-tailed; 2.5 gives a visibly skewed degree distribution).

    Returns:
        The graph with its true pair probabilities.
    """
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(len(spec.sizes)), spec.sizes)
    num_nodes = block.shape[0]
    weights = np.ones(num_nodes)
    if degree_exponent is not None:
        weights = rng.pareto(degree_exponent, size=num_nodes) + 1.0
        weights /= weights.mean()
    probability = np.clip(
        np.outer(weights, weights) * spec.affinity[np.ix_(block, block)], 0.0, 1.0
    )
    np.fill_diagonal(probability, 0.0)
    upper = np.triu(rng.random((num_nodes, num_nodes)) < probability, k=1)
    adjacency = sp.csr_matrix((upper | upper.T).astype(np.float64))
    kinds = np.empty((len(spec.sizes), len(spec.sizes)), dtype=object)
    for a, row in enumerate(spec.kinds):
        for b, name in enumerate(row):
            kinds[a, b] = name
    return BlockGraph(adjacency, probability, block, kinds[np.ix_(block, block)])


def assortative_spec(
    num_blocks: int = 5, size: int = 100, p_in: float = 0.15, p_out: float = 0.01
) -> BlockSpec:
    """Communities with denser within-block than between-block edges."""
    affinity = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(affinity, p_in)
    kinds = [
        ["within" if a == b else "between" for b in range(num_blocks)]
        for a in range(num_blocks)
    ]
    return BlockSpec(affinity, [size] * num_blocks, kinds)


def bipartite_spec(
    size: int = 250, p_cross: float = 0.08, p_within: float = 0.005
) -> BlockSpec:
    """Two sides with edges mostly across (the extreme disassortative case)."""
    affinity = np.array([[p_within, p_cross], [p_cross, p_within]])
    kinds = [["within", "cross"], ["cross", "within"]]
    return BlockSpec(affinity, [size, size], kinds)


def hub_spoke_spec(
    num_hubs: int = 10,
    leaves_per_hub: int = 50,
    p_hub_leaf: float = 0.6,
    p_leaf_leaf: float = 0.01,
    p_hub_hub: float = 0.5,
) -> BlockSpec:
    """Hubs, each linked to its own leaves, which barely link to each other.

    Blocks ``0..num_hubs-1`` are single hubs; blocks ``num_hubs..2*num_hubs-1`` are each
    hub's leaves. A hub links to its own leaves with ``p_hub_leaf`` and to other hubs'
    leaves never; leaves link to each other, within or across hubs, with
    ``p_leaf_leaf``.
    """
    k = 2 * num_hubs
    affinity = np.full((k, k), p_leaf_leaf)
    kinds = [["leaf-leaf-diff-hub"] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            hub_a, hub_b = a < num_hubs, b < num_hubs
            if hub_a and hub_b:
                affinity[a, b] = p_hub_hub if a != b else 0.0
                kinds[a][b] = "hub-hub"
            elif hub_a != hub_b:
                hub, leaf = (a, b - num_hubs) if hub_a else (b, a - num_hubs)
                affinity[a, b] = p_hub_leaf if hub == leaf else 0.0
                kinds[a][b] = "hub-own-leaf" if hub == leaf else "hub-other-leaf"
            elif a == b:
                kinds[a][b] = "leaf-leaf-same-hub"
    return BlockSpec(affinity, [1] * num_hubs + [leaves_per_hub] * num_hubs, kinds)
