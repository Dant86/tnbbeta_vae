"""Block-structured synthetic graphs with known edge probabilities.

A generalization of :mod:`tnbbeta_vae.data.stochastic_block_model` whose block affinity
matrix is arbitrary (assortative, bipartite / disassortative, core-periphery, ...), with
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
    "core_periphery_spec",
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


def core_periphery_spec(
    num_groups: int = 5,
    core_size: int = 20,
    periphery_size: int = 80,
    p_core_core: float = 0.5,
    p_core_periphery: float = 0.2,
    p_other: float = 0.004,
) -> BlockSpec:
    """Groups, each a dense core with a sparse periphery that attaches to the core.

    Blocks ``0..num_groups-1`` are the cores and ``num_groups..2*num_groups-1`` the
    peripheries. Within a group, cores link densely (``p_core_core``) and to their own
    periphery (``p_core_periphery``); every other pair, including periphery-periphery
    even within a group, links with ``p_other``. So periphery nodes sit near their core
    in a dot-product embedding yet should not link to each other: a disassortative
    pattern a dot-product decoder cannot represent exactly. Each periphery node has
    about ``core_size * p_core_periphery`` core neighbours, so its group is recoverable
    from structure.
    """
    k = 2 * num_groups
    affinity = np.full((k, k), p_other)
    kinds = [["periphery-periphery-diff"] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            core_a, core_b = a < num_groups, b < num_groups
            group_a, group_b = a % num_groups, b % num_groups
            same = group_a == group_b
            if core_a and core_b:
                kinds[a][b] = "core-core-same" if same else "core-core-diff"
                if same:
                    affinity[a, b] = p_core_core
            elif core_a != core_b:
                kinds[a][b] = "core-own-periphery" if same else "core-other-periphery"
                if same:
                    affinity[a, b] = p_core_periphery
            elif same:
                kinds[a][b] = "periphery-periphery-same"
    return BlockSpec(
        affinity, [core_size] * num_groups + [periphery_size] * num_groups, kinds
    )
