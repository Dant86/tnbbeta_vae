"""A parameterized space of synthetic graphs with known edge probabilities.

Every graph is a draw from ``P = clip(c * W * w w')`` with a family-specific shape
matrix ``W`` (communities, latent geometry, ...), optional degree weights ``w``, and a
scalar ``c`` fixing the mean degree. Each family has one monotone *contrast* parameter
(how strongly the structure shows); :func:`calibrate_contrast` bisects it so the graph's
expected oracle link-prediction AUC hits a requested target, which makes graphs from
different families comparable in difficulty.

See ``writeup/results/graph_search_preregistration_*.md`` for how the space is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

__all__ = [
    "FAMILIES",
    "GraphSpec",
    "SyntheticGraph",
    "calibrate_contrast",
    "expected_oracle_auc",
    "informative_features",
    "sample_graph",
    "true_probability",
]

FAMILIES = (
    "assortative",
    "bipartite",
    "core_periphery",
    "hierarchical",
    "mixed_membership",
    "latent_kernel",
    "geometric",
    "ring",
    "scale_free",
    "degree_corrected",
)
_HETEROGENEITY_TAIL = {"none": None, "mild": 6.0, "heavy": 2.5}
_FEATURE_DIM = 32
_MAX_CONTRAST = 1.0e4


@dataclass(frozen=True)
class GraphSpec:
    """One point of the graph space.

    Attributes:
        family: One of :data:`FAMILIES`.
        num_nodes: Number of nodes.
        mean_degree: Expected degree (before clipping probabilities at 1).
        num_groups: Blocks (community families) or intrinsic dimension
            (``latent_kernel``, ``geometric``); ignored by the rest.
        heterogeneity: ``none``, ``mild`` or ``heavy`` degree weights (``scale_free``
            and ``degree_corrected`` set their own).
        features: ``none`` (identity features) or ``informative``.
        contrast: Structure strength; see :func:`calibrate_contrast`.
        target_auc: The oracle AUC the contrast was calibrated to.
        achieved_auc: The calibrated graph's expected oracle AUC.
    """

    family: str
    num_nodes: int
    mean_degree: float
    num_groups: int
    heterogeneity: str
    features: str
    contrast: float = 1.0
    target_auc: float = float("nan")
    achieved_auc: float = float("nan")


@dataclass
class SyntheticGraph:
    """A sampled graph and its ground truth.

    Attributes:
        adjacency: Symmetric binary adjacency without self-loops.
        probability: True edge probability of every pair, zero diagonal.
        features: Node features, shape ``(n, f)`` (identity unless informative).
    """

    adjacency: sp.csr_matrix
    probability: npt.NDArray[np.float64]
    features: sp.csr_matrix


def sample_graph(spec: GraphSpec, seed: int) -> SyntheticGraph:
    """Draws one graph (positions, memberships, weights, edges, features) from ``spec``.

    Args:
        spec: A calibrated graph specification.
        seed: Seed of this draw; draws with different seeds are independent.

    Returns:
        The graph with its true pair probabilities and node features.
    """
    rng = np.random.default_rng(seed)
    probability = true_probability(spec, rng)
    num_nodes = spec.num_nodes
    upper = np.triu(rng.random((num_nodes, num_nodes)) < probability, k=1)
    adjacency = sp.csr_matrix((upper | upper.T).astype(np.float64))
    features: sp.csr_matrix = (
        informative_features(probability, rng)
        if spec.features == "informative"
        else sp.csr_matrix(sp.identity(num_nodes))
    )
    return SyntheticGraph(adjacency, probability, features)


def true_probability(
    spec: GraphSpec, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Builds the true pair-probability matrix of one draw of ``spec``."""
    n = spec.num_nodes
    shape = _shape_matrix(spec, rng)
    weights = _degree_weights(spec, rng)
    raw = shape * np.outer(weights, weights)
    np.fill_diagonal(raw, 0.0)
    return _scale_to_degree(raw, spec.mean_degree, n)


def informative_features(
    probability: npt.NDArray[np.float64], rng: np.random.Generator
) -> sp.csr_matrix:
    """Returns node features that are informative about the structure.

    A random projection of each node's true edge-probability row, plus noise of equal
    variance (signal-to-noise 1). Given the truth they are independent of the sampled
    edges, so no held-out edge is encoded.
    """
    projection = rng.standard_normal((probability.shape[0], _FEATURE_DIM))
    clean = probability @ projection
    noise = rng.standard_normal(clean.shape) * clean.std()
    return sp.csr_matrix(clean + noise)


def expected_oracle_auc(probability: npt.NDArray[np.float64]) -> float:
    """Expected AUC of ranking pairs by their true probability.

    Positives are pairs weighted by ``P`` (an edge is drawn proportionally to ``P``) and
    negatives pairs weighted by ``1 - P``, as when held-out edges are scored against
    random non-edges; ties count one half.
    """
    upper = probability[np.triu_indices_from(probability, k=1)]
    values, inverse = np.unique(upper, return_inverse=True)
    positive = np.bincount(inverse, weights=upper)
    negative = np.bincount(inverse, weights=1.0 - upper)
    if positive.sum() <= 0.0:  # no edges at all: the AUC is undefined
        return float("nan")
    below = np.cumsum(negative) - negative
    wins = (positive * (below + 0.5 * negative)).sum()
    return float(wins / (positive.sum() * negative.sum()))


def calibrate_contrast(
    spec: GraphSpec, target_auc: float, *, seed: int = 0, iterations: int = 14
) -> GraphSpec:
    """Sets ``spec.contrast`` so the expected oracle AUC is about ``target_auc``.

    Bisects the contrast on a log scale on one fixed calibration draw. If the target
    exceeds what the family can reach (few blocks cap the AUC), the returned spec has
    the strongest contrast and its ``achieved_auc`` is below ``target_auc``.

    Args:
        spec: The graph specification (its ``contrast`` is ignored).
        target_auc: Desired expected oracle AUC.
        seed: Seed of the calibration draw.
        iterations: Bisection steps.

    Returns:
        A copy of ``spec`` with ``contrast``, ``target_auc`` and ``achieved_auc`` set.
    """

    def auc_at(contrast: float) -> float:
        trial = replace(spec, contrast=contrast)
        value = expected_oracle_auc(
            true_probability(trial, np.random.default_rng(seed))
        )
        # A contrast so strong the affinities underflow to zero is "beyond any target".
        return value if math.isfinite(value) else 1.0

    low, high = math.log(1e-3), math.log(_MAX_CONTRAST)
    if auc_at(math.exp(high)) < target_auc:
        contrast = math.exp(high)
    else:
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            if auc_at(math.exp(mid)) < target_auc:
                low = mid
            else:
                high = mid
        contrast = math.exp(0.5 * (low + high))
    return replace(
        spec, contrast=contrast, target_auc=target_auc, achieved_auc=auc_at(contrast)
    )


def _shape_matrix(spec: GraphSpec, rng: np.random.Generator) -> npt.NDArray[Any]:
    """The family's unnormalized pair affinity ``W`` for one draw."""
    n, k, t = spec.num_nodes, max(2, spec.num_groups), spec.contrast
    index = np.arange(n)
    if spec.family in ("assortative", "degree_corrected"):
        block = index * k // n
        return 1.0 + t * (block[:, None] == block[None, :])
    if spec.family == "bipartite":
        side = index * 2 // n
        return 1.0 + t * (side[:, None] != side[None, :])
    if spec.family == "core_periphery":
        group = index * k // n
        within = (index - np.searchsorted(group, group)) / np.maximum(
            np.bincount(group)[group], 1
        )
        core = within < 0.2
        same = group[:, None] == group[None, :]
        core_core = np.outer(core, core) & same
        core_periphery = (core[:, None] ^ core[None, :]) & same
        return 1.0 + t * core_core + (math.sqrt(1.0 + t) - 1.0) * core_periphery
    if spec.family == "hierarchical":
        blocks = max(4, k + k % 2)
        block = index * blocks // n
        same_block = block[:, None] == block[None, :]
        same_super = (block[:, None] // 2) == (block[None, :] // 2)
        return 1.0 + t * same_block + math.sqrt(t) * (same_super & ~same_block)
    if spec.family == "mixed_membership":
        membership = rng.dirichlet(np.full(k, 0.3), size=n)
        return membership @ (1.0 + t * np.eye(k)) @ membership.T
    if spec.family == "latent_kernel":
        z = rng.standard_normal((n, k))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return np.exp(t * (z @ z.T - 1.0))
    if spec.family == "geometric":
        points = rng.random((n, 2 + int(k > 10)))
        gap = np.abs(points[:, None, :] - points[None, :, :])
        gap = np.minimum(gap, 1.0 - gap)
        return np.exp(-t * (gap**2).sum(-1))
    if spec.family == "ring":
        gap = np.abs(index[:, None] - index[None, :])
        near = np.minimum(gap, n - gap) <= max(2, round(0.02 * n))
        return 1.0 + t * near
    if spec.family == "scale_free":
        return np.ones((n, n))
    raise ValueError(f"unknown graph family {spec.family!r}")


def _degree_weights(
    spec: GraphSpec, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Mean-one Pareto degree weights (heterogeneity axis, or the family's own)."""
    if spec.family == "scale_free":
        tail: float | None = 1.4 + 8.0 / (1.0 + spec.contrast)
    elif spec.family == "degree_corrected":
        tail = 2.5
    else:
        tail = _HETEROGENEITY_TAIL[spec.heterogeneity]
    if tail is None:
        return np.ones(spec.num_nodes)
    weights = rng.pareto(tail, size=spec.num_nodes) + 1.0
    return weights / weights.mean()


def _scale_to_degree(
    raw: npt.NDArray[np.float64], mean_degree: float, num_nodes: int
) -> npt.NDArray[np.float64]:
    """Scales ``raw`` so the clipped probabilities have the requested mean degree."""
    target = mean_degree * num_nodes
    if raw.sum() <= 0.0:
        return np.zeros_like(raw, dtype=np.float64)
    scale = target / raw.sum()
    for _ in range(6):
        probability = np.clip(scale * raw, 0.0, 1.0)
        total = probability.sum()
        if abs(total - target) < 1e-3 * target:
            break
        scale *= target / total
    return np.clip(scale * raw, 0.0, 1.0).astype(np.float64)
