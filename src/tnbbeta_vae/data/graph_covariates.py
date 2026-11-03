"""Graph-side covariates (Set G) of the metric-first graph-search study.

Every covariate is computed from one regenerated draw (``sample_graph(spec, seed)`` and
its edge split), with no trained model; see
``writeup/results/graph_search_metric_first_preregistration_2026-11-02.md`` for the
definitions this module implements. ``P`` is the draw's true probability matrix and
``t = P / (P + rho)`` its balanced truth.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from tnbbeta_vae.data.graph_space import GraphSpec, SyntheticGraph, sample_graph
from tnbbeta_vae.data.planetoid import LinkSplit, split_edges

__all__ = ["COVARIATE_NAMES", "config_axes", "draw_covariates", "draw_statistics"]

RANK = 16
COVARIATE_NAMES = (
    "log_n",
    "log_nominal_degree",
    "log_groups",
    "uses_groups",
    "hetero_level",
    "informative_features",
    "oracle_auc",
    "realised_degree",
    "expected_degree_cv",
    "clustering",
    "assortativity",
    "spectral_gap",
    "affinity_erank",
    "logit_rank16",
    "high_prob_fraction",
    "mass_top1pct",
    "clip_fraction",
    "log_kl0",
    "leaf_fraction",
    "giant_fraction",
)
_NO_GROUPS = ("bipartite", "scale_free", "ring")
_HETERO = {"none": 0.0, "mild": 1.0, "heavy": 2.0}


def config_axes(spec: GraphSpec) -> dict[str, float]:
    """Returns the covariates g1-g6 read directly from the config."""
    uses = spec.family not in _NO_GROUPS
    return {
        "log_n": math.log(spec.num_nodes),
        "log_nominal_degree": math.log(spec.mean_degree),
        "log_groups": math.log(max(2, spec.num_groups)) if uses else 0.0,
        "uses_groups": float(uses),
        "hetero_level": _HETERO[spec.heterogeneity],
        "informative_features": float(spec.features == "informative"),
        "oracle_auc": float(spec.achieved_auc),
    }


def draw_covariates(spec: GraphSpec, seed: int) -> dict[str, float]:
    """Regenerates one draw and returns all Set G covariates (g1-g19, ``kl0``).

    Args:
        spec: The calibrated graph spec.
        seed: The draw's seed (``SEED_BASE[stage] + 1000 * config index + draw``).

    Returns:
        The covariates by name (``log_kl0`` plus the raw ``kl0``).
    """
    sample = sample_graph(spec, seed)
    split = split_edges(sample.adjacency, seed=seed)
    return {**config_axes(spec), **draw_statistics(sample, split)}


def draw_statistics(sample: SyntheticGraph, split: LinkSplit) -> dict[str, float]:
    """Computes g7-g19 and ``kl0`` for one sampled graph and its split."""
    probability = np.asarray(sample.probability, dtype=np.float64)
    n = probability.shape[0]
    adjacency = sample.adjacency.tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    expected = probability.sum(1)
    upper = np.triu_indices(n, k=1)
    pair = probability[upper]
    rho = float(probability.sum() / (n * (n - 1)))
    truth = probability / (probability + rho)
    out = {
        "realised_degree": float(degree.mean()),
        "expected_degree_cv": float(expected.std() / max(expected.mean(), 1e-12)),
        "clustering": _transitivity(adjacency, degree),
        "assortativity": _assortativity(adjacency, degree),
        "high_prob_fraction": float(np.mean(pair > rho)),
        "mass_top1pct": _top_mass(pair, 0.01),
        "clip_fraction": float(np.mean(pair >= 1.0)),
    }
    out["spectral_gap"] = _spectral_gap(probability, expected)
    out["affinity_erank"] = _effective_rank(probability)
    out["logit_rank16"] = _logit_rank_fit(truth)
    kl0 = _constant_predictor_kl(truth, split)
    out["kl0"] = kl0
    out["log_kl0"] = math.log(max(kl0, 1e-12))
    train = split.train_adjacency.tocsr()
    train_degree = np.asarray(train.sum(1)).ravel()
    out["leaf_fraction"] = float(np.mean(train_degree <= 1))
    _, labels = connected_components(train, directed=False)
    out["giant_fraction"] = float(np.bincount(labels).max() / n)
    return out


def _transitivity(adjacency: sp.csr_matrix, degree: np.ndarray) -> float:
    closed = float(adjacency.multiply(adjacency @ adjacency).sum()) / 6.0
    wedges = float((degree * (degree - 1.0)).sum()) / 2.0
    return 3.0 * closed / wedges if wedges > 0 else 0.0


def _assortativity(adjacency: sp.csr_matrix, degree: np.ndarray) -> float:
    coo = adjacency.tocoo()
    x, y = degree[coo.row], degree[coo.col]
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _top_mass(pair: npt.NDArray[np.float64], fraction: float) -> float:
    total = pair.sum()
    if total <= 0:
        return 0.0
    count = max(1, int(round(fraction * pair.size)))
    top = np.partition(pair, pair.size - count)[pair.size - count :]
    return float(top.sum() / total)


def _spectral_gap(probability: np.ndarray, expected: np.ndarray) -> float:
    scale = 1.0 / np.sqrt(np.maximum(expected, 1e-12))
    normalised = probability * scale[:, None] * scale[None, :]
    values = np.linalg.eigvalsh(normalised)
    return float(1.0 - values[-2]) if values.size > 1 else 0.0


def _effective_rank(probability: np.ndarray) -> float:
    singular = np.abs(np.linalg.eigvalsh(probability))
    total = singular.sum()
    if total <= 0:
        return 0.0
    share = singular / total
    share = share[share > 0]
    return float(np.exp(-(share * np.log(share)).sum()))


def _logit_rank_fit(truth: np.ndarray) -> float:
    n = truth.shape[0]
    clipped = np.clip(truth, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    off = ~np.eye(n, dtype=bool)
    mean = logit[off].mean()
    filled = logit.copy()
    np.fill_diagonal(filled, mean)
    values, vectors = np.linalg.eigh(filled)
    top = np.argsort(-np.abs(values))[:RANK]
    approx = (vectors[:, top] * values[top]) @ vectors[:, top].T
    residual = float(((logit - approx)[off] ** 2).sum())
    total = float(((logit - mean)[off] ** 2).sum())
    return 1.0 - residual / total if total > 0 else 1.0


def _constant_predictor_kl(truth: np.ndarray, split: LinkSplit) -> float:
    held = np.triu(split.train_adjacency.toarray() == 0, k=1)
    t = np.clip(truth[held], 1e-6, 1 - 1e-6)
    p = np.clip(t.mean(), 1e-6, 1 - 1e-6)
    return float((t * np.log(t / p) + (1 - t) * np.log((1 - t) / (1 - p))).mean())
