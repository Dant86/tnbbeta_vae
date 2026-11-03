"""Extra per-run metrics for the metric-first graph-search stage (``fresh``).

``apps.synthetic.graph_search.evaluate`` scores a trained ``GraphVAE`` on the six
pre-registered metrics. The metric-first study
(``writeup/results/graph_search_metric_first_preregistration_2026-11-02.md``) adds
calibration (ECE), ranking against hard negatives (Hits@K, MRR), graph-statistic
distances to the generator's own statistics, latent recovery, alternative truth-KL
weightings and the learned decoder scalars. They live here, additive to the legacy
metrics, and are only computed for the ``fresh`` stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
import torch

from tnbbeta_vae.data.graph_space import GraphSpec
from tnbbeta_vae.models.predictive import (
    balanced_truth,
    dense_edge_probabilities,
    expected_calibration_error,
)

if TYPE_CHECKING:
    from tnbbeta_vae.data.graph_space import SyntheticGraph
    from tnbbeta_vae.data.planetoid import LinkSplit
    from tnbbeta_vae.models.graph_vae import GraphBatch, GraphVAE

__all__ = [
    "block_labels",
    "extended_metrics",
    "graph_statistic_distances",
    "hard_negative_ranking",
    "kmeans_labels",
    "latent_recovery",
    "onmi_lfk",
]

SAMPLES = 32
HARD_NEGATIVES = 100
GRAPH_SAMPLES = 30
_PPR_ALPHA = 0.85
_COVER_THRESHOLD = 0.25
_PAIR_SAMPLE = 200_000


def extended_metrics(
    model: GraphVAE,
    batch: GraphBatch,
    sample: SyntheticGraph,
    split: LinkSplit,
    spec: GraphSpec,
    *,
    seed: int,
) -> dict[str, float]:
    """Computes every extended metric of one trained model on one graph draw.

    Args:
        model: The trained model.
        batch: The graph batch over the *training* adjacency (as in ``evaluate``).
        sample: The sampled graph with its true probabilities.
        split: The edge split.
        spec: The graph spec of the draw.
        seed: Seed (a forked generator is used, so legacy metrics are unaffected).

    Returns:
        ``ece``, ``hard_*`` / ``hardc_*`` (Hits@10, Hits@50, MRR scored by the mean
        sampled prediction and by the centre), ``x_truth_kl`` and its ``eps = 1e-4`` and
        pair-weighted variants and decile split, ``gs_*`` graph-statistic distances,
        ``latent_*`` recovery scores (nan where undefined), and the learned
        ``temperature``, ``logit_bias`` and ``kappa_log_mean``.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 17)
        out: dict[str, float] = {}
        dense = dense_edge_probabilities(model, batch, num_samples=SAMPLES)
        centre_dense = dense_edge_probabilities(model, batch, num_samples=None)
        posterior, _ = model.posterior_and_prior(batch)
        probability = torch.as_tensor(sample.probability, dtype=torch.float32)
        truth = balanced_truth(probability)
        held = np.triu(split.train_adjacency.toarray() == 0, k=1)
        held_mask = torch.as_tensor(held)

        positive, negative = split.test_positive, split.test_negative
        pairs = np.concatenate([positive, negative], axis=1)
        targets = torch.cat(
            [torch.ones(positive.shape[1]), torch.zeros(negative.shape[1])]
        )
        mean = dense[pairs[0], pairs[1]]
        out["ece"] = float(expected_calibration_error(mean, targets))
        out.update(_hard_metrics(sample, split, dense, centre_dense, seed))
        out.update(_truth_kl_variants(truth, probability, dense, held_mask))
        out.update(_graph_stats(probability, dense, spec.num_nodes, seed))
        centres = _centres(model, batch)
        out.update(latent_recovery(spec, spec.num_nodes, centres, seed))
        out["temperature"] = (
            1.0
            if model.config.family == "gaussian"
            else float(model.temperature().detach())
        )
        out["logit_bias"] = float(model.logit_bias().detach())
        out["kappa_log_mean"] = _log_kappa(posterior, model.config.family)
    return out


def hard_negative_ranking(
    score: np.ndarray,
    positives: np.ndarray,
    full_adjacency: np.ndarray,
    common_neighbours: np.ndarray,
    ppr: np.ndarray,
    rng: np.random.Generator,
    *,
    num_negatives: int = HARD_NEGATIVES,
    ks: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    """Hits@K and MRR of each test positive against its hard negatives (HeaRT-style).

    For positive ``(a, b)`` an endpoint is chosen as the anchor ``u`` at random; its
    negatives are non-neighbours ``k`` of ``u`` in the *full* graph, the
    ``num_negatives / 2`` with the highest training-graph common-neighbour count and
    then the same number with the highest personalised-PageRank score among the rest
    (random tie-breaking; all remaining candidates if fewer exist). The positive's rank
    among itself and its negatives counts ties as half.

    Args:
        score: ``(n, n)`` pair scores (larger = more likely an edge).
        positives: ``(2, m)`` test positives.
        full_adjacency: Dense boolean adjacency of the full graph.
        common_neighbours: ``(n, n)`` common-neighbour counts in the training graph.
        ppr: ``(n, n)`` personalised PageRank scores of the training graph.
        rng: Random generator (anchor and tie-breaking).
        num_negatives: Negatives per positive.
        ks: Cut-offs for Hits@K.

    Returns:
        ``hits<K>`` for each K and ``mrr``.
    """
    half = num_negatives // 2
    ranks = []
    for a, b in positives.T:
        u, v = (int(a), int(b)) if rng.random() < 0.5 else (int(b), int(a))
        candidate = ~full_adjacency[u]
        candidate[u] = False
        by_cn = _top(common_neighbours[u], candidate, half, rng)
        candidate[by_cn] = False
        by_ppr = _top(ppr[u], candidate, num_negatives - by_cn.size, rng)
        negatives = np.concatenate([by_cn, by_ppr])
        if negatives.size == 0:
            continue
        negative_scores = score[u, negatives]
        positive_score = score[u, v]
        ranks.append(
            1.0
            + float((negative_scores > positive_score).sum())
            + 0.5 * float((negative_scores == positive_score).sum())
        )
    values = np.asarray(ranks)
    if values.size == 0:
        return {**{f"hits{k}": float("nan") for k in ks}, "mrr": float("nan")}
    return {
        **{f"hits{k}": float((values <= k).mean()) for k in ks},
        "mrr": float((1.0 / values).mean()),
    }


def graph_statistic_distances(
    predicted_edge_probability: torch.Tensor,
    true_probability: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Distances between graphs sampled from model and true edge probabilities.

    ``samples`` graphs are drawn from each matrix (independent Bernoulli per pair).
    Reported: ``gs_degree_w1``, the 1-Wasserstein distance between the pooled degree
    samples divided by the true mean degree; ``gs_clustering_err`` and
    ``gs_triangle_err``, the symmetric relative error ``|a - b| / (a + b)`` of the mean
    global clustering coefficient and of the mean triangles per node.

    Args:
        predicted_edge_probability: ``(n, n)`` probabilities of the model's graph.
        true_probability: ``(n, n)`` true generator probabilities.
        samples: Graphs drawn from each.
        generator: Random generator for the draws.

    Returns:
        The three distances.
    """
    model_degrees, model_clustering, model_triangles = _sample_statistics(
        predicted_edge_probability, samples, generator
    )
    true_degrees, true_clustering, true_triangles = _sample_statistics(
        true_probability, samples, generator
    )
    scale = max(float(true_degrees.mean()), 1e-12)
    return {
        "gs_degree_w1": float(
            wasserstein_distance(model_degrees.ravel(), true_degrees.ravel()) / scale
        ),
        "gs_clustering_err": _symmetric_error(model_clustering, true_clustering),
        "gs_triangle_err": _symmetric_error(model_triangles, true_triangles),
    }


def block_labels(spec: GraphSpec) -> tuple[np.ndarray, int] | None:
    """Returns the true block labels and their count for block families, else None."""
    n = spec.num_nodes
    k = max(2, spec.num_groups)
    index = np.arange(n)
    if spec.family in ("assortative", "degree_corrected", "core_periphery"):
        return index * k // n, k
    if spec.family == "hierarchical":
        blocks = max(4, k + k % 2)
        return index * blocks // n, blocks
    if spec.family == "bipartite":
        return index * 2 // n, 2
    return None


def latent_recovery(
    spec: GraphSpec, num_nodes: int, centres: np.ndarray, seed: int
) -> dict[str, float]:
    """Scores how well posterior centres recover the generator's latent structure.

    The generator's latent variables are re-derived from the draw's seed (they are
    the first random draw of ``graph_space.true_probability``). Returns, where
    defined and otherwise nan:

    * ``latent_gram_rho`` (``latent_kernel``, ``geometric``, ``ring``): Spearman
      correlation of the centres' cosine similarity with the true latent kernel
      (inner product, negative squared toroidal distance, negative ring gap) over a
      pair sample;
    * ``latent_procrustes`` (``latent_kernel`` only): ``||C R - Z||_F^2 / n`` for unit
      centres ``C`` and the true unit latent ``Z`` (zero-padded to the centre
      dimension) after the best orthogonal alignment ``R``; lower is better;
    * ``latent_nmi`` (block families and ``mixed_membership``): overlapping NMI
      (McDaid et al. 2011, LFK-normalised) between k-means clusters of the centres and
      the true blocks (for ``mixed_membership`` the cover is every block holding at
      least 0.25 of the node's membership, always including its largest).

    Args:
        spec: The graph spec.
        num_nodes: Number of nodes.
        centres: ``(n, d)`` posterior centres.
        seed: The draw's seed.

    Returns:
        The three scores (nan when undefined for the family).
    """
    out = {
        "latent_gram_rho": float("nan"),
        "latent_procrustes": float("nan"),
        "latent_nmi": float("nan"),
    }
    unit = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    k = max(2, spec.num_groups)
    n = num_nodes
    kernel: np.ndarray | None = None
    if spec.family == "latent_kernel":
        z = rng.standard_normal((n, k))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        kernel = z @ z.T
        padded = np.zeros((n, unit.shape[1]))
        padded[:, : min(k, unit.shape[1])] = z[:, : unit.shape[1]]
        u, _, vt = np.linalg.svd(unit.T @ padded)
        rotation = u @ vt
        out["latent_procrustes"] = float(np.sum((unit @ rotation - padded) ** 2) / n)
    elif spec.family == "geometric":
        points = rng.random((n, 2 + int(k > 10)))
        gap = np.abs(points[:, None, :] - points[None, :, :])
        gap = np.minimum(gap, 1.0 - gap)
        kernel = -(gap**2).sum(-1)
    elif spec.family == "ring":
        index = np.arange(n)
        gap = np.abs(index[:, None] - index[None, :])
        kernel = -np.minimum(gap, n - gap).astype(np.float64)
    if kernel is not None:
        pair_rng = np.random.default_rng(seed + 1)
        rows = pair_rng.integers(0, n, _PAIR_SAMPLE)
        cols = pair_rng.integers(0, n, _PAIR_SAMPLE)
        keep = rows != cols
        similarity = (unit[rows[keep]] * unit[cols[keep]]).sum(1)
        statistic = cast("Any", spearmanr(similarity, kernel[rows[keep], cols[keep]]))[
            0
        ]
        out["latent_gram_rho"] = float(statistic)
    blocks = block_labels(spec)
    truth_cover: list[np.ndarray] | None = None
    clusters = 0
    if blocks is not None:
        labels, clusters = blocks
        truth_cover = [labels == j for j in range(clusters)]
    elif spec.family == "mixed_membership":
        membership = rng.dirichlet(np.full(k, 0.3), size=n)
        member = membership >= _COVER_THRESHOLD
        member[np.arange(n), membership.argmax(1)] = True
        truth_cover = [member[:, j] for j in range(k)]
        clusters = k
    if truth_cover is not None:
        assigned = kmeans_labels(unit, clusters, np.random.default_rng(seed + 2))
        found = [assigned == j for j in range(clusters)]
        out["latent_nmi"] = onmi_lfk(found, truth_cover)
    return out


def kmeans_labels(
    points: np.ndarray, k: int, rng: np.random.Generator, *, restarts: int = 5
) -> np.ndarray:
    """K-means (k-means++ start, best of ``restarts`` by inertia) cluster labels."""
    best_labels = np.zeros(points.shape[0], dtype=np.int64)
    best_inertia = np.inf
    for _ in range(restarts):
        centres = [points[rng.integers(points.shape[0])]]
        for _ in range(1, k):
            distance = np.min([((points - c) ** 2).sum(1) for c in centres], axis=0)
            total = distance.sum()
            probability = distance / total if total > 0 else None
            centres.append(points[rng.choice(points.shape[0], p=probability)])
        centre_array = np.stack(centres)
        labels = np.zeros(points.shape[0], dtype=np.int64)
        for _ in range(50):
            distances = ((points[:, None, :] - centre_array[None]) ** 2).sum(-1)
            labels = distances.argmin(1)
            updated = np.stack(
                [
                    points[labels == j].mean(0)
                    if np.any(labels == j)
                    else centre_array[j]
                    for j in range(k)
                ]
            )
            if np.allclose(updated, centre_array):
                break
            centre_array = updated
        inertia = float(((points - centre_array[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def onmi_lfk(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    """Overlapping NMI of two covers (McDaid, Greene & Hurley 2011, LFK normalisation).

    Args:
        first: Boolean membership vectors, one per community.
        second: Boolean membership vectors of the other cover.

    Returns:
        ``1 - (H(X|Y)_norm + H(Y|X)_norm) / 2``, in ``[0, 1]`` (1 = identical covers).
    """
    x = np.stack(first).astype(np.float64)
    y = np.stack(second).astype(np.float64)
    return float(
        1.0 - 0.5 * (_normalised_conditional(x, y) + _normalised_conditional(y, x))
    )


def _normalised_conditional(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[1]
    n11 = x @ y.T
    n10 = x.sum(1)[:, None] - n11
    n01 = y.sum(1)[None, :] - n11
    n00 = n - n11 - n10 - n01
    cells = np.stack([n11, n10, n01, n00]) / n
    h = _plogp(cells)
    joint = h.sum(0)
    py = y.sum(1) / n
    h_y = _plogp(py) + _plogp(1.0 - py)
    h_x = _plogp(x.sum(1) / n) + _plogp(1.0 - x.sum(1) / n)
    valid = (h[0] + h[3]) > (h[1] + h[2])
    conditional = np.where(valid, joint - h_y[None, :], np.inf)
    best = np.minimum(conditional.min(1), h_x)
    keep = h_x > 0
    if not np.any(keep):
        return 1.0
    return float(np.mean(best[keep] / h_x[keep]))


def _plogp(p: np.ndarray) -> np.ndarray:
    safe = np.clip(p, 1e-300, None)
    return np.where(p > 0, -p * np.log(safe), 0.0)


def _top(
    score: np.ndarray, candidate: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of the ``count`` best-scoring candidates (random tie-breaking)."""
    pool = np.flatnonzero(candidate)
    if count <= 0 or pool.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((rng.random(pool.size), -score[pool]))
    return pool[order[:count]]


def _hard_metrics(
    sample: SyntheticGraph,
    split: LinkSplit,
    dense: torch.Tensor,
    centre_dense: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    train = split.train_adjacency.toarray().astype(np.float32)
    common = train @ train
    degree = np.maximum(train.sum(1), 1.0)
    n = train.shape[0]
    walk = train / degree[:, None]
    ppr = (1 - _PPR_ALPHA) * np.linalg.inv(np.eye(n) - _PPR_ALPHA * walk)
    full = sample.adjacency.toarray() > 0
    out = {}
    for prefix, score in (("hard", dense), ("hardc", centre_dense)):
        result = hard_negative_ranking(
            score.numpy(),
            split.test_positive,
            full,
            common,
            ppr,
            np.random.default_rng(seed + 3),
        )
        out.update({f"{prefix}_{key}": value for key, value in result.items()})
    return out


def _truth_kl_variants(
    truth: torch.Tensor,
    probability: torch.Tensor,
    dense: torch.Tensor,
    held: torch.Tensor,
) -> dict[str, float]:
    t0, p0 = truth[held], dense[held]
    out = {"x_truth_kl": float(_kl(t0, p0, 1e-6).mean())}
    out["truth_kl_eps1e4"] = float(_kl(t0, p0, 1e-4).mean())
    pair = probability[held].double()
    positive = pair / pair.sum()
    negative = (1.0 - pair) / (1.0 - pair).sum()
    weight = 0.5 * (positive + negative)
    target = (positive / (positive + negative)).float()
    out["truth_kl_w"] = float((weight * _kl(target, p0, 1e-6).double()).sum())
    elementwise = _kl(t0, p0, 1e-6)
    edges = torch.quantile(t0, torch.linspace(0, 1, 11))
    bucket = torch.bucketize(t0, edges[1:-1])
    for index in range(10):
        chosen = bucket == index
        out[f"truth_kl_dec{index}"] = (
            float(elementwise[chosen].mean()) if bool(chosen.any()) else float("nan")
        )
    return out


def _kl(truth: torch.Tensor, prediction: torch.Tensor, eps: float) -> torch.Tensor:
    t = truth.clamp(eps, 1 - eps)
    p = prediction.clamp(eps, 1 - eps)
    return t * (t / p).log() + (1 - t) * ((1 - t) / (1 - p)).log()


def _graph_stats(
    probability: torch.Tensor, dense: torch.Tensor, num_nodes: int, seed: int
) -> dict[str, float]:
    rho = probability.sum() / (num_nodes * (num_nodes - 1))
    clipped = dense.clamp(max=1 - 1e-6)
    implied = (rho * clipped / (1 - clipped)).clamp(max=1.0)
    implied.fill_diagonal_(0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed + 5)
    return graph_statistic_distances(
        implied.to(device),
        probability.to(device),
        samples=GRAPH_SAMPLES,
        generator=generator,
    )


def _sample_statistics(
    probability: torch.Tensor, samples: int, generator: torch.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = probability.shape[0]
    upper = torch.triu(torch.ones(n, n, device=probability.device), diagonal=1)
    degrees, clustering, triangles = [], [], []
    for _ in range(samples):
        draw = torch.rand(
            probability.shape, device=probability.device, generator=generator
        )
        half = ((draw < probability).float()) * upper
        adjacency = half + half.T
        degree = adjacency.sum(1)
        closed = ((adjacency @ adjacency) * adjacency).sum() / 6.0
        wedges = (degree * (degree - 1.0)).sum() / 2.0
        degrees.append(degree.cpu().numpy())
        triangles.append(float(closed) / n)
        clustering.append(float(3.0 * closed / wedges) if float(wedges) > 0 else 0.0)
    return np.stack(degrees), np.asarray(clustering), np.asarray(triangles)


def _symmetric_error(model: np.ndarray, truth: np.ndarray) -> float:
    a, b = float(model.mean()), float(truth.mean())
    return abs(a - b) / (a + b) if a + b > 0 else 0.0


def _centres(model: GraphVAE, batch: GraphBatch) -> np.ndarray:
    centres: Any = model.embeddings(batch)
    return np.asarray(centres.detach().cpu(), dtype=np.float64)


def _log_kappa(posterior: Any, family: str) -> float:
    if family not in ("vmf", "power_spherical"):
        return float("nan")
    kappa = getattr(posterior, "kappa", None)
    if kappa is None:
        kappa = getattr(posterior, "scale", None)
    if kappa is None:
        return float("nan")
    return float(torch.as_tensor(kappa).clamp_min(1e-12).log().mean())
