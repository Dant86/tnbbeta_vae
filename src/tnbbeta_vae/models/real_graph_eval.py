"""Scoring helpers for the real-graph revision (``apps/real_graph``).

Pair-level scoring of a trained :class:`~tnbbeta_vae.models.graph_vae.GraphVAE`
without ever building an ``n x n`` matrix (Pubmed and MAG-CS would not fit), the
graph statistics and edge thinning of the density manipulation, training-degree
buckets for test pairs, and per-node posterior-shape statistics. See
``writeup/results/real_graph_revision_preregistration_2026-11-03.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import torch

from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.models.predictive import (
    bernoulli_log_loss,
    expected_calibration_error,
)

if TYPE_CHECKING:
    from tnbbeta_vae.models.graph_vae import GraphBatch, GraphVAE

__all__ = [
    "BUCKET_EDGES",
    "BUCKET_LABELS",
    "degree_buckets",
    "graph_statistics",
    "node_shape_stats",
    "pair_metrics",
    "pair_probabilities",
    "thin_edges",
    "vmf_sampler_name",
]

# Training-degree bucket boundaries: {0-1, 2-3, 4-7, 8-15, 16+}.
BUCKET_EDGES = (2, 4, 8, 16)
BUCKET_LABELS = ("0-1", "2-3", "4-7", "8-15", "16+")


def thin_edges(adjacency: Any, keep_fraction: float, seed: int) -> Any:
    """Keeps a uniformly random ``keep_fraction`` of an undirected graph's edges.

    Args:
        adjacency: Symmetric binary adjacency without self-loops.
        keep_fraction: Fraction of the edges retained, in ``(0, 1]``. ``1.0`` returns
            the graph unchanged.
        seed: Seed of the random choice of retained edges.

    Returns:
        A symmetric binary CSR adjacency of the same shape (nodes are never removed).
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1]; got {keep_fraction}.")
    if keep_fraction == 1.0:
        return adjacency.tocsr()
    upper: Any = sp.triu(adjacency, k=1).tocoo()
    count = upper.row.shape[0]
    keep = np.sort(
        np.random.default_rng(seed).permutation(count)[: round(count * keep_fraction)]
    )
    one_direction: Any = sp.csr_matrix(
        (np.ones(keep.shape[0], dtype=np.float32), (upper.row[keep], upper.col[keep])),
        shape=adjacency.shape,
    )
    return (one_direction + one_direction.T).tocsr()


def graph_statistics(adjacency: Any) -> dict[str, float]:
    """Returns node and edge counts, degree moments, transitivity and isolated share.

    Args:
        adjacency: Symmetric binary adjacency without self-loops.

    Returns:
        ``num_nodes``, ``num_edges``, ``mean_degree``, ``degree_cv`` (std over mean),
        ``transitivity`` (3 x triangles over connected triples) and
        ``isolated_fraction``.
    """
    matrix: Any = adjacency.tocsr().astype(np.float64)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    triangles = float(matrix.multiply(matrix @ matrix).sum()) / 6.0
    triples = float((degree * (degree - 1)).sum()) / 2.0
    mean = float(degree.mean())
    return {
        "num_nodes": float(matrix.shape[0]),
        "num_edges": float(degree.sum() / 2),
        "mean_degree": mean,
        "degree_cv": float(degree.std() / mean) if mean > 0 else float("nan"),
        "transitivity": 3.0 * triangles / triples if triples > 0 else 0.0,
        "isolated_fraction": float((degree == 0).mean()),
    }


def degree_buckets(
    degree: npt.NDArray[np.float64], pairs: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Assigns each pair to the bucket of the lower of its endpoints' degrees.

    Args:
        degree: Training-graph degree per node.
        pairs: Node pairs, shape ``(2, n)``.

    Returns:
        Bucket index per pair into :data:`BUCKET_LABELS`.
    """
    lower = np.minimum(degree[pairs[0]], degree[pairs[1]])
    return np.digitize(lower, BUCKET_EDGES).astype(np.int64)


def vmf_sampler_name(posterior: Any) -> str:
    """Names the vMF sampler a posterior uses (``"n/a"`` for other families).

    Args:
        posterior: A distribution built by the model's head.

    Returns:
        ``"exact_envelope"`` or ``"taylor"`` for a vMF posterior, else ``"n/a"``.
    """
    flag = getattr(posterior, "exact_envelope", None)
    if flag is None:
        return "n/a"
    return "exact_envelope" if flag else "taylor"


@torch.no_grad()
def pair_probabilities(
    model: GraphVAE,
    batch: GraphBatch,
    pairs: npt.NDArray[np.int64],
    *,
    num_samples: int,
    seed: int,
    posterior: Any = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Returns the sampled posterior-predictive and the centre probability of pairs.

    One joint draw of ``num_samples`` samples of every node's latent is taken (seeded
    through a forked generator, so training randomness is unaffected); the pair
    probability is the mean over samples of ``sigmoid(logit(z_i^s, z_j^s))``, and the
    centre probability uses the posterior centres. No ``n x n`` matrix is built.

    Args:
        model: A trained model (switched to evaluation mode).
        batch: The graph the encoder runs over (the *training* adjacency).
        pairs: Node pairs, shape ``(2, n)``.
        num_samples: Number of posterior samples.
        seed: Seed of the sampling.
        posterior: The already-computed posterior (avoids a second encoder pass).

    Returns:
        ``sampled`` and ``centre`` probabilities and the sample standard deviation
        ``spread``, each of shape ``(n,)``.
    """
    from tnbbeta_vae.models.heads import posterior_centre  # noqa: PLC0415

    model.eval()
    device = batch.features.device
    if posterior is None:
        posterior, _ = model.posterior_and_prior(batch)
    index = torch.as_tensor(pairs, device=device)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        samples = posterior.rsample(torch.Size([num_samples]))
    draws = torch.stack(
        [torch.sigmoid(model.link_logits(z, index)) for z in samples]
    ).double()
    centre = posterior_centre(model.config.family, posterior)
    centre_probability = torch.sigmoid(model.link_logits(centre, index)).double()
    return {
        "sampled": draws.mean(0).cpu().numpy(),
        "centre": centre_probability.cpu().numpy(),
        "spread": draws.std(0).cpu().numpy(),
    }


def pair_metrics(
    sampled: npt.NDArray[np.float64],
    centre: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Returns the six pair-level metrics of the study.

    Args:
        sampled: Sampled posterior-predictive probabilities.
        centre: Centre probabilities.
        labels: 1 for held-out edges, 0 for non-edges.

    Returns:
        ``log_loss`` and ``centre_log_loss`` (clamped at 1e-6), ``sampled_auc``,
        ``centre_auc``, ``ap`` (of the sampled probability) and ``ece`` (10 bins).
    """
    positive = labels > 0.5
    target = torch.as_tensor(labels, dtype=torch.float64)
    sampled_tensor = torch.as_tensor(sampled, dtype=torch.float64)
    return {
        "log_loss": float(bernoulli_log_loss(sampled_tensor, target)),
        "centre_log_loss": float(
            bernoulli_log_loss(torch.as_tensor(centre, dtype=torch.float64), target)
        ),
        "sampled_auc": roc_auc(sampled[positive], sampled[~positive]),
        "centre_auc": roc_auc(centre[positive], centre[~positive]),
        "ap": average_precision(sampled[positive], sampled[~positive]),
        "ece": float(expected_calibration_error(sampled_tensor, target)),
    }


@torch.no_grad()
def node_shape_stats(
    model: GraphVAE,
    posterior: Any,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-node posterior-shape statistics.

    Args:
        model: The trained model (for its config).
        posterior: Per-node posteriors over the training graph.
        num_samples: Samples for the second-mode mass.
        seed: Seed of the sampling (forked generator).

    Returns:
        ``second_mode_mass`` (the lesser-hemisphere share of the samples about each
        node's mean direction; ``nan`` for the Gaussian) and, by family, ``kappa``
        (vMF and power-spherical) or ``p``, ``q``, ``epsilon`` and ``m = epsilon -
        (latent_dim - 1) / 2`` (TNBBeta). All have shape ``(n,)``.
    """
    config = model.config
    out: dict[str, npt.NDArray[np.float64]] = {}
    device = model.log_temperature.device
    direction = getattr(posterior, "mean_direction", getattr(posterior, "loc", None))
    if direction is None:
        count = posterior.base_dist.loc.shape[0]
        out["second_mode_mass"] = np.full(count, np.nan)
    else:
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            samples = posterior.rsample(torch.Size([num_samples]))
        opposite = ((samples * direction).sum(-1) < 0).float().mean(dim=0)
        lesser = torch.minimum(opposite, 1 - opposite)
        out["second_mode_mass"] = lesser.double().cpu().numpy()
    if config.family == "vmf":
        out["kappa"] = posterior.scale.reshape(-1).double().cpu().numpy()
    elif config.family == "power_spherical":
        out["kappa"] = posterior.kappa.reshape(-1).double().cpu().numpy()
    elif config.family == "tnbbeta":
        epsilon = posterior.epsilon.double().cpu().numpy()
        out["p"] = posterior.p.double().cpu().numpy()
        out["q"] = posterior.q.double().cpu().numpy()
        out["epsilon"] = epsilon
        out["m"] = epsilon - (config.latent_dim - 1) / 2
    return out
