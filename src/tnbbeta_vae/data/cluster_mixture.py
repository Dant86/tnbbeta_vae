"""Synthetic clustered items for the pairwise/ranking-loss ablation.

``K`` Gaussian-blob clusters in a small ``R^d``: each point is its cluster's
randomly-placed center plus Gaussian noise. Built for ``apps/synthetic/
pairwise_cluster_recovery.py``, the "many communities" ablation arm that holds
``GraphVAE``'s exact pairwise dot-product + BCE objective and its plain-MLP
encoder fixed while swapping out the graph itself, to test whether that
objective alone (independent of graph structure, scale or real features)
drives ``TNBBetaSpherical``'s posterior into the bimodal regime the way it
does on every real link-prediction dataset tried.

Unlike com-DBLP's identity ("featureless") node features -- where a GCN's
message passing is what injects structure, since an identity matrix alone
carries no signal about community membership -- this dataset's cluster
identity IS encoded directly in each point's features (its nearby position in
``R^d``). That is deliberate: a plain MLP encoder (no message-passing, no
graph) has no other way to see cluster structure at all, so the Gaussian-blob
placement is the ONLY source of signal this ablation gives it. This is the key
way this task differs from com-DBLP's setup despite both being trained with
the same dot-product+BCE objective.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cluster_mixture_data", "positive_pairs"]


def cluster_mixture_data(
    num_clusters: int = 10,
    points_per_cluster: int = 50,
    dim: int = 20,
    cluster_std: float = 0.5,
    center_scale: float = 5.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draws ``num_clusters`` Gaussian blobs in ``R^dim``.

    Args:
        num_clusters: Number of clusters (``K``).
        points_per_cluster: Number of points per cluster.
        dim: Ambient dimension of the features.
        cluster_std: Standard deviation of the per-point Gaussian noise around
            its cluster's center. Small enough, relative to ``center_scale``,
            that clusters are clearly separable (see
            ``test_clusters_are_separable_well_above_chance``).
        center_scale: Standard deviation of the randomly-placed cluster
            centers. Large enough, relative to ``cluster_std``, that the
            centers land far apart from each other.
        generator: Optional RNG for the centers and the noise.

    Returns:
        ``(features, labels)``: features of shape ``(num_clusters *
        points_per_cluster, dim)``, and labels of shape ``(num_clusters *
        points_per_cluster,)`` giving each point's cluster index (ground
        truth only -- never fed to a model).
    """
    centers = center_scale * torch.randn(num_clusters, dim, generator=generator)
    labels = torch.arange(num_clusters).repeat_interleave(points_per_cluster)
    noise = cluster_std * torch.randn(
        num_clusters * points_per_cluster, dim, generator=generator
    )
    features = centers[labels] + noise
    return features, labels


def positive_pairs(
    labels: Tensor, num_pairs: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draws ``num_pairs`` same-cluster index pairs.

    For each pair, a first item is drawn uniformly at random over all items;
    the second is drawn uniformly at random from the SAME cluster, excluding
    the first item itself (guaranteed distinct via a modular offset within
    that cluster's own index pool).

    Args:
        labels: Cluster index per item, shape ``(num_items,)``.
        num_pairs: Number of pairs to draw.
        generator: Optional RNG.

    Returns:
        Tensor of shape ``(2, num_pairs)``; for every column ``labels[pairs[0]]
        == labels[pairs[1]]`` and ``pairs[0] != pairs[1]``.
    """
    num_items = labels.shape[0]
    first = torch.randint(num_items, (num_pairs,), generator=generator)
    second = first.clone()
    for cluster in labels.unique().tolist():
        pool = (labels == cluster).nonzero(as_tuple=True)[0]
        pool_size = pool.shape[0]
        in_cluster = labels[first] == cluster
        count = int(in_cluster.sum())
        if count == 0:
            continue
        if pool_size < 2:
            # No distinct partner exists in a singleton cluster; pair it with
            # itself rather than crash. A degenerate edge case, not expected
            # at the default points_per_cluster.
            second[in_cluster] = pool[0]
            continue
        position = torch.searchsorted(pool, first[in_cluster])
        shift = torch.randint(1, pool_size, (count,), generator=generator)
        second[in_cluster] = pool[(position + shift) % pool_size]
    return torch.stack([first, second])
