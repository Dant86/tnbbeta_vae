"""Tests for tnbbeta_vae.data.cluster_mixture."""

from __future__ import annotations

import torch

from tnbbeta_vae.data.cluster_mixture import cluster_mixture_data, positive_pairs


def test_shapes() -> None:
    features, labels = cluster_mixture_data(
        num_clusters=5, points_per_cluster=10, dim=8
    )

    assert features.shape == (50, 8)
    assert labels.shape == (50,)
    assert set(labels.unique().tolist()) == set(range(5))


def test_reproducible_given_a_seeded_generator() -> None:
    kwargs = {"num_clusters": 4, "points_per_cluster": 6, "dim": 3}
    first_features, first_labels = cluster_mixture_data(
        **kwargs, generator=torch.Generator().manual_seed(0)
    )
    second_features, second_labels = cluster_mixture_data(
        **kwargs, generator=torch.Generator().manual_seed(0)
    )

    assert torch.equal(first_features, second_features)
    assert torch.equal(first_labels, second_labels)


def test_clusters_are_separable_well_above_chance() -> None:
    """Each point's nearest OTHER point is usually in its own cluster.

    With the default ``cluster_std``/``center_scale`` the clusters should be
    clearly separable -- far better than the 1/num_clusters chance rate a
    random assignment would give.
    """
    generator = torch.Generator().manual_seed(0)
    features, labels = cluster_mixture_data(
        num_clusters=10, points_per_cluster=50, generator=generator
    )
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(torch.inf)
    nearest = distances.argmin(dim=1)

    same_cluster_rate = (labels[nearest] == labels).float().mean().item()

    assert same_cluster_rate > 0.95  # chance would be ~1/10 = 0.1


def test_positive_pairs_are_always_same_cluster_and_distinct() -> None:
    data_generator = torch.Generator().manual_seed(0)
    _, labels = cluster_mixture_data(
        num_clusters=5, points_per_cluster=10, generator=data_generator
    )

    pairs = positive_pairs(labels, 200, generator=torch.Generator().manual_seed(1))

    assert pairs.shape == (2, 200)
    assert torch.equal(labels[pairs[0]], labels[pairs[1]])
    assert (pairs[0] != pairs[1]).all()


def test_positive_pairs_reproducible_given_a_seeded_generator() -> None:
    data_generator = torch.Generator().manual_seed(0)
    _, labels = cluster_mixture_data(
        num_clusters=5, points_per_cluster=10, generator=data_generator
    )

    first = positive_pairs(labels, 50, generator=torch.Generator().manual_seed(2))
    second = positive_pairs(labels, 50, generator=torch.Generator().manual_seed(2))

    assert torch.equal(first, second)
