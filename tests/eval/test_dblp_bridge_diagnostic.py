"""Tests for com-DBLP bridge-node diagnostic utilities."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apps.eval.dblp_bridge_diagnostic import (
    _link_prediction_by_bridge_edges,
    _link_prediction_by_community_count,
    _posterior_stats,
)
from tnbbeta_vae.distributions import (
    PowerSpherical,
    TNBBetaSpherical,
    VonMisesFisher,
)


def test_link_prediction_by_bridge_edges_splits_by_real_community_membership() -> None:
    """Classifies test edges by the real primary/secondary assignment, not randomly."""
    # Node 0 is the only bridge node: primary community 0, secondary community 1.
    # Node 1 is a member of community 0 (so edge (0, 1) is a "primary" instance).
    # Node 2 is a member of community 1 (so edge (0, 2) is a "secondary" instance).
    # Node 3/4 belong to neither of node 0's communities and are not bridges themselves,
    # so edge (3, 4) must not show up in either bucket.
    bridge_communities = {0: (0, {1})}
    communities = {0: {0, 1}, 1: {0}, 2: {1}, 3: {2}, 4: {2}}
    embeddings = torch.tensor(
        [
            [1.0, 0.0],  # node 0
            [1.0, 0.0],  # node 1 (primary)
            [0.9, 0.1],  # node 2 (secondary)
            [0.0, 1.0],  # node 3
            [0.0, 1.0],  # node 4
            [-1.0, 0.0],  # node 5 (negative partner)
        ]
    )
    split = SimpleNamespace(
        test_positive=np.array([[0, 0, 3], [1, 2, 4]]),
        test_negative=np.array([[0, 3], [5, 5]]),
    )

    metrics = _link_prediction_by_bridge_edges(
        embeddings, split, bridge_communities, communities
    )

    assert metrics["primary"]["count"] == 1
    assert metrics["secondary"]["count"] == 1
    # Both positive scores (1.0 and 0.9) outrank both negative scores (-1.0 and 0.0).
    assert metrics["primary"]["auc"] == 1.0
    assert metrics["secondary"]["auc"] == 1.0


def test_link_prediction_by_bridge_edges_counts_both_bridge_endpoints() -> None:
    """An edge between two bridge nodes can be classified from each endpoint."""
    # Node 0: primary community 0, secondary community 1. Node 1 is in community 0 (and
    # 5), so from node 0's side the edge is a "primary" instance.
    # Node 1: primary community 5, secondary community 0. Node 0 is in community 0 (and
    # 1), so from node 1's side the same edge is a "secondary" instance.
    bridge_communities = {0: (0, {1}), 1: (5, {0})}
    communities = {0: {0, 1}, 1: {0, 5}}
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    split = SimpleNamespace(
        test_positive=np.array([[0], [1]]),
        test_negative=np.array([[0], [2]]),
    )

    metrics = _link_prediction_by_bridge_edges(
        embeddings, split, bridge_communities, communities
    )

    assert metrics["primary"]["count"] == 1
    assert metrics["secondary"]["count"] == 1


def test_link_prediction_by_bridge_edges_reports_nan_for_an_empty_category() -> None:
    """A category with no qualifying test edges reports nan rather than crashing."""
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    split = SimpleNamespace(
        test_positive=np.empty((2, 0), dtype=np.int64),
        test_negative=np.array([[0], [1]]),
    )

    metrics = _link_prediction_by_bridge_edges(embeddings, split, {}, {})

    assert math.isnan(metrics["primary"]["auc"])
    assert math.isnan(metrics["primary"]["ap"])
    assert metrics["primary"]["count"] == 0
    assert metrics["secondary"]["count"] == 0


def test_link_prediction_by_community_count_buckets_by_endpoint_count() -> None:
    """Each test edge contributes to each endpoint's own community-count bucket."""
    # Node 0: 1 community. Node 1: 2 communities. Node 2: 0 (absent from `communities`).
    communities = {0: {0}, 1: {0, 1}}
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    split = SimpleNamespace(
        test_positive=np.array([[0], [1]]),  # edge (0, 1): bucket "1" and bucket "2"
        test_negative=np.array([[0], [2]]),
    )

    metrics = _link_prediction_by_community_count(embeddings, split, communities)

    assert metrics["1"]["count"] == 1
    assert metrics["2"]["count"] == 1
    assert metrics["0"]["count"] == 0  # node 2 has no test-positive edges
    assert metrics["3"]["count"] == 0


def test_link_prediction_by_community_count_pools_the_tail_into_max_count_plus() -> (
    None
):
    """Counts at or above max_count are pooled into one "<max_count>+" bucket."""
    communities = {0: {0, 1, 2}, 1: {0, 1, 2, 3, 4}}  # counts 3 and 5
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    split = SimpleNamespace(
        test_positive=np.array([[0], [1]]),
        test_negative=np.array([[0], [2]]),
    )

    metrics = _link_prediction_by_community_count(
        embeddings, split, communities, max_count=3
    )

    assert set(metrics) == {"0", "1", "2", "3+"}
    assert metrics["3+"]["count"] == 2  # both endpoints (counts 3 and 5) land here


def test_link_prediction_by_community_count_reports_nan_for_an_empty_bucket() -> None:
    """A bucket with no qualifying test edges reports nan rather than crashing."""
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    split = SimpleNamespace(
        test_positive=np.empty((2, 0), dtype=np.int64),
        test_negative=np.array([[0], [1]]),
    )

    metrics = _link_prediction_by_community_count(embeddings, split, {})

    assert math.isnan(metrics["0"]["auc"])
    assert math.isnan(metrics["0"]["ap"])
    assert metrics["0"]["count"] == 0


def _normalized(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True)


def test_posterior_stats_power_spherical_entropy_matches_direct_computation() -> None:
    """Regression test: a previous version wrapped posterior in
    Independent(posterior, 1) before calling entropy(), which reinterprets the
    per-node batch dimension into the event and collapses entropy() to a 0-d scalar,
    crashing on entropy[mask] ("too many indices for tensor of dimension 0"). Power
    Spherical has a real entropy() (unlike TNBBetaSpherical), so it hit this directly.
    """
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    posterior = PowerSpherical(mean_direction, kappa)
    mask = torch.tensor([True, True, False, False, False])

    stats = _posterior_stats(posterior, mask, {"family": "power_spherical"})

    expected = posterior.entropy()[mask].mean().item()
    assert math.isfinite(stats["entropy_mean"])
    assert stats["entropy_mean"] == pytest.approx(expected)
    assert stats["num_nodes"] == 2


def test_posterior_stats_von_mises_fisher_entropy_matches_direct_computation() -> None:
    """Same regression as above, for vMF -- it also implements entropy() directly."""
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.tensor([False, True, True, False, False])

    stats = _posterior_stats(posterior, mask, {"family": "vmf"})

    expected = posterior.entropy()[mask].mean().item()
    assert math.isfinite(stats["entropy_mean"])
    assert stats["entropy_mean"] == pytest.approx(expected)
    assert stats["num_nodes"] == 2


def test_posterior_stats_von_mises_fisher_r_bar_does_not_crash() -> None:
    """Regression test: VonMisesFisher.rsample only accepts torch.Size or a bare int
    (unlike TNBBetaSpherical/PowerSpherical, which accept a plain tuple too) -- a
    previous version called rsample((100,)), a plain tuple, which VonMisesFisher
    mishandles as a single non-int "size" and raises TypeError at torch.Size
    construction, uncaught by _posterior_stats's (AttributeError, RuntimeError)
    except clause."""
    mean_direction = _normalized(torch.randn(5, 3))
    kappa = torch.full((5, 1), 3.0)
    posterior = VonMisesFisher(mean_direction, kappa)
    mask = torch.ones(5, dtype=torch.bool)

    stats = _posterior_stats(posterior, mask, {"family": "vmf"})

    assert math.isfinite(stats["r_bar"])
    assert stats["r_bar"] >= 0.0


def test_posterior_stats_tnbbeta_spherical_entropy_is_nan_not_a_crash() -> None:
    """TNBBetaSpherical has no closed-form entropy -- confirms this stays a graceful
    NaN (the except branch) after the fix, not a regression."""
    mean_direction = _normalized(torch.randn(5, 3))
    p = torch.full((5,), 0.7)
    q = torch.full((5,), 0.1)
    epsilon = torch.full((5,), 1.0)
    posterior = TNBBetaSpherical(mean_direction, p, q, epsilon)
    mask = torch.ones(5, dtype=torch.bool)

    stats = _posterior_stats(posterior, mask, {"family": "tnbbeta"})

    assert math.isnan(stats["entropy_mean"])
