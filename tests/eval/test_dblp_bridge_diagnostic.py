"""Tests for com-DBLP bridge-node diagnostic utilities."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from apps.eval.dblp_bridge_diagnostic import _link_prediction_by_bridge_edges


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
