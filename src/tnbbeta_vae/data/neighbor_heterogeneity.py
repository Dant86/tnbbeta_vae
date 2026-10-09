"""Per-node neighbor-heterogeneity: Shannon entropy of a node's neighbor communities.

The live mechanistic theory in
``writeup/results/graph_bimodality_investigation_summary_2026-10-08.md`` is that a
GCN's neighbor-averaging pulls a node's embedding toward a compromise position that
scores poorly against a *heterogeneous* neighborhood (neighbors spread across many
communities/directions), and that ``TNBBetaSpherical``'s bimodal capacity -- unlike a
unimodal vMF/Power Spherical cap -- lets training align with different neighbor
subsets on different stochastic draws instead of being stuck with that one
compromise. This module provides the one shared ingredient every probe of that
mechanism needs: a per-node measurement of exactly how angularly/community-mixed a
node's neighborhood actually is, independent of any model or checkpoint.

Both ``tnbbeta_vae.data.snap_community.load_snap_communities`` (after
``remap_communities``) and ``tnbbeta_vae.data.mag_coauthor.load_mag_communities``
already return the same ``node_id -> set[community_index]`` shape this module's
:func:`neighbor_heterogeneity` expects -- a node can belong to more than one
community at once (genuine overlap). :func:`labels_to_communities` adapts a
single-label source (``tnbbeta_vae.data.stochastic_block_model.sbm_community_labels``)
into that same shape, so the synthetic SBM graph feeds the identical function with no
special case.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["labels_to_communities", "neighbor_heterogeneity"]


def neighbor_heterogeneity(
    adjacency: Any, communities: dict[int, set[int]]
) -> npt.NDArray[np.float64]:
    """Computes each node's neighbor-heterogeneity as a Shannon entropy.

    For each node, builds the multiset of its neighbors' community memberships (a
    neighbor belonging to more than one community contributes a full count to each
    one it belongs to, not a fractional vote split across them), normalizes it into a
    probability distribution over communities, and returns that distribution's
    Shannon entropy in nats. A node with no neighbors at all, or none of whose
    neighbors carry any community label, gets ``NaN`` rather than a crash or a
    misleading ``0.0`` (which would otherwise be indistinguishable from "neighbors
    all agree on one community").

    A node's *own* community membership (if any) never affects its own
    heterogeneity -- only its neighbors' do.

    Args:
        adjacency: Symmetric graph adjacency, any scipy.sparse format (converted to
            CSR internally), shape ``(n, n)``.
        communities: Node ID -> set of community indices, in the same internal node-ID
            space as ``adjacency`` -- e.g.
            ``tnbbeta_vae.data.snap_community.remap_communities``'s or
            ``tnbbeta_vae.data.mag_coauthor.load_mag_communities``'s output, or
            :func:`labels_to_communities`'s for a single-label source. Nodes absent
            from this mapping are treated as carrying no community label.

    Returns:
        Array of shape ``(n,)``, dtype ``float64``: each node's neighbor-heterogeneity
        in nats, or ``NaN`` where it's undefined.
    """
    adjacency = adjacency.tocsr()
    num_nodes = adjacency.shape[0]
    heterogeneity = np.full(num_nodes, np.nan, dtype=np.float64)

    for node in range(num_nodes):
        start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
        neighbors = adjacency.indices[start:end]

        counts: dict[int, int] = defaultdict(int)
        for neighbor in neighbors:
            for community in communities.get(int(neighbor), ()):
                counts[community] += 1

        total = sum(counts.values())
        if total == 0:
            continue

        probabilities = np.array(list(counts.values()), dtype=np.float64) / total
        heterogeneity[node] = float(-np.sum(probabilities * np.log(probabilities)))

    return heterogeneity


def labels_to_communities(labels: npt.NDArray[np.int64]) -> dict[int, set[int]]:
    """Adapts a single-label array into the ``node_id -> set[community_index]`` shape.

    For a source like
    ``tnbbeta_vae.data.stochastic_block_model.sbm_community_labels``, where each node
    has exactly one community, this wraps each label in a singleton set so it feeds
    :func:`neighbor_heterogeneity` with no special case.

    Args:
        labels: Array of shape ``(n,)``: node ``i``'s single community index is
            ``labels[i]``.

    Returns:
        Node ID -> ``{labels[i]}``, for every node ``0..n-1``.
    """
    return {i: {int(label)} for i, label in enumerate(labels)}
