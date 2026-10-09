"""Evaluates TNBBeta's expressivity advantage on com-DBLP bridge nodes.

Usage:
    uv run python -m apps.eval.dblp_bridge_diagnostic --run-name NAME \
        [--checkpoint final] [--device cpu]

com-DBLP is a co-authorship network where nodes (authors) can belong to multiple
publication venues (communities). Nodes in multiple communities are "bridge" nodes —
genuine multi-community members with documented ground truth.

This diagnostic tests whether TNBBeta's extra capacity for bimodal/multi-peaked
posteriors (unavailable to vMF or Power Spherical) provides an advantage specifically
for these bridge nodes. We test two hypotheses:

1. **Shape hypothesis**: Bridge nodes have posteriors with higher entropy or bimodal
   structure compared to single-community nodes. Only TNBBeta can express this; vMF
   and Power Spherical are structurally limited to unimodal shapes.

2. **Link prediction hypothesis**: Link-prediction accuracy on a bridge node's edges
   to its *secondary* communities (non-primary) improves for TNBBeta compared to
   baselines, while aggregate AUC/AP stay tied (same pattern as other datasets).

3. **Dose-response hypothesis**: if (2) is really about representing genuine multi-
   community membership rather than some other difference between the families, the
   TNBBeta-vs-baseline AUC/AP gap should grow with a node's community count (1, 2, 3,
   ...), not just step once at the bridge/non-bridge threshold -- a sharper, more
   falsifiable version of (2) (``_link_prediction_by_community_count``).

**Design decisions:**

- **Primary vs. secondary community**: For each bridge node, ``snap_community.py``'s
  ``primary_secondary_communities`` counts, for each community the node belongs to, how
  many of the node's *graph* neighbors (from the full, un-split adjacency -- this is
  ground-truth structure, not something being predicted) are themselves members of that
  community. The community with the highest such count is "primary"; every other
  community the node belongs to is "secondary". Link-prediction accuracy is then
  measured separately on test edges classified against each bridge node's primary vs.
  secondary communities, to isolate the effect.

- **Posterior statistics**: For TNBBeta, report (p, q, epsilon) marginals for bridge vs.
  non-bridge nodes. For vMF and Power Spherical, report concentration (equivalent kappa
  via r-bar, analogous to svae_concentration.py). Report posterior entropy as a
  model-agnostic bimodality proxy.

- **Node-ID space**: community membership and bridge nodes are loaded in the raw SNAP ID
  space and translated into the graph's internal ``0..n-1`` IDs via
  ``Graph.node_id_map`` (``snap_community.remap_communities``), not by assuming any
  particular order the loader happened to use.

- **Correctness caveat**: This diagnostic is hand-rolled for com-DBLP's specific
  structure. If a future dataset has different community semantics (not publication
  venues, not membership lists), the primary/secondary split and bridge-node definition
  may need to be adapted.

Writes ``bridge_diagnostic_<checkpoint>.json`` next to the checkpoint, containing
per-model statistics for bridge vs. non-bridge nodes and per-model link-prediction
accuracy broken down by edge type.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from tnbbeta_vae.data.planetoid import normalized_adjacency, split_edges
from tnbbeta_vae.data.snap_community import (
    bridge_nodes as identify_bridge_nodes,
)
from tnbbeta_vae.data.snap_community import (
    load_snap_communities,
    load_snap_community,
    primary_secondary_communities,
    remap_communities,
)
from tnbbeta_vae.models import GraphBatch
from tnbbeta_vae.models.heads import posterior_centre
from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint


def main(argv: list[str] | None = None) -> None:
    """Computes and writes the bridge-node diagnostic for one run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)

    # Load graph and communities.
    graph = load_snap_community(data_dir() / "snap_community", "dblp")
    communities = load_snap_communities(data_dir() / "snap_community", "dblp")
    split = split_edges(graph.adjacency, seed=0)

    # Communities and bridge nodes are loaded in the raw SNAP ID space; translate them
    # into the graph's internal 0..n-1 IDs via the loader's own node_id_map, rather than
    # assuming any particular order.
    remapped_communities = remap_communities(graph.node_id_map, communities)
    remapped_bridges = identify_bridge_nodes(remapped_communities, min_multiplicity=2)
    bridge_communities = primary_secondary_communities(
        graph.adjacency, remapped_communities, remapped_bridges
    )

    # Build batch.
    features = graph.features
    if hasattr(features, "tocoo"):
        features = torch.as_tensor(features.toarray(), dtype=torch.float32)
    elif isinstance(features, torch.Tensor):
        features = features.to(dtype=torch.float32)

    # Sparse upper-triangle extraction, not .toarray() -- densifying the whole
    # adjacency (317,080 x 317,080 for com-DBLP) would try to allocate ~375GiB.
    upper_triangle: Any = sp.triu(split.train_adjacency, k=1).tocoo()
    upper = np.stack([upper_triangle.row, upper_triangle.col])
    batch = GraphBatch(
        features=features,
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(upper, dtype=torch.long),
    ).to(device)

    # Compute posteriors and embeddings.
    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore
        embeddings = posterior_centre(
            checkpoint["config"].get("family", "tnbbeta"),
            posterior,  # type: ignore
        ).cpu()

    # Compute statistics for bridge vs. non-bridge nodes.
    bridge_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    bridge_mask[list(remapped_bridges)] = True

    bridge_stats = posterior_stats(posterior, bridge_mask, checkpoint["config"])
    non_bridge_stats = posterior_stats(posterior, ~bridge_mask, checkpoint["config"])

    # Compute link-prediction metrics per edge type.
    link_metrics = _link_prediction_by_bridge_edges(
        embeddings, split, bridge_communities, remapped_communities
    )
    link_metrics_by_count = _link_prediction_by_community_count(
        embeddings, split, remapped_communities
    )

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "dataset": "dblp",
        "checkpoint": args.checkpoint,
        "bridge_nodes": {
            "count": len(remapped_bridges),
            "posterior_stats": bridge_stats,
        },
        "non_bridge_nodes": {
            "count": batch.num_nodes - len(remapped_bridges),
            "posterior_stats": non_bridge_stats,
        },
        "link_prediction_metrics": link_metrics,
        "link_prediction_by_community_count": link_metrics_by_count,
    }

    output = run_dir / f"bridge_diagnostic_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def _link_prediction_by_bridge_edges(
    embeddings: torch.Tensor,
    split: Any,
    bridge_communities: dict[int, tuple[int, set[int]]],
    communities: dict[int, set[int]],
) -> dict[str, Any]:
    """Computes link-prediction metrics broken down by edge type.

    For a bridge node ``b`` with primary community ``p`` and secondary communities
    ``s`` (``bridge_communities[b]``, from
    ``snap_community.primary_secondary_communities``), a test edge ``(b, other)`` is a
    "primary" instance if ``other`` is a member of ``p``, and a "secondary" instance if
    ``other`` is a member of some community in ``s`` (but not ``p``). Both endpoints of
    an edge are checked independently, so an edge between two bridge nodes can
    contribute an instance to each category (once per bridge endpoint's perspective).
    Every instance in a category is scored against the same pool of sampled test
    non-edges (``split.test_negative``), the same background used for the aggregate
    AUC/AP computed elsewhere (e.g. ``apps/link_prediction/main.py``).

    Args:
        embeddings: Node embeddings, shape (num_nodes, latent_dim).
        split: LinkSplit with val and test edges.
        bridge_communities: Bridge node ID -> (primary community, secondary
            communities), from ``snap_community.primary_secondary_communities``.
        communities: Remapped node ID -> set of community indices, covering every node
            (bridge or not) that belongs to at least one community.

    Returns:
        Dict with AUC/AP/count broken down by edge type ("primary"/"secondary").
    """
    primary_edges: list[tuple[int, int]] = []
    secondary_edges: list[tuple[int, int]] = []

    for raw_u, raw_v in split.test_positive.T:
        u, v = int(raw_u), int(raw_v)
        for bridge, other in ((u, v), (v, u)):
            classification = bridge_communities.get(bridge)
            if classification is None:
                continue
            primary, secondary = classification
            other_communities = communities.get(other, set())
            if primary in other_communities:
                primary_edges.append((u, v))
            elif other_communities & secondary:
                secondary_edges.append((u, v))

    values = embeddings.numpy()
    neg_scores = (values[split.test_negative[0]] * values[split.test_negative[1]]).sum(
        -1
    )

    metrics: dict[str, Any] = {}
    for edge_type, edges in (
        ("primary", primary_edges),
        ("secondary", secondary_edges),
    ):
        if not edges:
            metrics[edge_type] = {"auc": float("nan"), "ap": float("nan"), "count": 0}
            continue
        pos_edges = np.array(edges).T
        pos_scores = (values[pos_edges[0]] * values[pos_edges[1]]).sum(-1)
        metrics[edge_type] = {
            "auc": float(roc_auc(pos_scores, neg_scores)),
            "ap": float(average_precision(pos_scores, neg_scores)),
            "count": len(pos_scores),
        }
    return metrics


def _link_prediction_by_community_count(
    embeddings: torch.Tensor,
    split: Any,
    communities: dict[int, set[int]],
    max_count: int = 5,
) -> dict[str, Any]:
    """Computes link-prediction metrics broken down by a node's community count.

    A more direct test of the expressivity hypothesis than the binary bridge/non-
    bridge split in :func:`_link_prediction_by_bridge_edges`: if TNBBeta's edge is
    really about representing genuine multi-community membership, the TNBBeta-vs-
    baseline gap should grow with community count (a dose-response pattern), not just
    step once at some threshold.

    For each test edge, each endpoint contributes an instance to its own community-
    count bucket -- so, as in :func:`_link_prediction_by_bridge_edges`, an edge between
    two nodes with different counts contributes to both buckets, once per endpoint's
    perspective. A node absent from ``communities`` has zero recorded communities.
    Counts at or above ``max_count`` are pooled into one ``"<max_count>+"`` bucket,
    since community count is expected to be long-tailed and a per-exact-count
    breakdown would get noisy fast at the tail; every bucket's edge count is reported
    alongside its AUC/AP for exactly this reason -- a thin bucket should be visibly
    thin, not silently misleading.

    Args:
        embeddings: Node embeddings, shape (num_nodes, latent_dim).
        split: LinkSplit with val and test edges.
        communities: Remapped node ID -> set of community indices, covering every node
            (bridge or not) that belongs to at least one community.
        max_count: Community counts at or above this are pooled into one bucket.

    Returns:
        Dict keyed by bucket label (``"0"``, ``"1"``, ..., ``"<max_count>+"``), each
        with AUC/AP/count.
    """

    def bucket(node: int) -> str:
        count = len(communities.get(node, set()))
        return f"{max_count}+" if count >= max_count else str(count)

    edges_by_bucket: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for raw_u, raw_v in split.test_positive.T:
        u, v = int(raw_u), int(raw_v)
        for node in (u, v):
            edges_by_bucket[bucket(node)].append((u, v))

    values = embeddings.numpy()
    neg_scores = (values[split.test_negative[0]] * values[split.test_negative[1]]).sum(
        -1
    )

    labels = [str(count) for count in range(max_count)] + [f"{max_count}+"]
    metrics: dict[str, Any] = {}
    for label in labels:
        edges = edges_by_bucket.get(label, [])
        if not edges:
            metrics[label] = {"auc": float("nan"), "ap": float("nan"), "count": 0}
            continue
        pos_edges = np.array(edges).T
        pos_scores = (values[pos_edges[0]] * values[pos_edges[1]]).sum(-1)
        metrics[label] = {
            "auc": float(roc_auc(pos_scores, neg_scores)),
            "ap": float(average_precision(pos_scores, neg_scores)),
            "count": len(pos_scores),
        }
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
