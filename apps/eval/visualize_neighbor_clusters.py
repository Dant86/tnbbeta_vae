"""Qualitative check: do a heterogeneous node's neighbors' posteriors visibly split?

Usage:
    uv run python -m apps.eval.visualize_neighbor_clusters --run-name NAME \
        [--checkpoint final] [--device cpu] \
        [--num-communities 10] [--nodes-per-community 50] \
        [--p-in 0.3] [--p-out 0.05] [--graph-seed 0] [--feature-noise-std NONE] \
        [--num-high 3] [--num-low 3] [--min-degree 4]

The live mechanistic theory
(``writeup/results/graph_bimodality_investigation_summary_2026-10-08.md``, section
5) is that a node with angularly/community-heterogeneous neighbors gives a GCN
encoder no single good compromise direction, which is exactly where TNBBeta's
bimodal capacity would show up as two separated neighbor clusters rather than one.
This script picks a trained checkpoint's highest- and lowest-
``tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`` nodes (degree-
filtered via ``--min-degree`` so there are enough neighbors to plot), PCA-projects
each selected node's *neighbors'* posterior centres (``posterior_centre``, which is
defined for every family -- this diagnostic isn't TNBBeta-specific) onto 2
components, and plots a high-heterogeneity node's neighbor scatter next to a
low-heterogeneity node's. The hypothesis: the high-heterogeneity scatter visibly
splits into 2+ clusters, the low-heterogeneity one stays a single tight cluster.

This is illustrative, not a statistical test -- no assertion is made here (or should
be made from this script's output) about exact visual clustering; see the dated
write-up for an honest reading of what one real run's figure actually shows,
including if it does not show a clean split.

Rebuilds the exact synthetic stochastic-block-model graph a checkpoint was trained
on from the same CLI parameters ``apps/synthetic/sbm_recovery.py`` took (matching
``--p-out``/``--feature-noise-std``/etc. to the training run is the caller's
responsibility, exactly as already required to reproduce any of that script's
other diagnostics against one of its checkpoints).

Writes ``neighbor_clusters_<checkpoint>.json`` (the selected nodes, their degree and
heterogeneity) and ``neighbor_clusters_<checkpoint>.html`` (the plotly figure) next
to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from tnbbeta_vae.data.neighbor_heterogeneity import (
    labels_to_communities,
    neighbor_heterogeneity,
)
from tnbbeta_vae.data.stochastic_block_model import (
    sbm_community_labels,
    stochastic_block_model,
)
from tnbbeta_vae.models.heads import posterior_centre
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME
from tnbbeta_vae.training import load_model_checkpoint

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Builds and writes the neighbor-cluster figure for one checkpoint.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        ValueError: If ``--min-degree`` excludes every node, leaving nothing to plot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-communities", type=int, default=10)
    parser.add_argument("--nodes-per-community", type=int, default=50)
    parser.add_argument("--p-in", type=float, default=0.3)
    parser.add_argument("--p-out", type=float, default=0.05)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--feature-noise-std", type=float, default=None)
    parser.add_argument("--num-high", type=int, default=3)
    parser.add_argument("--num-low", type=int, default=3)
    parser.add_argument(
        "--min-degree",
        type=int,
        default=4,
        help="Nodes with fewer neighbors than this are excluded -- too few points "
        "to meaningfully PCA-project and plot.",
    )
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    graph = stochastic_block_model(
        num_communities=args.num_communities,
        nodes_per_community=args.nodes_per_community,
        p_in=args.p_in,
        p_out=args.p_out,
        seed=args.graph_seed,
        feature_noise_std=args.feature_noise_std,
    )
    communities = labels_to_communities(
        sbm_community_labels(args.num_communities, args.nodes_per_community)
    )
    heterogeneity = neighbor_heterogeneity(graph.adjacency, communities)
    degrees = np.asarray(graph.adjacency.sum(axis=1)).flatten()

    eligible = np.where(np.isfinite(heterogeneity) & (degrees >= args.min_degree))[0]
    if eligible.size == 0:
        raise ValueError(
            f"--min-degree={args.min_degree} excludes every node (max degree "
            f"{int(degrees.max())}); lower it to select nodes to plot."
        )

    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    batch = graph_to_batch(graph, device)
    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]
    family = checkpoint["config"]["family"]
    centres = posterior_centre(family, posterior).detach().cpu().numpy()

    order = eligible[np.argsort(heterogeneity[eligible])]
    low_nodes = order[: args.num_low]
    high_nodes = order[-args.num_high :][::-1]

    adjacency = graph.adjacency.tocsr()
    figure = _figure(adjacency, centres, heterogeneity, degrees, high_nodes, low_nodes)
    html = run_dir / f"neighbor_clusters_{args.checkpoint}.html"
    figure.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "high_heterogeneity_nodes": _node_summaries(high_nodes, heterogeneity, degrees),
        "low_heterogeneity_nodes": _node_summaries(low_nodes, heterogeneity, degrees),
    }
    output = run_dir / f"neighbor_clusters_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def _figure(
    adjacency: Any,
    centres: npt.NDArray[np.float64],
    heterogeneity: npt.NDArray[np.float64],
    degrees: npt.NDArray[np.float64],
    high_nodes: npt.NDArray[np.intp],
    low_nodes: npt.NDArray[np.intp],
) -> go.Figure:
    """A 2-row grid: each selected node's neighbors' PCA-projected posterior centres.

    Row 1 is the highest-heterogeneity nodes, row 2 the lowest; one subplot per
    selected node.
    """
    num_cols = max(len(high_nodes), len(low_nodes), 1)
    titles = []
    for row_nodes in (high_nodes, low_nodes):
        titles.extend(
            [
                f"node {node} (H={heterogeneity[node]:.2f}, deg={int(degrees[node])})"
                for node in row_nodes
            ]
            + [""] * (num_cols - len(row_nodes))
        )

    figure = make_subplots(
        rows=2, cols=num_cols, subplot_titles=titles, row_titles=["High H", "Low H"]
    )
    for row, nodes in enumerate((high_nodes, low_nodes), start=1):
        for col, node in enumerate(nodes, start=1):
            points = _neighbor_centres_2d(adjacency, centres, int(node))
            figure.add_trace(
                go.Scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    mode="markers",
                    marker={"size": 6, "opacity": 0.7},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    figure.update_layout(
        template=TEMPLATE_NAME,
        title="Neighbor posterior-centre scatter: high- vs low-heterogeneity nodes",
        height=350 * 2,
    )
    return figure


def _neighbor_centres_2d(
    adjacency: Any, centres: npt.NDArray[np.float64], node: int
) -> npt.NDArray[np.float64]:
    """PCA-projects one node's neighbors' posterior centres onto 2 components."""
    start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
    neighbors = adjacency.indices[start:end]
    return _pca_2d(centres[neighbors])


def _pca_2d(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Projects ``points`` (shape ``(n, d)``) onto their top-2 principal components.

    Degenerate cases (fewer than 2 points, or fewer than 2 nonzero singular
    directions) are padded with zero columns rather than raising, since this is a
    qualitative/illustrative plot, not a statistical computation that needs to fail
    loudly on too little data.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(2, vt.shape[0])]
    projected = centered @ components.T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected


def _node_summaries(
    nodes: npt.NDArray[np.intp],
    heterogeneity: npt.NDArray[np.float64],
    degrees: npt.NDArray[np.float64],
) -> list[dict[str, float]]:
    """Builds the JSON-serializable per-node summary for the selected node list."""
    return [
        {
            "node": int(node),
            "heterogeneity": float(heterogeneity[node]),
            "degree": int(degrees[node]),
        }
        for node in nodes
    ]


if __name__ == "__main__":
    main(sys.argv[1:])
