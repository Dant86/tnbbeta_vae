"""Dataset-agnostic posterior-shape summary for a trained GraphVAE checkpoint.

Usage:
    uv run python -m apps.eval.graph_posterior_shape --run-name NAME \
        --dataset cora|citeseer|pubmed|dblp|amazon [--checkpoint final] [--device cpu]

``apps.eval.dblp_bridge_diagnostic``'s corrected shape diagnostic found that *every*
com-DBLP node, bridge or not, sits in TNBBeta's proven bimodal regime (``m =
epsilon - (latent_dim - 1) / 2 < 0``) -- not just the multi-community nodes the
original hypothesis singled out. That rules out "bimodality encodes which bridge
nodes are ambiguous" as the mechanism, but leaves open whether it's specific to
com-DBLP (its overlapping-community structure) or a property of the training setup
itself (no reconstruction decoder, no real node features, a huge pairwise
dot-product/ranking objective) that would show up on any dataset trained the same
way -- including ones with no community structure at all, like Planetoid.

This script reuses ``dblp_bridge_diagnostic``'s posterior-shape computation
(factored out to ``tnbbeta_vae.models.posterior_stats`` for exactly this reason) over
every node in the graph, with no bridge/non-bridge or community split -- so it also
runs on Cora/Citeseer/Pubmed, which have no community labels to split on.

Writes ``posterior_shape_<checkpoint>.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

import torch

from tnbbeta_vae.data.planetoid import (
    PLANETOID_DATASETS,
    load_planetoid,
    normalized_adjacency,
)
from tnbbeta_vae.data.snap_community import SNAP_COMMUNITY_DATASETS, load_snap_community
from tnbbeta_vae.models import GraphBatch
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

_DATASETS = (*PLANETOID_DATASETS, *SNAP_COMMUNITY_DATASETS)


def main(argv: list[str] | None = None) -> None:
    """Computes and writes the whole-graph posterior-shape summary for one run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True, choices=list(_DATASETS))
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    batch = _load_batch(args.dataset, device)

    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]

    all_nodes = torch.ones(batch.num_nodes, dtype=torch.bool)
    stats = posterior_stats(posterior, all_nodes, checkpoint["config"])

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "model_name": checkpoint["model_name"],
        "num_nodes": batch.num_nodes,
        "posterior_stats": stats,
    }

    output = run_dir / f"posterior_shape_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def _load_batch(dataset: str, device: torch.device) -> GraphBatch:
    """Loads the whole graph (no train/val/test split -- this only needs a forward
    pass through the encoder, not link prediction) as a :class:`GraphBatch`."""
    if dataset in PLANETOID_DATASETS:
        graph = load_planetoid(data_dir() / "planetoid", dataset)
    else:
        graph = load_snap_community(data_dir() / "snap_community", dataset)

    features = graph.features
    if hasattr(features, "tocoo"):
        # scipy.sparse matrix (Planetoid): convert to dense for backward compatibility
        # with GraphVAE.posterior_and_prior's dense-features branch.
        features_tensor = torch.as_tensor(
            cast("Any", features).toarray(), dtype=torch.float32
        )
    else:
        # Already a torch tensor (torch.sparse identity features, SNAP graphs).
        features_tensor = cast("torch.Tensor", features).to(dtype=torch.float32)

    return GraphBatch(
        features=features_tensor,
        norm_adjacency=normalized_adjacency(graph.adjacency),
        # posterior_and_prior never reads positive_edges -- only link prediction's
        # training/scoring path does, which this script doesn't run.
        positive_edges=torch.zeros((2, 0), dtype=torch.long),
    ).to(device)


if __name__ == "__main__":
    main(sys.argv[1:])
