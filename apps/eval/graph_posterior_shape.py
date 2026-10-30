"""Dataset-agnostic posterior-shape summary for a trained GraphVAE checkpoint.

Usage:
    uv run python -m apps.eval.graph_posterior_shape --run-name NAME \
        --dataset cora|citeseer|pubmed|dblp|amazon|mag_cs|mag_eng|mag_chem|mag_med \
        [--checkpoint final] [--device cpu] [--ignore-features]

``--ignore-features`` rebuilds the batch with an identity feature matrix instead of
``--dataset``'s real features -- for a checkpoint trained with
``apps.link_prediction.main --ignore-features``, which needs the identical kind of
batch (identity features) at eval time to match what it actually saw during training.
``--dataset`` still names the real graph topology to load (e.g. ``cora``); only
``.features`` is overridden. ``--run-name`` is unaffected: it's whatever the training
run was named (e.g. a run trained with ``--ignore-features --run-name
cora_nofeat_tnbbeta`` produces checkpoints at ``cora_nofeat_tnbbeta_seed<N>``), so this
script doesn't derive it -- it just needs this flag to reconstruct the matching batch.

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

:func:`graph_to_batch` -- the "build a ``GraphBatch`` from a ``Graph`` object" step --
is factored out as a public function (not just this module's dataset dispatch) so
other scripts that already have a ``Graph`` in hand, e.g.
``apps/synthetic/sbm_recovery.py``'s synthetic stochastic-block-model graph, can reuse
it directly instead of duplicating the sparse/dense-features handling a third time.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

import torch

from tnbbeta_vae.data.mag_coauthor import MAG_COAUTHOR_DATASETS, load_mag_coauthor
from tnbbeta_vae.data.planetoid import (
    PLANETOID_DATASETS,
    load_planetoid,
    normalized_adjacency,
)
from tnbbeta_vae.data.planetoid import Graph as PlanetoidGraph
from tnbbeta_vae.data.snap_community import (
    SNAP_COMMUNITY_DATASETS,
    identity_features,
    load_snap_community,
)
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.models import GraphBatch
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

__all__ = ["graph_to_batch", "main"]

_DATASETS = (
    *PLANETOID_DATASETS,
    *SNAP_COMMUNITY_DATASETS,
    *(f"mag_{name}" for name in MAG_COAUTHOR_DATASETS),
)


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
    parser.add_argument(
        "--ignore-features",
        action="store_true",
        help="Rebuild the batch with identity features instead of --dataset's real "
        "features, matching a checkpoint trained with "
        "apps.link_prediction.main --ignore-features.",
    )
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    batch = _load_batch(args.dataset, device, ignore_features=args.ignore_features)

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


def graph_to_batch(
    graph: PlanetoidGraph | SnapGraph, device: torch.device
) -> GraphBatch:
    """Builds the whole-graph :class:`GraphBatch` for a forward pass (no split).

    No train/val/test split is applied -- this only needs a forward pass through the
    encoder, not link prediction (``positive_edges`` is left empty, since
    ``posterior_and_prior`` never reads it). Handles both scipy.sparse features
    (Planetoid, densified for ``GraphVAE.posterior_and_prior``'s dense-features
    branch) and torch.sparse identity features (SNAP community graphs and the
    synthetic stochastic-block-model graph, kept sparse), exactly as
    ``apps/link_prediction/main.py``'s ``run_once`` and this module's own
    :func:`_load_batch` dispatch on the same duck-typed distinction.

    Args:
        graph: Any ``Graph``-shaped object with ``.adjacency`` (scipy.sparse) and
            ``.features`` (scipy.sparse or torch tensor) -- Planetoid, SNAP community,
            or a synthetic graph built the same way (e.g.
            ``tnbbeta_vae.data.stochastic_block_model.stochastic_block_model``).
        device: Device to move the batch to.

    Returns:
        A :class:`GraphBatch` with an empty ``positive_edges``.
    """
    features = graph.features
    if hasattr(features, "tocoo"):
        # scipy.sparse matrix (Planetoid): convert to dense for backward compatibility
        # with GraphVAE.posterior_and_prior's dense-features branch.
        features_tensor = torch.as_tensor(
            cast("Any", features).toarray(), dtype=torch.float32
        )
    else:
        # Already a torch tensor (torch.sparse identity features, SNAP/synthetic
        # graphs).
        features_tensor = cast("torch.Tensor", features).to(dtype=torch.float32)

    return GraphBatch(
        features=features_tensor,
        norm_adjacency=normalized_adjacency(graph.adjacency),
        # posterior_and_prior never reads positive_edges -- only link prediction's
        # training/scoring path does, which this script doesn't run.
        positive_edges=torch.zeros((2, 0), dtype=torch.long),
    ).to(device)


def _load_batch(
    dataset: str, device: torch.device, *, ignore_features: bool = False
) -> GraphBatch:
    """Loads a named dataset (Planetoid, SNAP community or MAG co-authorship) as a
    :class:`GraphBatch`.

    Args:
        dataset: The real graph topology to load, e.g. ``"cora"``.
        device: Device to move the batch to.
        ignore_features: If true, overrides the loaded graph's ``.features`` with an
            identity matrix before building the batch -- mirroring
            ``apps/link_prediction/main.py --ignore-features`` exactly, so a
            checkpoint trained that way gets the same kind of batch at eval time.
    """
    if dataset in PLANETOID_DATASETS:
        graph: PlanetoidGraph | SnapGraph = load_planetoid(
            data_dir() / "planetoid", dataset
        )
    elif dataset.startswith("mag_"):
        graph = load_mag_coauthor(
            data_dir() / "mag_coauthor", dataset.removeprefix("mag_")
        )
    else:
        graph = load_snap_community(data_dir() / "snap_community", dataset)
    if ignore_features:
        graph.features = identity_features(graph.features.shape[0])
    return graph_to_batch(graph, device)


if __name__ == "__main__":
    main(sys.argv[1:])
