"""Decomposes a GraphVAE checkpoint's link-prediction AUC/AP by node bucket.

Usage:
    uv run python -m apps.eval.bucketed_link_prediction --dataset mag_cs \
        --run-name fixedm_cora_m-8 --seed 0 [--ignore-features] \
        [--num-degree-buckets 5]

Rebuilds the exact edge split ``apps.link_prediction.main.run_once`` trained on (the
split is a deterministic function of ``--seed``), runs the encoder over the *training*
adjacency (what the model saw), and scores held-out edges within buckets of node
training-degree and, where ground-truth communities exist (``mag_*``), of the number of
distinct communities among a node's training neighbors. A held-out edge counts toward
the bucket of both endpoints (see ``tnbbeta_vae.data.node_buckets``). For TNBBeta, each
bucket also gets its posterior ``p``/``q``/``epsilon``/``m`` summary.

Reads ``$TNBBETA_CHECKPOINT_DIR/<run-name>_seed<seed>/final.pt`` and writes
``bucketed_link_prediction.json`` beside it. The ``overall`` entry is the unbucketed
test AUC/AP and should equal the run's own reported test metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

import numpy as np
import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from apps.link_prediction.main import score_edges
from tnbbeta_vae.data.mag_coauthor import load_mag_coauthor, load_mag_communities
from tnbbeta_vae.data.neighbor_heterogeneity import neighbor_community_count
from tnbbeta_vae.data.node_buckets import bucketed_link_scores, quantile_buckets
from tnbbeta_vae.data.planetoid import (
    load_planetoid,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.data.snap_community import identity_features
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint, select_device

# Bucket k < len - 1 is "k + 1 communities"; the last bucket pools "this many or more".
_MAX_COMMUNITY_BUCKET = 4


def main(argv: list[str] | None = None) -> None:
    """Writes the bucketed decomposition for one checkpoint.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-features", action="store_true")
    parser.add_argument("--num-degree-buckets", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = select_device(args.device)
    graph, communities = _load(args.dataset, args.ignore_features)
    split = split_edges(graph.adjacency, seed=args.seed)

    run_dir = checkpoint_dir() / f"{args.run_name}_seed{args.seed}"
    model, checkpoint = load_model_checkpoint(run_dir / "final.pt", device)
    batch = graph_to_batch(graph, device)
    batch = dataclasses.replace(
        batch,
        norm_adjacency=normalized_adjacency(split.train_adjacency).to(device),
    )
    model.eval()
    with torch.no_grad():
        embeddings = model.embeddings(batch).cpu()  # type: ignore[attr-defined]
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]
    values = embeddings.numpy().astype(np.float64)

    auc, ap = score_edges(embeddings, split.test_positive, split.test_negative)
    results: dict[str, Any] = {
        "dataset": args.dataset,
        "run_name": args.run_name,
        "seed": args.seed,
        "overall": {"auc": auc, "ap": ap},
    }

    degree = np.asarray(split.train_adjacency.sum(axis=1)).ravel().astype(np.float64)
    degree_buckets = quantile_buckets(degree, args.num_degree_buckets)
    results["by_degree"] = _decompose(
        values, split, degree_buckets, posterior, checkpoint["config"], degree
    )

    if communities is not None:
        counts = neighbor_community_count(split.train_adjacency, communities)
        community_buckets = np.clip(counts - 1, -1, _MAX_COMMUNITY_BUCKET - 1)
        results["by_neighbor_communities"] = _decompose(
            values, split, community_buckets, posterior, checkpoint["config"], degree
        )
        # Community count within degree bins, to separate it from degree.
        results["by_neighbor_communities_within_degree"] = {
            str(bucket): _decompose(
                values,
                split,
                np.where(degree_buckets == bucket, community_buckets, -1),
                posterior,
                checkpoint["config"],
                degree,
            )
            for bucket in range(args.num_degree_buckets)
        }

    output = run_dir / "bucketed_link_prediction.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def _load(
    dataset: str, ignore_features: bool
) -> tuple[Any, dict[int, set[int]] | None]:
    """Loads the graph (and ground-truth communities, if the dataset has them)."""
    if dataset.startswith("mag_"):
        name = dataset.removeprefix("mag_")
        root = data_dir() / "mag_coauthor"
        graph: Any = load_mag_coauthor(root, name)
        communities: dict[int, set[int]] | None = load_mag_communities(root, name)
    else:
        graph = load_planetoid(data_dir() / "planetoid", dataset)
        communities = None
    if ignore_features:
        graph.features = identity_features(graph.features.shape[0])
    return graph, communities


def _decompose(
    values: np.ndarray,
    split: Any,
    buckets: np.ndarray,
    posterior: Any,
    config: dict[str, Any],
    degree: np.ndarray,
) -> dict[str, Any]:
    """Per-bucket link scores, node count, mean degree and posterior summary."""
    scores = bucketed_link_scores(
        values, split.test_positive, split.test_negative, buckets
    )
    decomposed: dict[str, Any] = {}
    for bucket, bucket_scores in scores.items():
        mask = torch.as_tensor(buckets == bucket)
        decomposed[str(bucket)] = {
            **bucket_scores,
            "num_nodes": int(mask.sum()),
            "mean_degree": float(degree[buckets == bucket].mean()),
            "posterior": posterior_stats(posterior, mask, config),
        }
    return decomposed


if __name__ == "__main__":
    main()
