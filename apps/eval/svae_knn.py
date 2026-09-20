"""Latent-space k-NN classification on MNIST (S-VAE paper, Table 2).

Usage:
    uv run python -m apps.eval.svae_knn --run-name NAME [--checkpoint final] \
        [--label-counts 100 600 1000] [--num-subsets 20] [--k 5]

For each label count N, draws ``--num-subsets`` random sets of N labelled training
images, classifies every test image by a ``--k``-nearest-neighbour vote among them
in latent space, and records the accuracy. The latent of an image is the centre of
its posterior: the mean for the Gaussian (Euclidean distance) and the mean
direction for the sphere models (geodesic distance, i.e. arccos of the dot
product). TNBBeta's centre is its ``mode_direction`` (the mean direction, negated
when p < 0.5, which undoes the (mu, p) ~ (-mu, 1 - p) alias).

Writes ``svae_knn_<checkpoint>.json`` next to the checkpoint with, per N, the mean
(``acc_<N>``) and standard deviation (``acc_<N>_std``) over the subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint


def main(argv: list[str] | None = None) -> None:
    """Runs the k-NN experiment for one trained run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--label-counts", nargs="+", type=int, default=[100, 600, 1000])
    parser.add_argument("--num-subsets", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    geometry = "euclidean" if kind == "conv_gaussian_vae" else "geodesic"

    train_set = load_mnist(data_dir(), split="train")
    test_set = load_mnist(data_dir(), split="test")
    train_features = _encode(model, kind, train_set, args, device)
    test_features = _encode(model, kind, test_set, args, device)
    train_labels, test_labels = train_set.labels(), test_set.labels()

    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": kind,
        "geometry": geometry,
        "k": args.k,
        "num_subsets": args.num_subsets,
    }
    for count in args.label_counts:
        accuracies = []
        for _ in range(args.num_subsets):
            chosen = torch.as_tensor(
                rng.choice(len(train_features), count, replace=False)
            )
            accuracies.append(
                knn_accuracy(
                    train_features[chosen],
                    train_labels[chosen],
                    test_features,
                    test_labels,
                    args.k,
                    geometry,
                )
            )
        results[f"acc_{count}"] = float(np.mean(accuracies))
        results[f"acc_{count}_std"] = float(np.std(accuracies, ddof=1))

    output = run_dir / f"svae_knn_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def knn_accuracy(
    labelled_features: torch.Tensor,
    labelled_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    k: int,
    geometry: str,
) -> float:
    """Returns the accuracy of a k-nearest-neighbour vote.

    Args:
        labelled_features: Reference points, shape ``(n, d)``.
        labelled_labels: Their integer labels, shape ``(n,)``.
        query_features: Points to classify, shape ``(m, d)``.
        query_labels: Their true labels, shape ``(m,)``.
        k: Number of neighbours (capped at ``n``).
        geometry: ``"euclidean"``, or ``"geodesic"`` for unit vectors, where the
            distance is the arccos of the dot product.

    Returns:
        The fraction of queries whose neighbour vote matches ``query_labels``.
    """
    if geometry == "geodesic":
        distances = torch.arccos((query_features @ labelled_features.T).clamp(-1, 1))
    else:
        distances = torch.cdist(query_features, labelled_features)
    nearest = distances.topk(min(k, len(labelled_features)), largest=False).indices
    votes = torch.mode(labelled_labels[nearest], dim=1).values
    return (votes == query_labels).float().mean().item()


@torch.no_grad()
def _encode(
    model: Any,
    kind: str,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    parts = []
    for batch in loader:
        posterior, _ = model.posterior_and_prior(batch.to(device))
        parts.append(_centre(posterior, kind).cpu())
    return torch.cat(parts)


def _centre(posterior: Any, kind: str) -> torch.Tensor:
    """Returns the posterior's centre: the mean, or the mode direction on a sphere."""
    if kind == "conv_gaussian_vae":
        return posterior.base_dist.loc
    if kind == "conv_vmf_vae":
        return posterior.loc
    direction = posterior.mean_direction
    return torch.where((posterior.p > 0.5)[:, None], direction, -direction)


if __name__ == "__main__":
    main(sys.argv[1:])
