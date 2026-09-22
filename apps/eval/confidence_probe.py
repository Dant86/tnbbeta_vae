"""Class-probe accuracy from a posterior's confidence scalar alone (not its direction).

Usage:
    uv run python -m apps.eval.confidence_probe --run-name NAME [--checkpoint final] \
        [--label-counts 100 600 1000] [--num-subsets 20] [--k 5]

The Gaussian's per-example mean standard deviation, vMF's kappa and TNBBeta's p are each
a single scalar carrying no directional information. Running the same k-NN probe as
``apps.eval.svae_knn`` on that scalar alone tests whether a family routes class
information through it, separately from the mean direction: vMF's kappa reflects only
"how confident", so it should carry little class signal on its own; if TNBBeta's p does
carry class signal, that is a channel vMF has no analogue of (its posterior is always a
cap at the mean direction; p instead sets where a TNBBeta posterior's cap or ring sits).

Writes ``confidence_probe_<checkpoint>.json`` next to the checkpoint, in the same
shape as ``svae_knn_<checkpoint>.json`` (``acc_<N>`` and ``acc_<N>_std`` per label
count), using plain Euclidean distance on the scalar.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from apps.eval.svae_knn import knn_accuracy
from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

_CONFIDENCE_NAMES = {
    "conv_gaussian_vae": "mean_std",
    "conv_vmf_vae": "kappa",
    "conv_tnbbeta_spherical_vae": "p",
}


def main(argv: list[str] | None = None) -> None:
    """Runs the confidence-only probe for one trained run.

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
    confidence_name = _CONFIDENCE_NAMES[kind]

    train_set = load_mnist(data_dir(), split="train")
    test_set = load_mnist(data_dir(), split="test")
    train_confidence = _confidences(model, kind, train_set, args, device)
    test_confidence = _confidences(model, kind, test_set, args, device)
    train_labels, test_labels = train_set.labels(), test_set.labels()

    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": kind,
        "confidence": confidence_name,
        "k": args.k,
        "num_subsets": args.num_subsets,
    }
    for count in args.label_counts:
        accuracies = []
        for _ in range(args.num_subsets):
            chosen = torch.as_tensor(
                rng.choice(len(train_confidence), count, replace=False)
            )
            accuracies.append(
                knn_accuracy(
                    train_confidence[chosen],
                    train_labels[chosen],
                    test_confidence,
                    test_labels,
                    args.k,
                    "euclidean",
                )
            )
        results[f"acc_{count}"] = float(np.mean(accuracies))
        results[f"acc_{count}_std"] = float(np.std(accuracies, ddof=1))

    output = run_dir / f"confidence_probe_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


@torch.no_grad()
def _confidences(
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
        parts.append(_confidence(posterior, kind).cpu())
    return torch.cat(parts)


def _confidence(posterior: Any, kind: str) -> torch.Tensor:
    """Returns the posterior's confidence scalar, shape ``(batch, 1)``."""
    if kind == "conv_gaussian_vae":
        return posterior.base_dist.scale.mean(dim=-1, keepdim=True)
    if kind == "conv_vmf_vae":
        return posterior.scale
    return posterior.p.unsqueeze(-1)


if __name__ == "__main__":
    main(sys.argv[1:])
