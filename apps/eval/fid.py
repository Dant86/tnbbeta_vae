"""Frechet Inception Distance of a trained model's prior samples on CIFAR-10.

Usage:
    uv run python -m apps.eval.fid --run-name NAME [--checkpoint final] \
        [--num-samples 10000] [--batch-size 250]

Reads ``$TNBBETA_CHECKPOINT_DIR/<run-name>/<checkpoint>.pt`` and writes
``fid_<checkpoint>.json`` next to it with:

* ``fid_prior``: FID between ``--num-samples`` images decoded from prior
  samples and the CIFAR-10 test set.
* ``fid_real_floor``: FID between the first ``--num-samples`` train images and
  the test set, the value a perfect generator would approach. FID is biased by
  sample size, so compare ``fid_prior`` to this floor and only across runs with
  the same ``--num-samples``.

Uses the pytorch-fid Inception weights, read from
``$TNBBETA_DATA_DIR/torch_hub``. Download them once beforehand (a 91.2 MB file;
compute nodes may have no internet access, so use the login node) with
``uv run python -m apps.data.download_inception_weights``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pytorch_fid.inception import InceptionV3
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.cifar10 import load_cifar10
from tnbbeta_vae.paths import checkpoint_dir, data_dir, torch_hub_dir
from tnbbeta_vae.training import load_model_checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor

    Extractor = Callable[[Tensor], Tensor]


def main(argv: list[str] | None = None) -> None:
    """Computes prior-sample FID for a checkpoint and writes it next to it.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.hub.set_dir(str(torch_hub_dir()))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    loaded, checkpoint = load_model_checkpoint(
        run_dir / f"{args.checkpoint}.pt", device
    )
    model: Any = loaded

    extractor = _build_extractor(device)
    n = args.num_samples
    test = load_cifar10(data_dir(), train=False)
    train = load_cifar10(data_dir(), train=True)
    test_features = _features(_batches(test, n, args), extractor, device)
    train_features = _features(_batches(train, n, args), extractor, device)
    generated_features = _features(
        _generated_batches(model, n, args.batch_size), extractor, device
    )

    results = {
        "run_name": args.run_name,
        "model_name": checkpoint["model_name"],
        "epochs_completed": checkpoint["epochs_completed"],
        "num_samples": n,
        "fid_prior": frechet_distance(generated_features, test_features),
        "fid_real_floor": frechet_distance(train_features, test_features),
    }
    output = run_dir / f"fid_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Returns the Frechet distance between Gaussians fit to two feature sets.

    Args:
        features_a: Array of shape ``(n_a, d)``.
        features_b: Array of shape ``(n_b, d)``.

    Returns:
        The squared Frechet distance between the two fitted Gaussians.
    """
    mean_a, mean_b = features_a.mean(axis=0), features_b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(features_a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(features_b, rowvar=False))
    # tr(sqrt(cov_a cov_b)) is the sum of square roots of the product's eigenvalues,
    # which are real and non-negative up to numerical noise. (pytorch_fid's own
    # calculate_frechet_distance breaks on current scipy, which dropped sqrtm's disp.)
    eigenvalues = np.linalg.eigvals(cov_a @ cov_b)
    trace_sqrt = np.real(np.sqrt(eigenvalues.astype(complex))).sum()
    difference = mean_a - mean_b
    return float(
        difference @ difference + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_sqrt
    )


@torch.no_grad()
def _features(
    batches: Iterable[Tensor], extractor: Extractor, device: torch.device
) -> np.ndarray:
    parts = [extractor(batch.to(device)).cpu().double() for batch in batches]
    return torch.cat(parts).numpy()


def _batches(dataset: Any, n: int, args: argparse.Namespace) -> Iterable[Tensor]:
    """Yields the first ``n`` dataset images in batches."""
    subset = torch.utils.data.Subset(dataset, range(min(n, len(dataset))))
    return DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers)


@torch.no_grad()
def _generated_batches(model: Any, n: int, batch_size: int) -> Iterable[Tensor]:
    for start in range(0, n, batch_size):
        yield cast("Tensor", model.generate(min(batch_size, n - start)))


def _build_extractor(device: torch.device) -> Extractor:
    """Returns the pytorch-fid Inception pool3 feature extractor (2048-d)."""
    network = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    def extract(images: Tensor) -> Tensor:
        return network(images)[0].flatten(1)

    return extract


if __name__ == "__main__":
    main(sys.argv[1:])
