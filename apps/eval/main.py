"""Evaluates a trained model checkpoint on CIFAR-10 and saves metrics and images.

Usage:
    uv run python -m apps.eval.main --run-name NAME [--checkpoint final|latest] \
        [--split test|train] [--num-samples K] [--num-prior-samples N]

Reads ``$TNBBETA_CHECKPOINT_DIR/<run-name>/<checkpoint>.pt`` and the data in
``$TNBBETA_DATA_DIR``, and writes into the same checkpoint directory:

* ``eval_<checkpoint>_<split>.json``: ELBO, log-likelihood, KL (nats per
  image), reconstruction MSE/PSNR, and the prior-sample nearest-neighbour
  score (see below).
* ``prior_samples_<checkpoint>.png`` and ``reconstructions_<checkpoint>.png``.

The nearest-neighbour score is the mean per-pixel squared distance from
each prior sample to its nearest training image, divided by the same
quantity for held-out real images; 1.0 means prior samples are as close to
the data as real images are, and larger values mean the prior places mass
where the decoder produces off-data images ("prior holes").
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tnbbeta_vae.data.cifar10 import load_cifar10
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

if TYPE_CHECKING:
    from torch import Tensor

_NN_CHUNK = 256


def main(argv: list[str] | None = None) -> None:
    """Evaluates a checkpoint and writes results next to it.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument(
        "--num-samples", type=int, default=16, help="z draws per image for the ELBO."
    )
    parser.add_argument("--num-prior-samples", type=int, default=1000)
    parser.add_argument(
        "--reference-size",
        type=int,
        default=10_000,
        help="Train images used as NN reference.",
    )
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
    loaded, checkpoint = load_model_checkpoint(
        run_dir / f"{args.checkpoint}.pt", device
    )
    model: Any = loaded
    model.config = model.config.model_copy(
        update={"num_elbo_samples": args.num_samples}
    )

    dataset = load_cifar10(data_dir(), train=args.split == "train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": checkpoint["model_name"],
        "config": checkpoint["config"],
        "epochs_completed": checkpoint["epochs_completed"],
        "split": args.split,
        "elbo_samples": args.num_samples,
        "likelihood_scale": model.learned_scale().item(),
        **_elbo_metrics(model, loader, device),
        **_prior_metrics(model, args, device),
    }
    _save_images(model, loader, run_dir, args.checkpoint, device)

    output = run_dir / f"eval_{args.checkpoint}_{args.split}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


@torch.no_grad()
def _elbo_metrics(
    model: Any, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    totals = {"elbo": 0.0, "log_likelihood": 0.0, "kl": 0.0, "mse": 0.0}
    count = 0
    for batch in loader:
        batch = batch.to(device)
        n = batch.shape[0]
        out = cast("dict[str, Tensor]", model.training_step(batch))
        reconstruction = model(batch)[0]
        totals["elbo"] += -out["loss"].item() * n
        totals["log_likelihood"] += out["log_likelihood"].item() * n
        totals["kl"] += out["kl"].item() * n
        totals["mse"] += (reconstruction - batch).pow(2).mean().item() * n
        count += n
    metrics = {f"{k}": v / count for k, v in totals.items()}
    metrics["psnr_db"] = -10 * math.log10(max(metrics["mse"], 1e-12))
    return metrics


@torch.no_grad()
def _prior_metrics(
    model: Any, args: argparse.Namespace, device: torch.device
) -> dict[str, float]:
    train_set = load_cifar10(data_dir(), train=True)
    reference = torch.stack(
        [train_set[i] for i in range(min(args.reference_size, len(train_set)))]
    )
    reference = reference.flatten(1).to(device)
    n = args.num_prior_samples
    generated = cast("Tensor", model.generate(n))
    held_out = load_cifar10(
        data_dir(), train=False
    )  # always held-out, whatever --split is
    real = torch.stack([held_out[i] for i in range(min(n, len(held_out)))]).to(device)
    prior_nn = _nearest_neighbour_mse(generated.flatten(1), reference)
    real_nn = _nearest_neighbour_mse(real.flatten(1), reference)
    return {
        "prior_nn_mse": prior_nn,
        "real_nn_mse": real_nn,
        "prior_nn_ratio": prior_nn / real_nn,
    }


def _nearest_neighbour_mse(queries: Tensor, reference: Tensor) -> float:
    """Mean over queries of the per-pixel squared distance to the nearest reference."""
    total = 0.0
    for start in range(0, queries.shape[0], _NN_CHUNK):
        chunk = queries[start : start + _NN_CHUNK]
        distances = torch.cdist(chunk, reference) ** 2 / reference.shape[1]
        total += distances.min(dim=1).values.sum().item()
    return total / queries.shape[0]


@torch.no_grad()
def _save_images(
    model: Any, loader: DataLoader, run_dir: Path, tag: str, device: torch.device
) -> None:
    generated = cast("Tensor", model.generate(64))
    save_image(generated, run_dir / f"prior_samples_{tag}.png", nrow=8)
    batch = next(iter(loader)).to(device)[:32]
    reconstruction = model(batch)[0]
    save_image(
        torch.cat([batch, reconstruction]),
        run_dir / f"reconstructions_{tag}.png",
        nrow=8,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
