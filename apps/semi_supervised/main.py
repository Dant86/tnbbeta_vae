"""Semi-supervised MNIST with the stacked M1+M2 model (S-VAE paper, Table 3).

Usage:
    uv run python -m apps.semi_supervised.main --run-name NAME \
        --z1-family vmf --z2-family gaussian --z1-dim 10 --z2-dim 10 \
        [--num-labels 100] [--alpha 0.5] [--batch-size 100] [--patience 50]

Trains :class:`M1M2VAE` end to end. Each step uses all ``--num-labels`` labelled
examples (class balanced, chosen by ``--seed``) and ``--batch-size`` unlabelled ones
from the 50k-image training set. Early stopping uses the *unlabelled* objective on the
10k validation images, so no labels beyond ``--num-labels`` are used, and the final
checkpoint is the best validation epoch. Writes ``semi_supervised_final.json`` next to
it with the test accuracy of the classifier applied to the centre of ``q(z1|x)`` on
the 10k test images. ``--resume`` continues from the latest checkpoint and skips a
finished run.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

import torch

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.data.semi_supervised import SemiSupervisedBatches, UnlabeledBatches
from tnbbeta_vae.models import M1M2VAE, M1M2Config
from tnbbeta_vae.paths import checkpoint_dir, data_dir, runs_dir
from tnbbeta_vae.training import Trainer, load_model_checkpoint, select_device

_VAL_BATCH_SIZE = 1000
_FAMILIES = ["gaussian", "vmf", "tnbbeta"]


def main(argv: list[str] | None = None) -> None:
    """Trains one model and writes its test accuracy.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--z1-family", choices=_FAMILIES, default="vmf")
    parser.add_argument("--z2-family", choices=_FAMILIES, default="gaussian")
    parser.add_argument("--z1-dim", type=int, default=10)
    parser.add_argument("--z2-dim", type=int, default=10)
    parser.add_argument("--num-labels", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--kappa-init",
        choices=["reference", "dimension"],
        default="reference",
        help="Start vMF kappa at ~1.7 (reference) or at the latent dimension.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    checkpoints = checkpoint_dir() / args.run_name
    if args.resume and (checkpoints / "final.pt").exists():
        print(f"{args.run_name}: already complete ({checkpoints / 'final.pt'}).")
        return

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_set = load_mnist(data_dir(), split="train")
    config = M1M2Config(
        z1_family=cast("Any", args.z1_family),
        z2_family=cast("Any", args.z2_family),
        z1_dim=args.z1_dim,
        z2_dim=args.z2_dim,
        alpha=args.alpha,
        num_examples=len(train_set),
        kappa_init=args.kappa_init,
    )
    model = M1M2VAE(config).to(device)
    trainer = Trainer(
        model=model,  # pyright: ignore[reportArgumentType]
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        model_name="m1m2_vae",
        config=config,
        runs_dir=runs_dir(),
        run_id=args.run_name,
    )
    if args.resume and (checkpoints / "latest.pt").exists():
        trainer.load_checkpoint(checkpoints / "latest.pt")
        print(f"Resumed from epoch {trainer.epochs_completed}.")

    train_batches = SemiSupervisedBatches(
        train_set, args.num_labels, args.batch_size, device, args.seed
    )
    val_batches = UnlabeledBatches(
        load_mnist(data_dir(), split="val"), _VAL_BATCH_SIZE, device
    )
    print(f"Training m1m2_vae on {device}; checkpoints in {checkpoints}.")
    trainer.fit(
        train_batches,
        args.epochs,
        checkpoint_dir=checkpoints,
        val_dataloader=val_batches,
        patience=args.patience,
    )

    best, checkpoint = load_model_checkpoint(checkpoints / "final.pt", device)
    accuracy = _test_accuracy(best, device)
    results = {
        "run_name": args.run_name,
        "z1_family": args.z1_family,
        "z2_family": args.z2_family,
        "z1_dim": args.z1_dim,
        "z2_dim": args.z2_dim,
        "num_labels": args.num_labels,
        "alpha": args.alpha,
        "kappa_init": args.kappa_init,
        "seed": args.seed,
        "epochs_completed": checkpoint["epochs_completed"],
        "test_accuracy": accuracy,
    }
    output = checkpoints / "semi_supervised_final.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


@torch.no_grad()
def _test_accuracy(model: Any, device: torch.device) -> float:
    test_set = load_mnist(data_dir(), split="test")
    images = test_set.images.flatten(1).to(device)
    labels = test_set.labels().to(device)
    predictions = torch.cat(
        [model.predict(chunk) for chunk in images.split(_VAL_BATCH_SIZE)]
    )
    return (predictions == labels).float().mean().item()


if __name__ == "__main__":
    main(sys.argv[1:])
