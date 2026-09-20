"""CLI entrypoint for training a registered model on CIFAR-10 or MNIST.

Usage:
    uv run python -m apps.train.main --list
    uv run python -m apps.train.main --model <name> [--set key=value ...] \
        [--dataset cifar10|mnist] [--epochs N] [--batch-size N] [--lr X] \
        [--seed N] [--patience N] [--kl-warmup-epochs N] \
        [--run-name NAME [--resume]]

``--dataset mnist`` trains on dynamically binarized MNIST (50k train images),
validates on the 10k validation images after every epoch, keeps the best
validation epoch as ``final.pt``, and sets the model's ``image_channels=1``,
``image_size=28`` and ``likelihood=bernoulli`` unless ``--set`` overrides them.
The S-VAE paper's protocol is ``--batch-size 64 --epochs 1000 --patience 50
--kl-warmup-epochs 100``.

Data, checkpoint and run-log locations come from ``.env`` (see
``.env.sample``). With ``--run-name``, checkpoints go to
``$TNBBETA_CHECKPOINT_DIR/<run-name>/`` and ``--resume`` continues from
``latest.pt`` there (or exits immediately if ``final.pt`` already exists),
which makes the command safe to re-run after a job is preempted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from tnbbeta_vae.data.cifar10 import load_cifar10
from tnbbeta_vae.data.mnist import DeviceBatches, load_mnist
import tnbbeta_vae.models  # noqa: F401 -- import for its @register_model side effects
from tnbbeta_vae.paths import checkpoint_dir, data_dir, runs_dir
from tnbbeta_vae.registry import (
    build_model,
    get_registered_model,
    list_registered_models,
)
from tnbbeta_vae.training import Trainer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch import Tensor

_SLURM_GPU_VARIABLES = ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
# EX_TEMPFAIL: scripts/slurm/train.sbatch resubmits the job on this exit code.
NO_GPU_EXIT_CODE = 75
_MNIST_MODEL_DEFAULTS = {
    "image_channels": "1",
    "image_size": "28",
    "likelihood": "bernoulli",
}
_VAL_BATCH_SIZE = 1000


def main(argv: list[str] | None = None) -> None:
    """Parses CLI args and trains a registered model.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List registered model names and exit."
    )
    parser.add_argument("--model", type=str, help="Registered model name to train.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Model config override, may be repeated.",
    )
    parser.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="MNIST: stop after this many epochs without a better validation loss.",
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=0,
        help="Raise the KL weight linearly from 0 to 1 over this many epochs.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default=None, help="Default: cuda if available."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Fixed run id (names the run/checkpoint directories).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <run-name>'s latest checkpoint if one exists.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in list_registered_models():
            print(name)
        return
    if not args.model:
        parser.error("--model is required unless --list is passed.")
    if args.resume and not args.run_name:
        parser.error("--resume requires --run-name.")

    if args.dataset == "cifar10" and args.patience is not None:
        parser.error("--patience needs a validation set; use --dataset mnist.")
    overrides = _parse_overrides(args.set)
    if args.dataset == "mnist":
        overrides = {**_MNIST_MODEL_DEFAULTS, **overrides}

    checkpoints = checkpoint_dir() / args.run_name if args.run_name else None
    if args.resume and checkpoints and (checkpoints / "final.pt").exists():
        print(f"{args.run_name}: already complete ({checkpoints / 'final.pt'}).")
        return

    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    model = cast("nn.Module", build_model(args.model, **overrides)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(
        model=model,  # pyright: ignore[reportArgumentType]
        optimizer=optimizer,
        model_name=args.model,
        config=get_registered_model(args.model).config_cls(**overrides),
        runs_dir=runs_dir(),
        run_id=args.run_name,
    )
    if checkpoints is None:
        checkpoints = checkpoint_dir() / trainer.run_logger.run_id
    if args.resume and (checkpoints / "latest.pt").exists():
        trainer.load_checkpoint(checkpoints / "latest.pt")
        print(f"Resumed from epoch {trainer.epochs_completed}.")

    (trainer.run_logger.run_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2)
    )
    train_batches, val_batches = _batches(args, device)
    print(f"Training {args.model} on {device}; checkpoints in {checkpoints}.")
    trainer.fit(
        train_batches,
        args.epochs,
        checkpoint_dir=checkpoints,
        val_dataloader=val_batches,
        patience=args.patience,
        kl_warmup_epochs=args.kl_warmup_epochs,
    )


class _OnDevice:
    """Re-iterable wrapper that moves each batch of a dataloader to a device."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[Tensor]:
        for batch in self._loader:
            yield batch.to(self._device, non_blocking=True)


def _batches(
    args: argparse.Namespace, device: torch.device
) -> tuple[Iterable[Tensor], Iterable[Tensor] | None]:
    """Builds the train batches and, for MNIST, the validation batches (on device)."""
    if args.dataset == "mnist":
        train = DeviceBatches(
            load_mnist(data_dir(), split="train"),
            args.batch_size,
            device,
            shuffle=True,
            drop_last=True,
        )
        val = DeviceBatches(
            load_mnist(data_dir(), split="val"),
            _VAL_BATCH_SIZE,
            device,
            shuffle=False,
        )
        return train, val
    loader = DataLoader(
        load_cifar10(data_dir(), train=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return _OnDevice(loader, device), None


def _select_device(requested: str | None) -> torch.device:
    """Picks the training device, refusing a silent CPU fallback under Slurm.

    If a GPU was allocated to the job but CUDA can't initialize (a node
    problem), PyTorch quietly falls back to the CPU and the run crawls.
    Failing here makes that visible immediately. An explicit ``--device``
    is always honored.

    Args:
        requested: The ``--device`` argument, or ``None`` to choose.

    Returns:
        The device to train on.

    Raises:
        SystemExit: With ``NO_GPU_EXIT_CODE`` if a Slurm GPU was allocated but
            CUDA is unavailable.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if any(os.environ.get(name) for name in _SLURM_GPU_VARIABLES):
        print(
            "Slurm allocated a GPU but CUDA is unavailable on this node; refusing "
            "to fall back to the CPU. Resubmit, excluding this node (sbatch "
            "--exclude=<node>), or pass --device cpu to train on the CPU anyway.",
            file=sys.stderr,
        )
        raise SystemExit(NO_GPU_EXIT_CODE)
    return torch.device("cpu")


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Parses ``key=value`` strings into a dict.

    Args:
        pairs: Strings of the form ``"key=value"``.

    Returns:
        Mapping from key to value.
    """
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key] = value
    return overrides


if __name__ == "__main__":
    main(sys.argv[1:])
