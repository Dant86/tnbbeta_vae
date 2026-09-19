"""CLI entrypoint for training a registered model on CIFAR-10.

Usage:
    uv run python -m apps.train.main --list
    uv run python -m apps.train.main --model <name> [--set key=value ...] \
        [--epochs N] [--batch-size N] [--lr X] [--seed N] \
        [--run-name NAME [--resume]] [--uniform-prior]

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
import tnbbeta_vae.models  # noqa: F401 -- import for its @register_model side effects
from tnbbeta_vae.models.priors import uniform_prior_params
from tnbbeta_vae.paths import checkpoint_dir, data_dir, runs_dir
from tnbbeta_vae.registry import (
    build_model,
    get_registered_model,
    list_registered_models,
)
from tnbbeta_vae.training import Trainer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

_SLURM_GPU_VARIABLES = ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
# EX_TEMPFAIL: scripts/slurm/train.sbatch resubmits the job on this exit code.
NO_GPU_EXIT_CODE = 75


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--uniform-prior",
        action="store_true",
        help="For TNBBeta models: set (p, q, epsilon) so the prior is Uniform(sphere).",
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

    overrides = _parse_overrides(args.set)
    if args.uniform_prior:
        overrides.update(_uniform_prior_overrides(args.model, overrides))

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
    loader = DataLoader(
        load_cifar10(data_dir(), train=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Training {args.model} on {device}; checkpoints in {checkpoints}.")
    trainer.fit(_OnDevice(loader, device), args.epochs, checkpoint_dir=checkpoints)


class _OnDevice:
    """Re-iterable wrapper that moves each batch of a dataloader to a device."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[Tensor]:
        for batch in self._loader:
            yield batch.to(self._device, non_blocking=True)


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


def _uniform_prior_overrides(
    model_name: str, overrides: dict[str, str]
) -> dict[str, str]:
    """Returns overrides that make a TNBBeta model's prior uniform on the sphere."""
    fields = get_registered_model(model_name).config_cls.model_fields
    if "prior_p" not in fields:
        raise SystemExit(
            f"--uniform-prior only applies to TNBBeta models, not {model_name!r}."
        )
    latent_dim = int(
        overrides.get("latent_dim") or cast("int", fields["latent_dim"].default)
    )
    p, q, epsilon = uniform_prior_params(latent_dim)
    return {"prior_p": str(p), "prior_q": str(q), "prior_epsilon": str(epsilon)}


if __name__ == "__main__":
    main(sys.argv[1:])
