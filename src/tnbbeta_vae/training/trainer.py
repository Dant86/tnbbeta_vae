"""Generic training loop shared across models registered in tnbbeta_vae.

This is intentionally minimal: a model-agnostic step/epoch loop plus run
logging. Model-specific behavior (loss composition, metric computation)
lives on the model via the ``training_step`` protocol below, not here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from tnbbeta_vae.training.run_logging import RunLogger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pydantic import BaseModel

__all__ = ["Trainer", "TrainableModel"]

_DEFAULT_RUNS_DIR = Path("runs")


class TrainableModel[BatchT](Protocol):
    """Protocol a model must satisfy to be driven by :class:`Trainer`.

    Matches the subset of ``torch.nn.Module``'s interface this trainer
    relies on, plus ``training_step``, so any ``nn.Module`` subclass that
    also implements ``training_step`` satisfies this structurally.
    """

    def train(self, mode: bool = True) -> object:
        """Puts the model in training mode (see ``torch.nn.Module.train``)."""
        ...

    def training_step(self, batch: BatchT) -> dict[str, torch.Tensor]:
        """Runs one training step and returns a dict including a "loss" key."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Returns the model's parameters (see ``torch.nn.Module.state_dict``)."""
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> object:
        """Loads parameters (see ``torch.nn.Module.load_state_dict``)."""
        ...


class Trainer[BatchT]:
    """Runs a model-agnostic epoch loop over a dataloader, logging metrics."""

    def __init__(
        self,
        model: TrainableModel[BatchT],
        optimizer: torch.optim.Optimizer,
        model_name: str,
        config: BaseModel,
        runs_dir: Path = _DEFAULT_RUNS_DIR,
        run_id: str | None = None,
    ) -> None:
        """Initializes the trainer and starts a new run log.

        Args:
            model: Model to train. Must implement :class:`TrainableModel`.
            optimizer: Optimizer over ``model``'s parameters.
            model_name: Registry name of ``model``, for run metadata.
            config: The run's config, recorded verbatim in the run log.
            runs_dir: Parent directory for the run's log directory.
            run_id: Run identifier; pass a fixed value to resume a run
                into the same log directory. Defaults to a fresh id.
        """
        self.model = model
        self.optimizer = optimizer
        self.model_name = model_name
        self.config = config
        self.run_logger = RunLogger(
            model_name=model_name, config=config, runs_dir=runs_dir, run_id=run_id
        )
        self.step = 0
        self.epochs_completed = 0

    def fit(
        self,
        dataloader: Iterable[BatchT],
        num_epochs: int,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Trains until ``num_epochs`` epochs are complete.

        Continues from ``self.epochs_completed`` (nonzero after
        :meth:`load_checkpoint`), so a resumed run only does the remaining
        epochs.

        Args:
            dataloader: Iterable of batches, each passed to the model's
                ``training_step``. Must be re-iterable once per epoch.
            num_epochs: Total number of epochs to have completed at the end.
            checkpoint_dir: If given, ``latest.pt`` is rewritten after every
                epoch and ``final.pt`` once training completes.
        """
        self.model.train()
        for epoch in range(self.epochs_completed, num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model.training_step(batch)
                outputs["loss"].backward()
                self.optimizer.step()

                metrics = {k: v.item() for k, v in outputs.items()}
                self.run_logger.log_metrics(step=self.step, metrics=metrics)
                self.step += 1
            self.epochs_completed = epoch + 1
            self.run_logger.log_metrics(step=self.step, metrics={"epoch": epoch})
            if checkpoint_dir is not None:
                self.save_checkpoint(checkpoint_dir / "latest.pt")
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir / "final.pt")
        self.run_logger.close()

    def save_checkpoint(self, path: Path) -> None:
        """Atomically writes a checkpoint (model, optimizer, progress, config).

        Written to a temporary file and renamed, so a job killed mid-write
        never leaves a truncated checkpoint behind.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "model_name": self.model_name,
                "config": self.config.model_dump(),
                "run_id": self.run_logger.run_id,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epochs_completed": self.epochs_completed,
                "step": self.step,
            },
            temporary,
        )
        os.replace(temporary, path)

    def load_checkpoint(self, path: Path) -> None:
        """Restores model, optimizer and progress from a checkpoint.

        Args:
            path: A file written by :meth:`save_checkpoint`.
        """
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epochs_completed = checkpoint["epochs_completed"]
        self.step = checkpoint["step"]
