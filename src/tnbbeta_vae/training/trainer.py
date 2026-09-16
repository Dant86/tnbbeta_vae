"""Generic training loop shared across models registered in tnbbeta_vae.

This is intentionally minimal: a model-agnostic step/epoch loop plus run
logging. Model-specific behavior (loss composition, metric computation)
lives on the model via the ``training_step`` protocol below, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from tnbbeta_vae.training.run_logging import RunLogger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel

__all__ = ["Trainer", "TrainableModel"]


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


class Trainer[BatchT]:
    """Runs a model-agnostic epoch loop over a dataloader, logging metrics."""

    def __init__(
        self,
        model: TrainableModel[BatchT],
        optimizer: torch.optim.Optimizer,
        model_name: str,
        config: BaseModel,
    ) -> None:
        """Initializes the trainer and starts a new run log.

        Args:
            model: Model to train. Must implement :class:`TrainableModel`.
            optimizer: Optimizer over ``model``'s parameters.
            model_name: Registry name of ``model``, for run metadata.
            config: The run's config, recorded verbatim in the run log.
        """
        self.model = model
        self.optimizer = optimizer
        self.run_logger = RunLogger(model_name=model_name, config=config)

    def fit(self, dataloader: Iterable[BatchT], num_epochs: int) -> None:
        """Trains for ``num_epochs`` passes over ``dataloader``.

        Args:
            dataloader: Iterable of batches, each passed to the model's
                ``training_step``.
            num_epochs: Number of passes over ``dataloader``.
        """
        step = 0
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model.training_step(batch)
                outputs["loss"].backward()
                self.optimizer.step()

                metrics = {k: v.item() for k, v in outputs.items()}
                self.run_logger.log_metrics(step=step, metrics=metrics)
                step += 1
            self.run_logger.log_metrics(step=step, metrics={"epoch": epoch})
        self.run_logger.close()
