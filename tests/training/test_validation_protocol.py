"""Tests for the Trainer's validation, best-epoch, patience and KL warm-up logic."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest
import torch
from torch import nn

from tnbbeta_vae.training.trainer import Trainer


class _Config(BaseModel):
    pass


class _ScriptedModel(nn.Module):
    """Training loss is real; validation losses are read from a script."""

    def __init__(self, validation_losses: list[float]) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self._validation_losses = iter(validation_losses)
        self.kl_weights: list[float] = []

    def training_step(
        self, batch: torch.Tensor, kl_weight: float = 1.0
    ) -> dict[str, torch.Tensor]:
        if self.training:
            self.kl_weights.append(kl_weight)
            return {"loss": self.linear(batch).pow(2).mean()}
        return {"loss": torch.tensor(next(self._validation_losses))}


def _trainer(model: nn.Module, tmp_path: Path) -> Trainer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(
        model=model,  # pyright: ignore[reportArgumentType]
        optimizer=optimizer,
        model_name="scripted",
        config=_Config(),
        runs_dir=tmp_path / "runs",
        run_id="run",
    )


_TRAIN = [torch.randn(2, 4)]
_VAL = [torch.randn(2, 4)]


def test_best_checkpoint_becomes_final_and_patience_stops_training(
    tmp_path: Path,
) -> None:
    model = _ScriptedModel([5.0, 4.0, 4.5, 4.6, 4.7, 4.8])
    trainer = _trainer(model, tmp_path)

    trainer.fit(
        _TRAIN,
        num_epochs=10,
        checkpoint_dir=tmp_path / "ckpt",
        val_dataloader=_VAL,
        patience=2,
    )

    assert trainer.epochs_completed == 4
    final = torch.load(tmp_path / "ckpt" / "final.pt")
    assert final["epochs_completed"] == 2
    assert final["best_val_loss"] == pytest.approx(4.0)
    latest = torch.load(tmp_path / "ckpt" / "latest.pt")
    assert latest["epochs_completed"] == 4
    assert latest["epochs_since_improvement"] == 2


def test_kl_warmup_weights_and_best_tracking_start_after_warmup(
    tmp_path: Path,
) -> None:
    model = _ScriptedModel([0.1, 0.1, 5.0, 4.0, 4.5])
    trainer = _trainer(model, tmp_path)

    trainer.fit(
        _TRAIN,
        num_epochs=5,
        checkpoint_dir=tmp_path / "ckpt",
        val_dataloader=_VAL,
        kl_warmup_epochs=2,
    )

    assert model.kl_weights == [0.0, 0.5, 1.0, 1.0, 1.0]
    final = torch.load(tmp_path / "ckpt" / "final.pt")
    assert final["best_val_loss"] == pytest.approx(4.0)
    assert final["epochs_completed"] == 4
    metrics = (trainer.run_logger.run_dir / "metrics.jsonl").read_text()
    assert '"val_loss"' in metrics and '"kl_weight"' in metrics


def test_resume_restores_patience_state_and_stops_immediately(
    tmp_path: Path,
) -> None:
    first = _trainer(_ScriptedModel([1.0, 2.0, 2.0]), tmp_path)
    first.fit(
        _TRAIN,
        num_epochs=3,
        checkpoint_dir=tmp_path / "ckpt",
        val_dataloader=_VAL,
    )

    resumed_model = _ScriptedModel([])
    resumed = _trainer(resumed_model, tmp_path)
    resumed.load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    resumed.fit(
        _TRAIN,
        num_epochs=10,
        checkpoint_dir=tmp_path / "ckpt",
        val_dataloader=_VAL,
        patience=2,
    )

    assert resumed.best_val_loss == pytest.approx(1.0)
    assert resumed.epochs_completed == 3
    assert resumed_model.kl_weights == []


def test_without_validation_final_is_the_last_state(tmp_path: Path) -> None:
    trainer = _trainer(_ScriptedModel([]), tmp_path)

    trainer.fit(_TRAIN, num_epochs=2, checkpoint_dir=tmp_path / "ckpt")

    assert not (tmp_path / "ckpt" / "best.pt").exists()
    assert torch.load(tmp_path / "ckpt" / "final.pt")["epochs_completed"] == 2
