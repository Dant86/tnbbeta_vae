"""Tests for tnbbeta_vae.training.trainer."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import torch
from torch import nn

from tnbbeta_vae.training.trainer import Trainer


class _DummyConfig(BaseModel):
    pass


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.linear(batch)
        return {"loss": prediction.pow(2).mean()}


def test_trainer_fit_runs_and_logs_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    model = _DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, optimizer=optimizer, model_name="dummy", config=_DummyConfig()
    )
    dataloader = [torch.randn(2, 4) for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 2 * 3 + 2  # 3 steps/epoch + 1 epoch-end record, x2 epochs
