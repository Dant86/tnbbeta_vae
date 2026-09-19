"""Tests for Trainer checkpoint/resume and load_model_checkpoint."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import torch
from torch import nn

from tnbbeta_vae.models import ConvGaussianVAEConfig
from tnbbeta_vae.registry import build_model
from tnbbeta_vae.training import Trainer, load_model_checkpoint


class _Config(BaseModel):
    pass


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.linear(batch).pow(2).mean()}


def _trainer(tmp_path: Path, run_id: str = "run") -> Trainer[torch.Tensor]:
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return Trainer(
        model, optimizer, "dummy", _Config(), runs_dir=tmp_path / "runs", run_id=run_id
    )


def test_writes_latest_and_final_checkpoints(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    trainer.fit([torch.randn(2, 4)] * 3, num_epochs=2, checkpoint_dir=tmp_path / "ck")

    assert (tmp_path / "ck" / "latest.pt").exists()
    assert (tmp_path / "ck" / "final.pt").exists()
    assert not list((tmp_path / "ck").glob("*.tmp"))


def test_resume_continues_only_remaining_epochs(tmp_path: Path) -> None:
    data = [torch.randn(2, 4)] * 3
    first = _trainer(tmp_path)
    first.fit(data, num_epochs=1, checkpoint_dir=tmp_path / "ck")
    saved = {k: v.clone() for k, v in first.model.state_dict().items()}

    resumed = _trainer(tmp_path)
    resumed.load_checkpoint(tmp_path / "ck" / "latest.pt")

    assert resumed.epochs_completed == 1
    assert resumed.step == 3
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, saved[name])

    resumed.fit(data, num_epochs=3, checkpoint_dir=tmp_path / "ck")
    assert resumed.epochs_completed == 3
    assert resumed.step == 9
    metrics = (tmp_path / "runs" / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) == 3 * (3 + 1)


def test_load_model_checkpoint_rebuilds_registered_model(tmp_path: Path) -> None:
    config = ConvGaussianVAEConfig(latent_dim=4, hidden_channels=8)
    model = build_model("conv_gaussian_vae", **config.model_dump())
    assert isinstance(model, nn.Module)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(
        model,  # pyright: ignore[reportArgumentType]
        optimizer,
        "conv_gaussian_vae",
        config,
        runs_dir=tmp_path / "runs",
        run_id="r",
    )
    trainer.save_checkpoint(tmp_path / "ck.pt")

    loaded, checkpoint = load_model_checkpoint(tmp_path / "ck.pt")

    assert checkpoint["model_name"] == "conv_gaussian_vae"
    assert not loaded.training
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
