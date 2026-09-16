"""Tests for tnbbeta_vae.training.run_logging."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from tnbbeta_vae.training.run_logging import RunLogger


class _DummyConfig(BaseModel):
    learning_rate: float = 1e-3


def test_run_logger_writes_config_and_metadata(tmp_path: Path) -> None:
    logger = RunLogger(model_name="dummy", config=_DummyConfig(), runs_dir=tmp_path)

    config_on_disk = json.loads((logger.run_dir / "config.json").read_text())
    metadata = json.loads((logger.run_dir / "metadata.json").read_text())

    assert config_on_disk == {"learning_rate": 1e-3}
    assert metadata["model_name"] == "dummy"
    assert metadata["run_id"] == logger.run_id
    assert metadata["ended_at"] is None


def test_run_logger_logs_metrics_and_closes(tmp_path: Path) -> None:
    logger = RunLogger(model_name="dummy", config=_DummyConfig(), runs_dir=tmp_path)

    logger.log_metrics(step=0, metrics={"loss": 1.5})
    logger.log_metrics(step=1, metrics={"loss": 1.2})
    logger.close()

    lines = (logger.run_dir / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    metadata = json.loads((logger.run_dir / "metadata.json").read_text())

    assert [r["step"] for r in records] == [0, 1]
    assert [r["loss"] for r in records] == [1.5, 1.2]
    assert metadata["ended_at"] is not None
