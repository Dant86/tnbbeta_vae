"""Lightweight local run logging for train-run metadata and metrics.

Each run gets a directory (default: ``runs/<run_id>/``) containing:

* ``config.json``: the run's config, as passed to :class:`RunLogger`.
* ``metadata.json``: run id, model name, start/end timestamps.
* ``metrics.jsonl``: one JSON object per :meth:`RunLogger.log_metrics` call,
    each with a ``step`` and a UTC ``timestamp``.

This keeps every run's metadata in a plain, greppable, pandas-readable
format on disk, so past runs can be scanned or loaded without a database or
external tracking service.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RunLogger"]

_DEFAULT_RUNS_DIR = Path("runs")


class RunLogger:
    """Writes a single train run's config and metrics to a run directory."""

    def __init__(
        self,
        model_name: str,
        config: BaseModel,
        runs_dir: Path = _DEFAULT_RUNS_DIR,
        run_id: str | None = None,
    ) -> None:
        """Initializes the run directory and writes the config to disk.

        Args:
            model_name: Registry name of the model being trained.
            config: The model/train config for this run.
            runs_dir: Parent directory under which ``<run_id>/`` is created.
            run_id: Unique id for this run. Defaults to a timestamp-prefixed
                UUID so runs sort chronologically by directory name.
        """
        self.model_name = model_name
        self.run_id = run_id or self._make_run_id()
        self.run_dir = runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._started_at = datetime.now(UTC)
        (self.run_dir / "config.json").write_text(config.model_dump_json(indent=2))
        self._write_metadata(ended_at=None)

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Appends one line of metrics to ``metrics.jsonl``.

        Args:
            step: Training step or epoch these metrics correspond to.
            metrics: Metric name to value.
        """
        record = {
            "step": step,
            "timestamp": datetime.now(UTC).isoformat(),
            **metrics,
        }
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Marks the run as finished by recording an end timestamp."""
        self._write_metadata(ended_at=datetime.now(UTC))

    @staticmethod
    def _make_run_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def _write_metadata(self, *, ended_at: datetime | None) -> None:
        metadata = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "started_at": self._started_at.isoformat(),
            "ended_at": ended_at.isoformat() if ended_at else None,
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
