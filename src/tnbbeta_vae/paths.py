"""Filesystem locations, configured via environment variables or a ``.env`` file.

Three directories are configurable (see ``.env.sample``):

* ``TNBBETA_DATA_DIR``: where datasets (e.g. CIFAR-10) are stored.
* ``TNBBETA_CHECKPOINT_DIR``: where model checkpoints are written.
* ``TNBBETA_RUNS_DIR``: where run configs and metric logs are written.

A ``.env`` file in (or above) the current working directory is loaded if
present. Real environment variables take precedence over ``.env``, so a
job script can override any of them with ``export``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

__all__ = ["checkpoint_dir", "data_dir", "runs_dir", "torch_hub_dir"]

_DEFAULTS = {
    "TNBBETA_DATA_DIR": "data",
    "TNBBETA_CHECKPOINT_DIR": "checkpoints",
    "TNBBETA_RUNS_DIR": "runs",
}


def data_dir() -> Path:
    """Returns the dataset directory (``TNBBETA_DATA_DIR``)."""
    return _path("TNBBETA_DATA_DIR")


def checkpoint_dir() -> Path:
    """Returns the checkpoint directory (``TNBBETA_CHECKPOINT_DIR``)."""
    return _path("TNBBETA_CHECKPOINT_DIR")


def runs_dir() -> Path:
    """Returns the run-log directory (``TNBBETA_RUNS_DIR``)."""
    return _path("TNBBETA_RUNS_DIR")


def torch_hub_dir() -> Path:
    """Returns where pretrained weights are cached: ``<TNBBETA_DATA_DIR>/torch_hub``."""
    return data_dir() / "torch_hub"


def _path(name: str) -> Path:
    load_dotenv(find_dotenv(usecwd=True))
    return Path(os.environ.get(name, _DEFAULTS[name])).expanduser()
