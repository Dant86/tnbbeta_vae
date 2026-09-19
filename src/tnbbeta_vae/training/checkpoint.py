"""Rebuilding a trained model from a checkpoint written by :class:`Trainer`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

import tnbbeta_vae.models  # noqa: F401 -- import for its @register_model side effects
from tnbbeta_vae.registry import build_model

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

__all__ = ["load_model_checkpoint"]


def load_model_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuilds a registered model from a checkpoint and loads its weights.

    Args:
        path: A file written by ``Trainer.save_checkpoint``.
        map_location: Device to load the tensors onto.

    Returns:
        A tuple ``(model, checkpoint)``: the model in eval mode, and the raw
        checkpoint dict (config, progress counters, run id).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = cast(
        "nn.Module", build_model(checkpoint["model_name"], **checkpoint["config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint
