"""A synthetic image dataset with known ground-truth generative factors.

Meant for fast local experiments (e.g. collapse diagnostics) before
touching real data: a small, fully controlled task where "the latent code
carries information about x" has an unambiguous, checkable meaning.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["gaussian_blob_batch"]


def gaussian_blob_batch(
    batch_size: int,
    image_size: int = 32,
    sigma: float = 4.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generates a batch of colored Gaussian-blob images with known factors.

    Each image is a single Gaussian intensity blob on a black background,
    colored by a random hue and centered at a random position.
    Reconstructing an image well requires encoding both its hue and its
    position, so comparing a trained model's latents/reconstructions
    against these known factors can catch a posterior that's collapsed
    (i.e. independent of the input) in a way eyeballing images can't.

    Args:
        batch_size: Number of images to generate.
        image_size: Height/width of the (square) images.
        sigma: Standard deviation (in pixels) of the Gaussian blob.
        generator: Optional ``torch.Generator`` for reproducible batches.

    Returns:
        A tuple ``(images, factors)``: ``images`` has shape
        ``(batch_size, 3, image_size, image_size)`` with values in
        [0, 1]; ``factors`` has shape ``(batch_size, 3)`` with columns
        ``(hue, center_x, center_y)``, ``hue`` in [0, 1) and the centers
        in [0, image_size).
    """
    hue = torch.rand(batch_size, generator=generator)
    center = torch.rand(batch_size, 2, generator=generator) * image_size
    color = _hue_to_rgb(hue)

    coords = torch.arange(image_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    dx = grid_x.unsqueeze(0) - center[:, 0].view(-1, 1, 1)
    dy = grid_y.unsqueeze(0) - center[:, 1].view(-1, 1, 1)
    intensity = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    images = color.view(batch_size, 3, 1, 1) * intensity.unsqueeze(1)
    factors = torch.cat([hue.unsqueeze(-1), center], dim=-1)
    return images, factors


def _hue_to_rgb(hue: Tensor) -> Tensor:
    """Converts hue in [0, 1) to RGB at full saturation/value, shape (..., 3)."""
    h = hue * 6.0
    x = 1 - (h % 2 - 1).abs()
    zeros = torch.zeros_like(h)
    ones = torch.ones_like(h)

    sector = h.floor().long().clamp(max=5).unsqueeze(-1)
    r = torch.stack([ones, x, zeros, zeros, x, ones], dim=-1).gather(-1, sector)
    g = torch.stack([x, ones, ones, x, zeros, zeros], dim=-1).gather(-1, sector)
    b = torch.stack([zeros, zeros, x, ones, ones, x], dim=-1).gather(-1, sector)
    return torch.cat([r, g, b], dim=-1)
