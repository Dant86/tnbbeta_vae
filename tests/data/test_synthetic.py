"""Tests for tnbbeta_vae.data.synthetic."""

from __future__ import annotations

import torch

from tnbbeta_vae.data import gaussian_blob_batch
from tnbbeta_vae.data.synthetic import _hue_to_rgb


def test_output_shapes_and_ranges() -> None:
    images, factors = gaussian_blob_batch(batch_size=5, image_size=32)

    assert images.shape == (5, 3, 32, 32)
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)

    assert factors.shape == (5, 3)
    hue, center_x, center_y = factors[:, 0], factors[:, 1], factors[:, 2]
    assert torch.all((hue >= 0.0) & (hue < 1.0))
    assert torch.all((center_x >= 0.0) & (center_x < 32))
    assert torch.all((center_y >= 0.0) & (center_y < 32))


def test_generator_makes_batches_reproducible() -> None:
    images_a, factors_a = gaussian_blob_batch(
        batch_size=4, generator=torch.Generator().manual_seed(0)
    )
    images_b, factors_b = gaussian_blob_batch(
        batch_size=4, generator=torch.Generator().manual_seed(0)
    )

    assert torch.equal(images_a, images_b)
    assert torch.equal(factors_a, factors_b)


def test_blob_peak_is_near_claimed_center() -> None:
    """The brightest pixel should land within a pixel or two of the true center."""
    torch.manual_seed(1)
    images, factors = gaussian_blob_batch(batch_size=8, image_size=32, sigma=4.0)

    intensity = images.sum(dim=1)  # (batch, H, W)
    peak_flat = intensity.flatten(1).argmax(dim=-1)
    peak_y, peak_x = peak_flat // 32, peak_flat % 32

    assert torch.all((peak_x.float() - factors[:, 1]).abs() <= 2)
    assert torch.all((peak_y.float() - factors[:, 2]).abs() <= 2)


def test_hue_to_rgb_matches_known_primary_colors() -> None:
    hues = torch.tensor([0.0, 1 / 3, 2 / 3])
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rgb = _hue_to_rgb(hues)

    assert torch.allclose(rgb, expected, atol=1e-5)


def test_different_hues_give_different_colors() -> None:
    """Sanity check that hue actually controls color (not a constant)."""
    colors = _hue_to_rgb(torch.tensor([0.0, 0.5]))

    assert not torch.allclose(colors[0], colors[1])
