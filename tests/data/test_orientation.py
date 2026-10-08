"""Tests for tnbbeta_vae.data.orientation (structure-tensor orientation/coherence)."""

from __future__ import annotations

import math

import torch

from tnbbeta_vae.data.orientation import structure_tensor_orientation

_SIZE = 64


def _stripe_image(
    angle: float, *, cycles: float = 6.0, channels: int = 1
) -> torch.Tensor:
    """A single tiled sinusoidal pattern whose wave vector points at ``angle``.

    Args:
        angle: Direction (radians) the pattern varies along; the structure
            tensor should recover this mod pi.
        cycles: Number of full cycles across the image.
        channels: Number of (identical) channels to repeat the pattern over.

    Returns:
        A ``(channels, _SIZE, _SIZE)`` tensor in ``[0, 1]``.
    """
    ys, xs = torch.meshgrid(
        torch.arange(_SIZE, dtype=torch.float32),
        torch.arange(_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    freq = 2 * math.pi * cycles / _SIZE
    wave = torch.cos(freq * (math.cos(angle) * xs + math.sin(angle) * ys))
    image = (wave + 1) / 2
    return image.unsqueeze(0).expand(channels, -1, -1)


def _angle_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, both taken mod pi."""
    diff = (a - b + math.pi / 2) % math.pi - math.pi / 2
    return abs(diff)


def test_recovers_known_angle_for_a_striped_pattern() -> None:
    for degrees in (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0):
        angle = math.radians(degrees)
        batch = _stripe_image(angle).unsqueeze(0)

        recovered, coherence = structure_tensor_orientation(batch)

        assert _angle_diff_mod_pi(recovered.item(), angle) < math.radians(2.0)
        assert coherence.item() > 0.7


def test_coherence_is_low_for_pure_noise() -> None:
    generator = torch.Generator().manual_seed(0)
    noise = torch.rand(8, 1, _SIZE, _SIZE, generator=generator)

    _, coherence = structure_tensor_orientation(noise)

    assert (coherence < 0.2).all()


def test_multichannel_images_are_averaged_to_grayscale_first() -> None:
    angle = math.radians(37.0)
    single = _stripe_image(angle, channels=1).unsqueeze(0)
    triple = _stripe_image(angle, channels=3).unsqueeze(0)

    angle_single, coherence_single = structure_tensor_orientation(single)
    angle_triple, coherence_triple = structure_tensor_orientation(triple)

    assert torch.allclose(angle_single, angle_triple, atol=1e-5)
    assert torch.allclose(coherence_single, coherence_triple, atol=1e-5)


def test_output_shapes_match_batch_size() -> None:
    batch = torch.rand(5, 3, 32, 32)

    angle, coherence = structure_tensor_orientation(batch)

    assert angle.shape == (5,)
    assert coherence.shape == (5,)


def test_angle_is_within_the_canonical_half_open_range() -> None:
    generator = torch.Generator().manual_seed(1)
    batch = torch.rand(32, 1, _SIZE, _SIZE, generator=generator)

    angle, _ = structure_tensor_orientation(batch)

    assert (angle > -math.pi / 2).all()
    assert (angle <= math.pi / 2).all()
