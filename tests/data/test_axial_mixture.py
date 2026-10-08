"""Tests for tnbbeta_vae.data.axial_mixture."""

from __future__ import annotations

import math

import torch

from tnbbeta_vae.data.axial_mixture import axial_mixture_data
from tnbbeta_vae.data.circle_mixture import circle_mixture_data


def test_embed_is_invariant_under_a_shift_of_pi() -> None:
    data = axial_mixture_data(ambient_dim=20, seed=1)
    angle = torch.linspace(-3.0, 3.0, 13)

    assert torch.allclose(data.embed(angle), data.embed(angle + math.pi), atol=1e-5)


def test_embed_reuses_circle_mixture_on_the_doubled_angle() -> None:
    angle = torch.linspace(-3.0, 3.0, 13)

    axial = axial_mixture_data(ambient_dim=20, seed=1)
    circle = circle_mixture_data(ambient_dim=20, seed=1)

    assert torch.equal(axial.embed(angle), circle.embed(2 * angle))


def test_samples_are_reproducible_and_shaped() -> None:
    data = axial_mixture_data(ambient_dim=20, seed=1)

    first = data.sample(50, torch.Generator().manual_seed(3))
    second = data.sample(50, torch.Generator().manual_seed(3))

    x, angle, component = first
    assert x.shape == (50, 20)
    assert angle.shape == component.shape == (50,)
    assert set(component.tolist()) <= {0, 1}
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_components_are_roughly_balanced() -> None:
    _, _, component = axial_mixture_data(seed=0).sample(
        2000, torch.Generator().manual_seed(0)
    )

    fraction_ones = component.float().mean().item()
    assert 0.4 < fraction_ones < 0.6


def test_component_is_not_recoverable_from_the_observation_alone() -> None:
    """The whole point: a component-0 and a component-1 angle at the same
    recovered axis give (noise-free) identical observations."""
    data = axial_mixture_data(ambient_dim=20, seed=1)
    angle = torch.tensor([0.4])

    component_0 = data.embed(angle)
    component_1 = data.embed(angle + math.pi)

    assert torch.allclose(component_0, component_1, atol=1e-5)
