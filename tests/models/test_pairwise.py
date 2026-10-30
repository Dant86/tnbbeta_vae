"""Tests for tnbbeta_vae.models.pairwise."""

from __future__ import annotations

import torch

from tnbbeta_vae.models.pairwise import pairwise_logits


def test_gaussian_family_skips_temperature_scaling() -> None:
    z = torch.randn(5, 4)
    pairs = torch.tensor([[0, 1], [2, 3]])
    inner = (z[pairs[0]] * z[pairs[1]]).sum(-1)

    logits = pairwise_logits(z, pairs, "gaussian", torch.tensor(5.0))

    assert torch.allclose(logits, inner)


def test_sphere_family_scales_by_temperature() -> None:
    z = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    pairs = torch.tensor([[0, 1], [2, 3]])
    inner = (z[pairs[0]] * z[pairs[1]]).sum(-1)

    logits = pairwise_logits(z, pairs, "vmf", torch.tensor(3.0))

    assert torch.allclose(logits, 3.0 * inner)


def test_output_shape_matches_number_of_pairs() -> None:
    z = torch.randn(10, 6)
    pairs = torch.randint(10, (2, 7))

    logits = pairwise_logits(z, pairs, "tnbbeta", torch.tensor(1.0))

    assert logits.shape == (7,)
