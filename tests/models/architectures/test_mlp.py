"""Tests for tnbbeta_vae.models.architectures.mlp."""

from __future__ import annotations

import torch
from torch import nn

from tnbbeta_vae.models.architectures.mlp import mlp_stack


def test_mlp_stack_has_a_linear_relu_pair_per_consecutive_size() -> None:
    stack = mlp_stack([4, 8, 3])

    linears = [layer for layer in stack if isinstance(layer, nn.Linear)]
    relus = [layer for layer in stack if isinstance(layer, nn.ReLU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (4, 8),
        (8, 3),
    ]
    assert len(relus) == 2


def test_mlp_stack_output_shape() -> None:
    stack = mlp_stack([5, 16, 8])
    x = torch.randn(3, 5)

    out = stack(x)

    assert out.shape == (3, 8)
