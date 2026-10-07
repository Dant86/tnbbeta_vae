"""A small MLP that predicts a clean sphere point from a noised one.

latent_dim is a small vector (project default 8), not spatial, so this is
a plain MLP conditioned on a sinusoidal time embedding -- no convolutions,
unlike models/architectures/conv.py's image encoder/decoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["SphereDenoiserMLP"]


class SphereDenoiserMLP(nn.Module):
    """Predicts a unit-norm z_0 estimate from a noised z_t and timestep t."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        time_embed_dim: int = 64,
    ) -> None:
        """Initializes the network.

        Args:
            latent_dim: Ambient dimension of the sphere S^(latent_dim - 1).
            hidden_dim: Width of each hidden layer.
            depth: Number of hidden layers.
            time_embed_dim: Dimension of the sinusoidal time embedding fed
                in alongside z.
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        layers: list[nn.Module] = [
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """Predicts a unit-norm z_0 estimate.

        Args:
            z: Noised points, shape ``(..., latent_dim)``.
            t: Diffusion time, shape ``(...)`` matching ``z``'s batch dims.

        Returns:
            Unit-norm predictions, same shape as ``z``.
        """
        embed = _sinusoidal_embedding(t, self.time_embed_dim)
        prediction = self.net(torch.cat([z, embed], dim=-1))
        return prediction / prediction.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard transformer-style sinusoidal embedding of a scalar time.

    Args:
        t: Shape ``(...)``.
        dim: Output embedding dimension; must be even.

    Returns:
        Shape ``(..., dim)``.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    angles = t.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
