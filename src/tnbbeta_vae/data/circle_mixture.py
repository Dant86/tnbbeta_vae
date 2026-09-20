"""Noisy embeddings of a mixture of von Mises distributions on the circle.

The synthetic data of S-VAE paper section 5.1: angles are drawn from a mixture of
three von Mises distributions on S^1, mapped to the plane and then into R^100 by a
fixed random non-linear transformation, and observed with Gaussian noise. A model
with a circular latent should recover the angle; a Gaussian latent cannot both
match its N(0, I) prior and keep the circular structure.

The paper does not give the exact transformation, so the choices here (a fixed
random two-layer tanh network, noise standard deviation 0.05, three components at
0, 2pi/3 and 4pi/3 with concentration 20) are ours.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import VonMises

__all__ = ["CircleMixtureData", "circle_mixture_data"]

_NUM_COMPONENTS = 3
_CONCENTRATION = 20.0
_HIDDEN = 64


class CircleMixtureData:
    """A fixed embedding of the circle into R^``ambient_dim`` plus a sampler.

    Attributes:
        ambient_dim: Dimension of the observed vectors.
        noise_std: Standard deviation of the Gaussian observation noise.
    """

    def __init__(
        self, ambient_dim: int = 100, noise_std: float = 0.05, seed: int = 0
    ) -> None:
        """Draws the fixed embedding weights.

        Args:
            ambient_dim: Dimension of the observed vectors.
            noise_std: Standard deviation of the observation noise.
            seed: Seed for the embedding weights (not for :meth:`sample`).
        """
        generator = torch.Generator().manual_seed(seed)
        self.ambient_dim = ambient_dim
        self.noise_std = noise_std
        self._w1 = torch.randn(2, _HIDDEN, generator=generator)
        self._w2 = torch.randn(_HIDDEN, ambient_dim, generator=generator) / math.sqrt(
            _HIDDEN
        )

    def embed(self, angle: torch.Tensor) -> torch.Tensor:
        """Maps angles to noise-free points in R^``ambient_dim``.

        Args:
            angle: Angles in radians, shape ``(n,)``.

        Returns:
            Tensor of shape ``(n, ambient_dim)``.
        """
        circle = torch.stack([angle.cos(), angle.sin()], dim=-1)
        return torch.tanh(circle @ self._w1) @ self._w2

    def sample(
        self, num: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draws noisy observations.

        Args:
            num: Number of points.
            generator: Optional RNG for the angles, component choice and noise.

        Returns:
            ``(x, angle, component)``: observations ``(num, ambient_dim)``, the
            true angles in (-pi, pi] and the mixture component index of each.
        """
        component = torch.randint(_NUM_COMPONENTS, (num,), generator=generator)
        centres = component.float() * (2 * math.pi / _NUM_COMPONENTS)
        # VonMises has no generator argument, so seed the global RNG from ours.
        if generator is not None:
            torch.manual_seed(int(torch.randint(2**31, (1,), generator=generator)))
        angle = VonMises(centres, torch.tensor(_CONCENTRATION)).sample()
        noise = self.noise_std * torch.randn(num, self.ambient_dim, generator=generator)
        return self.embed(angle) + noise, angle, component


def circle_mixture_data(
    ambient_dim: int = 100, noise_std: float = 0.05, seed: int = 0
) -> CircleMixtureData:
    """Builds the dataset object (see :class:`CircleMixtureData`)."""
    return CircleMixtureData(ambient_dim, noise_std, seed)
