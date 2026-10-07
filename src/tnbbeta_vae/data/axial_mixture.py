"""Noisy embeddings of an axial (sign-ambiguous) von Mises mixture on the circle.

A sibling of :mod:`tnbbeta_vae.data.circle_mixture`'s S-VAE-paper replication,
built to need the opposite of what that task needs: instead of a *directed*
angle, the true latent here is an *axis* -- ``phi`` and ``phi + pi`` are
equally likely a priori and genuinely indistinguishable from the
observation, by construction. The angle is drawn from a 50/50 mixture of
two von Mises components at ``0`` and ``pi`` (same concentration), then
embedded through :meth:`CircleMixtureData.embed` on the *doubled* angle:
``embed(2*phi) == embed(2*(phi+pi))`` exactly, since ``2*phi`` and
``2*phi + 2*pi`` are the same angle. That reuses the existing fixed random
embedding network unchanged; only the generative distribution over the
angle and the doubling are new.

The Bayes-optimal posterior over ``phi`` given an observation is therefore
exactly bimodal with equal mass at the two components -- the shape a single
``TNBBetaSpherical`` component can represent (via ``epsilon`` below the
uniform-prior threshold) and no other latent family in this project can.
See
``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import VonMises

from tnbbeta_vae.data.circle_mixture import CircleMixtureData

__all__ = ["AxialMixtureData", "axial_mixture_data"]

_CONCENTRATION = 20.0


class AxialMixtureData:
    """A fixed axial embedding of the circle into R^``ambient_dim`` plus a sampler.

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
            seed: Seed for the embedding weights (not for :meth:`sample`),
                forwarded to
                :class:`~tnbbeta_vae.data.circle_mixture.CircleMixtureData`.
        """
        self.ambient_dim = ambient_dim
        self.noise_std = noise_std
        self._circle = CircleMixtureData(ambient_dim, noise_std, seed)

    def embed(self, angle: torch.Tensor) -> torch.Tensor:
        """Maps angles to noise-free points, invariant under ``angle + pi``.

        Args:
            angle: Angles in radians, shape ``(n,)``.

        Returns:
            Tensor of shape ``(n, ambient_dim)``. Equal for ``angle`` and
            ``angle + pi`` (reuses
            :meth:`~tnbbeta_vae.data.circle_mixture.CircleMixtureData.embed`
            on the doubled angle, which is exactly periodic in ``pi``).
        """
        return self._circle.embed(2 * angle)

    def sample(
        self, num: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draws noisy observations.

        Args:
            num: Number of points.
            generator: Optional RNG for the component, angle and noise.

        Returns:
            ``(x, angle, component)``: observations ``(num, ambient_dim)``,
            the true angles in ``(-pi, pi]`` and which of the two axial
            components (0 or 1, for the von Mises centred at 0 or at pi)
            produced it. ``component`` is never recoverable from ``x``
            alone by construction -- that is the point of the task.
        """
        component = torch.randint(2, (num,), generator=generator)
        centres = component.float() * math.pi
        # VonMises has no generator argument, so seed the global RNG from
        # ours (same workaround as CircleMixtureData.sample).
        if generator is not None:
            torch.manual_seed(int(torch.randint(2**31, (1,), generator=generator)))
        angle = VonMises(centres, torch.tensor(_CONCENTRATION)).sample()
        noise = self.noise_std * torch.randn(num, self.ambient_dim, generator=generator)
        return self.embed(angle) + noise, angle, component


def axial_mixture_data(
    ambient_dim: int = 100, noise_std: float = 0.05, seed: int = 0
) -> AxialMixtureData:
    """Builds the dataset object (see :class:`AxialMixtureData`)."""
    return AxialMixtureData(ambient_dim, noise_std, seed)
