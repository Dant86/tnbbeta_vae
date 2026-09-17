"""A fixed, non-learnable TNBBetaSpherical prior.

Mirrors the role a standard N(0, I) prior plays in a vanilla VAE: fixed,
not optimized, serving as a stable reference the encoder's posterior is
regularized toward via the ELBO's KL term.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import TNBBetaSpherical

__all__ = ["FixedTNBBetaSphericalPrior"]


class FixedTNBBetaSphericalPrior(nn.Module):
    """A TNBBetaSpherical prior with fixed (non-learnable) parameters.

    The mean direction is fixed at the canonical pole ``e_1 = (1, 0, ...,
    0)``: before training, the latent space has no privileged orientation,
    so any fixed unit vector serves equally well as this reference point.

    Attributes:
        mean_direction: The fixed pole, registered as a buffer so it
            follows the module across `.to(device)` calls.
        p: Median of the prior's latitude distribution, in (0, 1).
        q: Concentration parameter, in (0, 1).
        epsilon: Boundary parameter, > 0.
    """

    mean_direction: Tensor  # declared so pyright doesn't fall back to
    # nn.Module.__getattr__'s Tensor | Module return type for this buffer.

    def __init__(self, dim: int, p: float, q: float, epsilon: float) -> None:
        """Initializes the prior.

        Args:
            dim: Ambient dimension of the sphere S^(dim - 1).
            p: Median parameter, in (0, 1).
            q: Concentration parameter, in (0, 1).
            epsilon: Boundary parameter, > 0.
        """
        super().__init__()
        mean_direction = torch.zeros(dim)
        mean_direction[0] = 1.0
        self.register_buffer("mean_direction", mean_direction)
        self.p = p
        self.q = q
        self.epsilon = epsilon

    def forward(self) -> TNBBetaSpherical:
        """Builds the (fixed-parameter) prior distribution.

        Returns:
            A :class:`TNBBetaSpherical` with this module's parameters.
        """
        return TNBBetaSpherical(self.mean_direction, self.p, self.q, self.epsilon)
