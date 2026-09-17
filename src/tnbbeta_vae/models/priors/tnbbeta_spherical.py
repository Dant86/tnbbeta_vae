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


def uniform_prior_params(dim: int) -> tuple[float, float, float]:
    """Returns (p, q, epsilon) making TNBBetaSpherical exactly Uniform(S^(dim - 1)).

    Derivation: the spherical density factors as ``f_W(w) /
    [Area(S^(dim-2)) * (1-w^2)^((dim-3)/2)]`` (see this module's sibling
    :mod:`tnbbeta_vae.distributions.tnbbeta_spherical`), so it's constant
    in ``z`` iff ``f_W(w)`` is proportional to ``(1-w^2)^((dim-3)/2)``. At
    ``q=0`` and ``p=0.5``, ``TNBBetaUnivariate(0.5, 0, eps)`` reduces to
    plain ``Beta(eps, eps)`` (Corollary 3.1 of the source paper,
    specialized to ``p=0.5`` so the LNbeta tilt parameter ``(1-p)/p``
    equals 1), whose shifted density ``W = 2Y-1`` is proportional to
    ``(1-w^2)^(eps-1)``. Matching exponents gives ``eps = (dim-1)/2``.
    ``mean_direction`` becomes irrelevant at ``q=0`` -- there's no
    directional concentration left to point anywhere.

    Unlike a concentrated prior, matching this one gives an encoder no
    cheap way to "collapse": since it has no informative direction, a
    posterior that collapsed toward it would have to become uniform
    itself -- i.e. produce near-random output regardless of ``x`` -- which
    is a far worse reconstruction trade than collapsing toward a
    concentrated prior's single point.

    Args:
        dim: Ambient dimension of the sphere S^(dim - 1).

    Returns:
        ``(p, q, epsilon) = (0.5, 0.0, (dim - 1) / 2)``.
    """
    return 0.5, 0.0, (dim - 1) / 2
