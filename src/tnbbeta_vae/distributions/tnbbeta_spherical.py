"""Spherical extension of the TNBbeta distribution via a Householder lift.

Lederman & Schein (2026, arXiv:2606.11624) only define TNBbeta on the unit
interval (0, 1); the hyperspherical extension here is original to this
project. The construction mirrors how the von Mises-Fisher and Power
Spherical (De Cao & Aziz, 2020) distributions are built:

1. Sample a "latitude" ``w = 2*Y - 1 in [-1, 1]`` from a rescaled
   :class:`TNBBetaUnivariate`, representing the cosine similarity to a pole
   ``e_1 = (1, 0, ..., 0)``.
2. Sample a uniform direction ``v`` on ``S^(dim-2)`` (the sphere in the
   remaining ``dim - 1`` coordinates).
3. Combine into ``z = (w, sqrt(1-w^2) * v)``, a point on ``S^(dim-1)``
   whose polar angle from ``e_1`` has cosine ``w``.
4. Reflect ``z`` from the pole to the actual mean direction ``mu`` with a
   Householder reflection ``U(mu)`` satisfying ``U(mu) @ e_1 == mu``.

Because ``U(mu)`` is symmetric and an involution (``U(mu) @ U(mu) == I``),
``(U(mu) @ z)[0] == mu @ z`` for *any* ``z`` -- so ``log_prob`` only ever
needs a dot product, never an explicit reflection; the reflection is only
needed to generate samples.

Since TNBbeta's median/concentration/boundary parameters (p, q, epsilon)
carry over directly to the latitude ``w``, this inherits TNBbeta's full
range of boundary behavior (Proposition 3.1) -- including the boundary
divergence at epsilon < 1, which is not a defect here: it is exactly what
makes the induced spherical density smooth and finite exactly at the pole
(the (1 - w^2)^-((dim-3)/2) Jacobian below has to be canceled by a matching
singularity in the latitude density for generic dim, the same way
transforming a smooth density on a circle by y = cos(theta) forces a
1/sin(theta) blowup in y-space).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.types import _size

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate

__all__ = ["TNBBetaSpherical"]

_BOUNDARY_EPS = 1e-6
_HOUSEHOLDER_DEGENERACY_EPS = 1e-6


class _UnitSphere(constraints.Constraint):
    """Constrains to unit-norm vectors (points on a hypersphere)."""

    event_dim = 1

    def check(self, value: Tensor) -> Tensor:
        return torch.isclose(
            value.norm(dim=-1), torch.ones_like(value.norm(dim=-1)), atol=1e-4
        )


_unit_sphere = _UnitSphere()


class TNBBetaSpherical(Distribution):
    """TNBbeta lifted to the unit hypersphere S^(dim-1) via a Householder reflection.

    Rotationally symmetric around ``mean_direction``: the density depends
    on a sample ``z`` only through ``w = mean_direction @ z``, the cosine
    similarity to the mean direction, via a rescaled
    :class:`TNBBetaUnivariate` on ``w``. Inherits that distribution's
    median/concentration/boundary parameters (p, q, epsilon) directly,
    including its ability to be multimodal (Proposition 3.1) -- so, unlike
    von Mises-Fisher, this family can represent e.g. simultaneous
    concentration near ``mean_direction`` and its antipode.

    Attributes:
        mean_direction: Mean direction on S^(dim - 1), shape ``(..., dim)``.
            Normalized to unit norm in ``__init__`` if not already.
        p: Median of the latitude ``w``'s underlying TNBbeta, in (0, 1).
            ``p`` near 1 concentrates mass near ``mean_direction``.
        q: Concentration parameter, in (0, 1).
        epsilon: Boundary parameter, > 0.
    """

    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "mean_direction": constraints.real_vector,
        "p": constraints.interval(0.0, 1.0),
        "q": constraints.interval(0.0, 1.0),
        "epsilon": constraints.positive,
    }
    support = _unit_sphere  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    has_rsample = True

    def __init__(
        self,
        mean_direction: Tensor,
        p: Tensor | float,
        q: Tensor | float,
        epsilon: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the distribution.

        Args:
            mean_direction: Mean direction, shape ``(..., dim)`` with
                ``dim >= 2``. Normalized to unit norm.
            p: Median parameter of the latitude distribution, in (0, 1).
            q: Concentration parameter, in (0, 1).
            epsilon: Boundary parameter, > 0.
            validate_args: Whether to validate arguments and samples.

        Raises:
            ValueError: If ``mean_direction``'s last dimension is < 2.
        """
        if mean_direction.shape[-1] < 2:
            raise ValueError(
                "TNBBetaSpherical requires dim >= 2 (i.e. at least S^1); got "
                f"mean_direction.shape[-1] = {mean_direction.shape[-1]}."
            )
        self.mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)

        p_t, q_t, epsilon_t = broadcast_all(p, q, epsilon)
        batch_shape = torch.broadcast_shapes(self.mean_direction.shape[:-1], p_t.shape)
        self.p = p_t.expand(batch_shape)
        self.q = q_t.expand(batch_shape)
        self.epsilon = epsilon_t.expand(batch_shape)

        event_shape = self.mean_direction.shape[-1:]
        super().__init__(
            batch_shape=torch.Size(batch_shape),
            event_shape=torch.Size(event_shape),
            validate_args=validate_args,
        )

    @property
    def dim(self) -> int:
        """Ambient dimension of the sphere (mean_direction has `dim` components)."""
        return self.event_shape[0]

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density at ``value``.

        Args:
            value: Points on S^(dim - 1), shape ``(..., dim)``.

        Returns:
            Log-density, broadcast against the distribution's batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)

        w = (self.mean_direction * value).sum(-1)
        w = w.clamp(-1.0 + _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)
        y = (w + 1) / 2

        radial_log_prob = TNBBetaUnivariate(self.p, self.q, self.epsilon).log_prob(
            y
        ) - math.log(2)
        jacobian = -((self.dim - 3) / 2) * torch.log1p(-(w**2))
        log_area = _log_surface_area(self.dim - 1)

        return radial_log_prob + jacobian - log_area

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:  # noqa: B008
        """Draws reparameterized samples via the pole-then-reflect construction.

        Args:
            sample_shape: Shape of the i.i.d. sample batch to draw.

        Returns:
            Unit-norm samples in S^(dim - 1), shape
            ``sample_shape + batch_shape + (dim,)``, differentiable w.r.t.
            ``mean_direction``, ``p``, ``q``, and ``epsilon``.
        """
        shape = self._extended_shape(sample_shape)
        batch_shape, dim = shape[:-1], shape[-1]

        p = self.p.expand(batch_shape)
        q = self.q.expand(batch_shape)
        epsilon = self.epsilon.expand(batch_shape)
        mean_direction = self.mean_direction.expand(shape)

        y = TNBBetaUnivariate(p, q, epsilon).rsample()
        w = 2 * y - 1

        g = torch.randn((*batch_shape, dim - 1), dtype=w.dtype, device=w.device)
        v = g / g.norm(dim=-1, keepdim=True)

        radius = torch.sqrt((1 - w**2).clamp_min(0)).unsqueeze(-1)
        z = torch.cat([w.unsqueeze(-1), radius * v], dim=-1)

        return _householder_reflect(z, mean_direction)


def _log_surface_area(ambient_dim: int) -> float:
    """Log surface area of the unit sphere S^(ambient_dim - 1) in R^ambient_dim."""
    return (
        math.log(2)
        + (ambient_dim / 2) * math.log(math.pi)
        - math.lgamma(ambient_dim / 2)
    )


def _householder_reflect(z: Tensor, mean_direction: Tensor) -> Tensor:
    """Reflects ``z`` from the pole ``e_1`` to ``mean_direction`` (or back).

    Uses the Householder reflection ``U = I - 2*u*u^T`` with
    ``u = (e_1 - mean_direction) / ||e_1 - mean_direction||``, which
    satisfies ``U @ e_1 == mean_direction``. ``U`` is symmetric and its own
    inverse, so this same function also maps ``mean_direction``-frame
    vectors back to the pole frame.

    Args:
        z: Points on the sphere, shape ``(..., dim)``.
        mean_direction: Unit vectors, shape broadcastable to ``z``.

    Returns:
        Reflected points, same shape as ``z``.
    """
    pole = torch.zeros_like(mean_direction)
    pole[..., 0] = 1.0

    diff = pole - mean_direction
    diff_norm = diff.norm(dim=-1, keepdim=True)
    # mean_direction == pole (diff_norm == 0) needs no reflection; U -> I.
    u = diff / diff_norm.clamp_min(_HOUSEHOLDER_DEGENERACY_EPS)

    reflected = z - 2 * (u * z).sum(-1, keepdim=True) * u
    return torch.where(diff_norm > _HOUSEHOLDER_DEGENERACY_EPS, reflected, z)
